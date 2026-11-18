package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.interp.AttentionRollout;
import io.github.zzuegg.brainiac.experimental.nn.Attribution;
import io.github.zzuegg.brainiac.experimental.nn.ConceptActivationVectors;
import io.github.zzuegg.brainiac.experimental.nn.GradCam;
import io.github.zzuegg.brainiac.experimental.train.ModelAgnosticExplainer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Reshape;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Explanations scored against a planted ground truth.
 *
 * <p><b>Tabular</b> ({@link #tabular}): eight standard-normal features, the label
 * {@code x0 + x1 + 1.5 x2 x3 > 0}, so features 0-3 matter and 4-7 never do. An MLP and
 * {@link GradientBoostedTrees} are explained on 40 test rows by KernelSHAP, LIME
 * ({@link ModelAgnosticExplainer}), integrated gradients, SmoothGrad and saliency
 * ({@link Attribution} of the class-1 logit, MLP only), and random attributions. Scores: <em>recovery</em>, the
 * share of the four highest-|attribution| features that are relevant; and the
 * <em>deletion AUC</em>, the mean predicted-class probability while the features are reset
 * to their mean (0) in order of |attribution| (lower is better). Counterfactuals
 * ({@link ModelAgnosticExplainer#counterfactual}) are scored by validity, features changed
 * and the share of changed features that are relevant.
 *
 * <p><b>Images</b> ({@link #images}): 12x12 images of a square (class 0) or a plus
 * (class 1) at a random place, and a spurious 2x2 marker in the top-left corner. A CNN
 * trained where the marker is independent of the class is explained by Grad-CAM, saliency
 * and integrated gradients: the <em>pointing game</em> counts an image as a hit when the
 * map's maximum lies on the shape (its 3x3 box, one pixel of tolerance). A second CNN
 * trained where the marker comes with class 1 95% of the time shows the shortcut: TCAV
 * scores ({@link ConceptActivationVectors}) of the "marker" concept for class 1, against
 * the same score of the clean CNN and of random concepts, and how often each map points
 * at the marker.
 */
public final class ExplainabilityBenchmark {

    private ExplainabilityBenchmark() {
    }

    // ------------------------------------------------------------------ tabular

    private static final int D = 8;
    private static final int RELEVANT = 4;

    /** The tabular attribution methods. */
    public enum TabularMethod {
        /** KernelSHAP, all 254 coalitions, 20 background rows. */
        KERNEL_SHAP,
        /** LIME, 500 perturbations. */
        LIME,
        /** Integrated gradients from the zero baseline, 32 steps (differentiable models only). */
        INTEGRATED_GRADIENTS,
        /** SmoothGrad, 32 samples at sigma 0.3 (differentiable models only). */
        SMOOTH_GRAD,
        /** The plain gradient (differentiable models only). */
        SALIENCY,
        /** The boosted trees' global split-gain importance (trees only). */
        TREE_GAIN,
        /** Random attributions. */
        RANDOM
    }

    /**
     * Attribution quality.
     *
     * @param recovery the share of the top-4 features that are relevant (random: 0.5)
     * @param deletionAuc the mean predicted-class probability along the deletion curve, lower is better
     */
    public record TabularScores(double recovery, double deletionAuc) {
    }

    /**
     * Counterfactual quality.
     *
     * @param validity the share of rows whose prediction the counterfactual flips
     * @param featuresChanged the mean number of features changed, among valid ones
     * @param relevantShare the share of changed features that are relevant
     * @param distance the mean L1 distance of valid counterfactuals
     */
    public record CounterfactualScores(double validity, double featuresChanged, double relevantShare,
            double distance) {
    }

    private static double label(double[] x) {
        return x[0] + x[1] + 1.5 * x[2] * x[3] > 0 ? 1 : 0;
    }

    private static double[][] rows(int n, SplittableRandom r) {
        double[][] x = new double[n][D];
        for (double[] row : x) {
            for (int j = 0; j < D; j++) {
                row[j] = r.nextGaussian();
            }
        }
        return x;
    }

    private static Tensor tensor(double[][] x) {
        float[] v = new float[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                v[i * x[i].length + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, x[0].length), v);
    }

    private static Var nll(Var probs, Tensor ids) {
        int n = ids.shape().dim(0);
        Tensor oneHot = ids.reshape(n, 1).equalTo(Tensor.arange(2).reshape(1, 2));
        return probs.add(1e-7f).log().mul(oneHot).sum(1, false).mean().neg();
    }

    private static Model trainMlp(double[][] x, double[] y, long seed) {
        Model m = Model.sequential().input(D).dense(32, Activation.RELU).dense(32, Activation.RELU)
                .dense(2, Activation.IDENTITY).seed(seed).build();
        Tensor tx = tensor(x);
        float[] ids = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            ids[i] = (float) y[i];
        }
        Tensor ty = Tensor.of(Shape.of(y.length), ids);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        for (int s = 0; s < 1500; s++) {
            int from = r.nextInt(n - 100);
            Functional.crossEntropy(m.forward(Var.of(tx.slice(0, from, from + 100)), true),
                    ty.slice(0, from, from + 100)).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /** The MLP's class-1 probability, as a batch function. */
    private static Function<double[][], double[]> probability(Model m) {
        return rows -> {
            double[] v = Activation.SOFTMAX.forward(m.forward(Var.of(tensor(rows)))).value().toDoubleArray();
            double[] out = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                out[i] = v[2 * i + 1];
            }
            return out;
        };
    }

    /** The MLP's class-1 log-odds, the logit difference, as a batch function. */
    private static Function<double[][], double[]> logOdds(Model m) {
        return rows -> {
            double[] v = m.forward(Var.of(tensor(rows))).value().toDoubleArray();
            double[] out = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                out[i] = v[2 * i + 1] - v[2 * i];
            }
            return out;
        };
    }

    /**
     * Scores one attribution method on the tabular task.
     *
     * @param method the method
     * @param trees whether to explain gradient-boosted trees instead of the MLP
     * @param seed the seed
     * @return the scores
     */
    public static TabularScores tabular(TabularMethod method, boolean trees, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] train = rows(2000, r);
        double[] y = Arrays.stream(train).mapToDouble(ExplainabilityBenchmark::label).toArray();
        double[][] test = rows(40, r);
        double[][] background = Arrays.copyOf(train, 20);
        Function<double[][], double[]> f;
        Model mlp = null;
        GradientBoostedTrees gbt = null;
        if (trees) {
            gbt = GradientBoostedTrees.fit(train, y, GradientBoostedTrees.Objective.BINARY,
                    GradientBoostedTrees.Settings.defaults().withTrees(150));
            f = gbt::predict;
        } else {
            mlp = trainMlp(train, y, seed);
            f = probability(mlp);
        }
        ModelAgnosticExplainer explainer = new ModelAgnosticExplainer(f);
        double[] scale = new double[D];
        Arrays.fill(scale, 1);
        double recovery = 0;
        double deletion = 0;
        for (double[] x : test) {
            float[] xf = new float[D];
            for (int j = 0; j < D; j++) {
                xf[j] = (float) x[j];
            }
            double[] a = switch (method) {
                case KERNEL_SHAP -> explainer.kernelShap(x, background, 300, r);
                case LIME -> explainer.lime(x, scale, 500, 0.75 * Math.sqrt(D), r);
                case INTEGRATED_GRADIENTS -> doubles(Attribution.integratedGradients(gradientModel(mlp), xf,
                        new float[D], 1, 32));
                case SMOOTH_GRAD -> doubles(Attribution.smoothGrad(gradientModel(mlp), xf, 1, 32, 0.3f, r.nextLong()));
                case SALIENCY -> doubles(Attribution.saliency(gradientModel(mlp), xf, 1));
                case TREE_GAIN -> {
                    if (gbt == null) {
                        throw new IllegalArgumentException("TREE_GAIN needs trees");
                    }
                    yield gbt.featureImportance();
                }
                case RANDOM -> r.doubles(D).toArray();
            };
            int[] order = byMagnitude(a);
            int hits = 0;
            for (int k = 0; k < RELEVANT; k++) {
                hits += order[k] < RELEVANT ? 1 : 0;
            }
            recovery += hits / (double) RELEVANT / test.length;
            deletion += deletionAuc(f, x, order) / test.length;
        }
        return new TabularScores(recovery, deletion);
    }

    private static Model gradientModel(Model mlp) {
        if (mlp == null) {
            throw new IllegalArgumentException("gradient attributions need the MLP");
        }
        return mlp;
    }

    private static double[] doubles(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }

    private static int[] byMagnitude(double[] a) {
        return IntStream.range(0, a.length).boxed()
                .sorted((i, j) -> Double.compare(Math.abs(a[j]), Math.abs(a[i]))).mapToInt(Integer::intValue)
                .toArray();
    }

    private static double deletionAuc(Function<double[][], double[]> f, double[] x, int[] order) {
        double[][] steps = new double[D + 1][];
        double[] cur = x.clone();
        steps[0] = cur.clone();
        for (int k = 0; k < D; k++) {
            cur[order[k]] = 0;
            steps[k + 1] = cur.clone();
        }
        double[] p = f.apply(steps);
        boolean positive = p[0] > 0.5;
        double auc = 0;
        for (double v : p) {
            auc += (positive ? v : 1 - v) / p.length;
        }
        return auc;
    }

    /**
     * Scores counterfactual explanations on the tabular task.
     *
     * @param trees whether to explain gradient-boosted trees instead of the MLP
     * @param seed the seed
     * @return the scores
     */
    public static CounterfactualScores counterfactuals(boolean trees, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] train = rows(2000, r);
        double[] y = Arrays.stream(train).mapToDouble(ExplainabilityBenchmark::label).toArray();
        double[][] test = rows(40, r);
        Function<double[][], double[]> f;
        if (trees) {
            GradientBoostedTrees gbt = GradientBoostedTrees.fit(train, y, GradientBoostedTrees.Objective.BINARY,
                    GradientBoostedTrees.Settings.defaults().withTrees(150));
            f = rows -> Arrays.stream(gbt.predict(rows)).map(p -> Math.log(p / (1 - p))).toArray();
        } else {
            f = logOdds(trainMlp(train, y, seed));
        }
        ModelAgnosticExplainer explainer = new ModelAgnosticExplainer(f);
        double[] scale = new double[D];
        Arrays.fill(scale, 1);
        int valid = 0;
        double changed = 0;
        double relevant = 0;
        int changedTotal = 0;
        double distance = 0;
        for (double[] x : test) {
            ModelAgnosticExplainer.Counterfactual c = explainer.counterfactual(x, scale, 0, 1, 100, trees ? 0.3 : 1e-3);
            if (!c.valid()) {
                continue;
            }
            valid++;
            changed += c.features().length;
            for (int j : c.features()) {
                relevant += j < RELEVANT ? 1 : 0;
                changedTotal++;
                distance += Math.abs(c.changed()[j] - x[j]);
            }
        }
        return new CounterfactualScores(valid / (double) test.length, valid == 0 ? 0 : changed / valid,
                changedTotal == 0 ? 0 : relevant / changedTotal, valid == 0 ? 0 : distance / valid);
    }

    // ------------------------------------------------------------------ images

    private static final int SIDE = 12;
    private static final int PIXELS = SIDE * SIDE;
    private static final int CONV = 2;   // the second convolution's step: input reshape, conv, conv

    /** The image attribution maps. */
    public enum ImageMethod {
        /** Grad-CAM on the second convolution. */
        GRAD_CAM,
        /** |gradient|. */
        SALIENCY,
        /** |integrated gradients| from a black image. */
        INTEGRATED_GRADIENTS,
        /** A uniformly random pixel. */
        RANDOM
    }

    /**
     * Image explanation scores.
     *
     * @param pointingClean pointing-game accuracy on the clean CNN
     * @param markerPointingSpurious the share of class-1 images with a marker where the
     *        spurious CNN's map peaks at the marker
     */
    public record ImageScores(double pointingClean, double markerPointingSpurious) {
    }

    /**
     * TCAV results for the "marker" concept and class 1.
     *
     * @param spurious the TCAV score on the CNN trained with the shortcut
     * @param clean the score on the CNN trained without it
     * @param randomMean the mean score of random concepts (on the spurious CNN)
     * @param randomStd their standard deviation
     * @param spuriousZ the spurious CNN's concept sensitivity in standard deviations of its
     *        random concepts' sensitivities
     * @param cleanZ the same for the clean CNN
     * @param shortcutReliance how much class-1 probability the marker adds on class-1
     *        images of the spurious CNN (measured directly by removing it)
     */
    public record TcavScores(double spurious, double clean, double randomMean, double randomStd, double spuriousZ,
            double cleanZ, double shortcutReliance) {
    }

    /**
     * The image study.
     *
     * @param maps the scores of each attribution map
     * @param tcav the TCAV results
     */
    public record ImageStudy(Map<ImageMethod, ImageScores> maps, TcavScores tcav) {
    }

    private record Images(Tensor x, Tensor y, int[] row, int[] col, boolean[] marker) {
    }

    private static Images images(int n, double markerGivenClass1, double markerGivenClass0, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] img = new float[n * PIXELS];
        float[] ids = new float[n];
        int[] row = new int[n];
        int[] col = new int[n];
        boolean[] marker = new boolean[n];
        for (int i = 0; i < n; i++) {
            int c = r.nextInt(2);
            ids[i] = c;
            int y = 3 + r.nextInt(SIDE - 5);   // the shape's top-left, away from the marker
            int x = 3 + r.nextInt(SIDE - 5);
            row[i] = y;
            col[i] = x;
            int o = i * PIXELS;
            float b = 0.7f + 0.6f * r.nextFloat();
            for (int k = 0; k < 3; k++) {
                if (c == 0) {
                    img[o + y * SIDE + x + k] = b;
                    img[o + (y + 2) * SIDE + x + k] = b;
                    img[o + (y + k) * SIDE + x] = b;
                    img[o + (y + k) * SIDE + x + 2] = b;
                } else {
                    img[o + (y + 1) * SIDE + x + k] = b;
                    img[o + (y + k) * SIDE + x + 1] = b;
                }
            }
            marker[i] = r.nextDouble() < (c == 1 ? markerGivenClass1 : markerGivenClass0);
            if (marker[i]) {
                for (int p : new int[] {0, 1, SIDE, SIDE + 1}) {
                    img[o + p] = 1f;
                }
            }
            for (int p = 0; p < PIXELS; p++) {
                img[o + p] += (float) (0.1 * r.nextGaussian());
            }
        }
        return new Images(Tensor.of(Shape.of(n, PIXELS), img), Tensor.of(Shape.of(n), ids), row, col, marker);
    }

    private static Model trainCnn(Images d, long seed) {
        Model m = Model.sequential().input(PIXELS).layer(new Reshape(1, SIDE, SIDE)).conv2d(8, 3, Activation.RELU)
                .conv2d(8, 3, Activation.RELU).flatten().dense(2, Activation.SOFTMAX).seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        int n = d.x().shape().dim(0);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < 300; s++) {
            int from = r.nextInt(n - 50);
            nll(m.forward(Var.of(d.x().slice(0, from, from + 50)), true), d.y().slice(0, from, from + 50)).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    private static float[] row(Tensor x, int i) {
        return x.slice(0, i, i + 1).toArray();
    }

    private static int argmax(float[] map) {
        int best = 0;
        for (int i = 1; i < map.length; i++) {
            if (map[i] > map[best]) {
                best = i;
            }
        }
        return best;
    }

    private static float[] map(ImageMethod method, Model m, Tensor x, int i, int cls, SplittableRandom r) {
        float[] in = row(x, i);
        return switch (method) {
            case GRAD_CAM -> GradCam.upsample(GradCam.map(m, CONV, x.slice(0, i, i + 1), cls), SIDE, 1).toArray();
            case SALIENCY -> abs(Attribution.saliency(m, in, cls));
            case INTEGRATED_GRADIENTS -> abs(Attribution.integratedGradients(m, in, new float[PIXELS], cls, 16));
            case RANDOM -> {
                float[] v = new float[PIXELS];
                v[r.nextInt(PIXELS)] = 1;
                yield v;
            }
        };
    }

    private static float[] abs(float[] v) {
        for (int i = 0; i < v.length; i++) {
            v[i] = Math.abs(v[i]);
        }
        return v;
    }

    /**
     * Trains the clean and the spurious CNN and scores every attribution map and TCAV.
     *
     * @param seed the seed
     * @return the scores
     */
    public static ImageStudy images(long seed) {
        Model clean = trainCnn(images(1500, 0.5, 0.5, seed), seed);
        Model spurious = trainCnn(images(1500, 0.95, 0.05, seed + 1), seed);
        Images test = images(200, 0.5, 0.5, seed + 2);
        Map<ImageMethod, ImageScores> maps = new EnumMap<>(ImageMethod.class);
        for (ImageMethod method : ImageMethod.values()) {
            maps.put(method, pointing(method, clean, spurious, test, seed));
        }
        return new ImageStudy(maps, tcav(clean, spurious, seed));
    }

    private static ImageScores pointing(ImageMethod method, Model clean, Model spurious, Images test, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int hits = 0;
        int markerHits = 0;
        int markerCount = 0;
        float[] ids = test.y().toArray();
        for (int i = 0; i < ids.length; i++) {
            int cls = (int) ids[i];
            int peak = argmax(map(method, clean, test.x(), i, cls, r));
            int py = peak / SIDE;
            int px = peak % SIDE;
            if (py >= test.row()[i] - 1 && py <= test.row()[i] + 3 && px >= test.col()[i] - 1
                    && px <= test.col()[i] + 3) {
                hits++;
            }
            if (cls == 1 && test.marker()[i]) {
                markerCount++;
                int sp = argmax(map(method, spurious, test.x(), i, cls, r));
                if (sp / SIDE <= 2 && sp % SIDE <= 2) {
                    markerHits++;
                }
            }
        }
        return new ImageScores(hits / (double) ids.length, markerHits / (double) markerCount);
    }

    private static double[] zAgainstRandom(Model model, Tensor withMarker, Tensor without, Tensor pool, Tensor inputs,
            long seed) {
        float[] cav = ConceptActivationVectors.cav(model, CONV, withMarker, without);
        double score = ConceptActivationVectors.score(model, CONV, cav, inputs, 1);
        double sens = ConceptActivationVectors.sensitivity(model, CONV, cav, inputs, 1);
        List<float[]> random = ConceptActivationVectors.randomCavs(model, CONV, pool, 8, seed);
        double[] rs = new double[random.size()];
        double[] rScore = new double[random.size()];
        for (int k = 0; k < rs.length; k++) {
            rs[k] = ConceptActivationVectors.sensitivity(model, CONV, random.get(k), inputs, 1);
            rScore[k] = ConceptActivationVectors.score(model, CONV, random.get(k), inputs, 1);
        }
        double mean = Arrays.stream(rs).average().orElse(0);
        double sd = Math.sqrt(Arrays.stream(rs).map(v -> (v - mean) * (v - mean)).average().orElse(0));
        double sMean = Arrays.stream(rScore).average().orElse(0);
        double sSd = Math.sqrt(Arrays.stream(rScore).map(v -> (v - sMean) * (v - sMean)).average().orElse(0));
        return new double[] {score, (sens - mean) / Math.max(sd, 1e-12), sMean, sSd};
    }

    private static TcavScores tcav(Model clean, Model spurious, long seed) {
        Tensor withMarker = images(100, 1, 1, seed + 3).x();
        Tensor without = images(100, 0, 0, seed + 4).x();
        Tensor pool = images(100, 0.5, 0.5, seed + 6).x();
        Images class1 = images(400, 0, 0, seed + 5);
        Tensor marked = images(400, 1, 1, seed + 5).x();   // the same images with the marker
        float[] y = class1.y().toArray();
        int[] ones = IntStream.range(0, y.length).filter(i -> y[i] == 1).toArray();
        Tensor inputs = DataPruner.select(class1.x(), ones);
        double[] s = zAgainstRandom(spurious, withMarker, without, pool, inputs, seed);
        double[] c = zAgainstRandom(clean, withMarker, without, pool, inputs, seed);
        double pWith = spurious.forward(Var.of(DataPruner.select(marked, ones))).value().slice(1, 1, 2).mean().item();
        double pWithout = spurious.forward(Var.of(inputs)).value().slice(1, 1, 2).mean().item();
        return new TcavScores(s[0], c[0], s[2], s[3], s[1], c[1], pWith - pWithout);
    }

    // ------------------------------------------------------------------ attention (wave 9 item 9.4.6)

    /** Token-attribution methods for the attention copy task. */
    public enum AttentionMethod {
        /** The head-averaged attention of the last layer. */
        RAW_ATTENTION,
        /** Attention rollout ({@link AttentionRollout#rollout}). */
        ROLLOUT,
        /** Attention flow ({@link AttentionRollout#flow}). */
        FLOW,
        /** Gradient-weighted rollout ({@link AttentionRollout#gradientRollout}). */
        GRADIENT_ROLLOUT,
        /** The norm of the logit's gradient with respect to each input embedding. */
        SALIENCY,
        /** Integrated gradients on the input embeddings (zero baseline, 16 steps). */
        INTEGRATED_GRADIENTS
    }

    /**
     * Scores of a token-attribution method.
     *
     * @param agreement the share of sequences whose top-attributed position is the copied source
     * @param deletionAuc the mean predicted-class probability while positions are replaced by
     *     padding in attribution order (lower is more faithful)
     * @param millis time per sequence
     */
    public record AttentionScores(double agreement, double deletionAuc, double millis) {
    }

    private static final int SEQ = 8;
    private static final int VALUES = 5;
    private static final int MARKER = 5;
    private static final int QUERY = 6;
    private static final int PAD = 7;

    /** A two-layer, two-head attention model whose attention maps and their gradients are recorded. */
    private static final class CopyAttention {
        final int dim = 16;
        final int heads = 2;
        final Var embed;
        final Var pos;
        final Var[][] w = new Var[2][];
        final Var out;
        final List<Tensor> maps = new java.util.ArrayList<>();
        final List<Tensor> grads = new java.util.ArrayList<>();

        CopyAttention(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            embed = Var.param(Tensor.normal(Shape.of(8, dim), 0, 1f, r));
            pos = Var.param(Tensor.normal(Shape.of(SEQ, dim), 0, 1f, r));
            float s = (float) Math.sqrt(1.0 / dim);
            for (int l = 0; l < 2; l++) {
                w[l] = new Var[] {Var.param(Tensor.normal(Shape.of(dim, dim), 0, s, r)),
                    Var.param(Tensor.normal(Shape.of(dim, dim), 0, s, r)),
                    Var.param(Tensor.normal(Shape.of(dim, dim), 0, s, r)),
                    Var.param(Tensor.normal(Shape.of(dim, dim), 0, s, r)),
                    Var.param(Tensor.normal(Shape.of(dim, 2 * dim), 0, s, r)),
                    Var.param(Tensor.normal(Shape.of(2 * dim, dim), 0, (float) Math.sqrt(1.0 / (2 * dim)), r))};
            }
            out = Var.param(Tensor.normal(Shape.of(dim, VALUES), 0, s, r));
        }

        List<Var> parameters() {
            List<Var> p = new java.util.ArrayList<>(List.of(embed, pos, out));
            for (Var[] l : w) {
                p.addAll(List.of(l));
            }
            return p;
        }

        Var embeddings(int[][] tokens) {
            int b = tokens.length;
            float[] f = new float[b * SEQ];
            for (int i = 0; i < b; i++) {
                for (int t = 0; t < SEQ; t++) {
                    f[i * SEQ + t] = tokens[i][t];
                }
            }
            return embed.gather(0, Tensor.of(Shape.of(b * SEQ), f)).reshape(b, SEQ, dim).add(pos);
        }

        /** Logits at the last position; records maps (and their gradients) when {@code record}. */
        Var logits(Var x, boolean record) {
            int b = x.shape().dim(0);
            int dh = dim / heads;
            if (record) {
                maps.clear();
                grads.clear();
            }
            for (int l = 0; l < 2; l++) {
                Var q = x.matmul(w[l][0]).reshape(b, SEQ, heads, dh).permute(0, 2, 1, 3);
                Var k = x.matmul(w[l][1]).reshape(b, SEQ, heads, dh).permute(0, 2, 1, 3);
                Var v = x.matmul(w[l][2]).reshape(b, SEQ, heads, dh).permute(0, 2, 1, 3);
                Var a = Activation.SOFTMAX.forward(q.matmul(k.permute(0, 1, 3, 2)).mul((float) (1 / Math.sqrt(dh))));
                if (record) {
                    maps.add(a.value());
                    grads.add(null);
                    int slot = l;
                    if (a.requiresGrad()) {
                        a.onGradient(g -> {
                            grads.set(slot, g);
                            return null;
                        });
                    }
                }
                Var att = a.matmul(v).permute(0, 2, 1, 3).reshape(b, SEQ, dim).matmul(w[l][3]);
                x = x.add(att);
                x = x.add(x.matmul(w[l][4]).relu().matmul(w[l][5]));
            }
            return x.slice(1, SEQ - 1, SEQ).reshape(b, dim).matmul(out);
        }
    }

    private static int[][] copySequences(int n, int[] sources, int[] targets, SplittableRandom r) {
        int[][] s = new int[n][SEQ];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < SEQ - 1; t++) {
                s[i][t] = r.nextInt(VALUES);
            }
            int m = r.nextInt(SEQ - 2);
            s[i][m] = MARKER;
            s[i][SEQ - 1] = QUERY;
            sources[i] = m + 1;
            targets[i] = s[i][m + 1];
        }
        return s;
    }

    private static Tensor floats(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    private static double[][][][] example(List<Tensor> layers, int i, int heads) {
        double[][][][] a = new double[layers.size()][heads][SEQ][SEQ];
        for (int l = 0; l < layers.size(); l++) {
            float[] f = layers.get(l).toArray();
            for (int h = 0; h < heads; h++) {
                for (int q = 0; q < SEQ; q++) {
                    for (int k = 0; k < SEQ; k++) {
                        a[l][h][q][k] = f[((i * heads + h) * SEQ + q) * SEQ + k];
                    }
                }
            }
        }
        return a;
    }

    /**
     * Attention rollout and flow against raw attention and gradient attributions on a copy
     * task with known source positions: eight tokens, a marker at a random place, and the
     * model (two attention layers, two heads) must output the value right after the marker
     * at the last position, so the true source of the answer is the position after the
     * marker. Attributions explain the predicted logit of the last position.
     *
     * @param trainSteps Adam steps of 64 sequences
     * @param seed the seed
     * @return the copy model's accuracy and per method its scores
     */
    public static AttentionStudy attention(int trainSteps, long seed) {
        CopyAttention m = trainCopy(trainSteps, seed);
        double copyAccuracy = copyAccuracyOf(m, seed + 7);
        int n = 200;
        int[] sources = new int[n];
        int[] targets = new int[n];
        int[][] seqs = copySequences(n, sources, targets, new SplittableRandom(seed + 3));
        // predicted classes and recorded maps with gradients of the predicted logit
        Var x = Var.param(m.embeddings(seqs).value());
        Var logits = m.logits(x, true);
        int[] pred = new int[n];
        float[] lv = logits.value().toArray();
        float[] onehot = new float[n * VALUES];
        for (int i = 0; i < n; i++) {
            for (int c = 1; c < VALUES; c++) {
                pred[i] = lv[i * VALUES + c] > lv[i * VALUES + pred[i]] ? c : pred[i];
            }
            onehot[i * VALUES + pred[i]] = 1;
        }
        logits.mul(Tensor.of(Shape.of(n, VALUES), onehot)).sum().backward();
        float[] embGrad = x.grad().toArray();
        List<Tensor> maps = List.copyOf(m.maps);
        List<Tensor> grads = List.copyOf(m.grads);
        for (Var p : m.parameters()) {
            p.zeroGrad();
        }
        Map<AttentionMethod, AttentionScores> out = new EnumMap<>(AttentionMethod.class);
        for (AttentionMethod method : AttentionMethod.values()) {
            long t0 = System.nanoTime();
            double[][] attr = new double[n][];
            if (method == AttentionMethod.INTEGRATED_GRADIENTS) {
                attr = integratedGradients(m, seqs, onehot, 16);
            } else {
                for (int i = 0; i < n; i++) {
                    attr[i] = switch (method) {
                        case RAW_ATTENTION -> AttentionRollout.rawLast(example(maps, i, m.heads), SEQ - 1);
                        case ROLLOUT -> AttentionRollout.rollout(example(maps, i, m.heads), SEQ - 1);
                        case FLOW -> AttentionRollout.flow(example(maps, i, m.heads), SEQ - 1);
                        case GRADIENT_ROLLOUT -> AttentionRollout.gradientRollout(example(maps, i, m.heads),
                                example(grads, i, m.heads), SEQ - 1);
                        default -> saliency(embGrad, i, m.dim);
                    };
                }
            }
            double ms = (System.nanoTime() - t0) / 1e6 / n;
            int hit = 0;
            for (int i = 0; i < n; i++) {
                int best = 0;
                for (int t = 1; t < SEQ - 1; t++) {
                    best = attr[i][t] > attr[i][best] ? t : best;
                }
                hit += best == sources[i] ? 1 : 0;
            }
            out.put(method, new AttentionScores((double) hit / n, deletion(m, seqs, attr, pred), ms));
        }
        return new AttentionStudy(copyAccuracy, out);
    }

    /**
     * The attention study.
     *
     * @param copyAccuracy the copy model's held-out accuracy
     * @param scores per method its scores
     */
    public record AttentionStudy(double copyAccuracy, Map<AttentionMethod, AttentionScores> scores) {
    }


    private static CopyAttention trainCopy(int steps, long seed) {
        CopyAttention m = new CopyAttention(seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            int[] src = new int[64];
            int[] tgt = new int[64];
            int[][] seqs = copySequences(64, src, tgt, r);
            Functional.crossEntropy(m.logits(m.embeddings(seqs), false), floats(tgt)).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    private static double copyAccuracyOf(CopyAttention m, long seed) {
        int[] src = new int[500];
        int[] tgt = new int[500];
        int[][] seqs = copySequences(500, src, tgt, new SplittableRandom(seed));
        float[] p = Var.noGrad(() -> m.logits(m.embeddings(seqs), false)).value().argmax(1).toArray();
        int hit = 0;
        for (int i = 0; i < tgt.length; i++) {
            hit += p[i] == tgt[i] ? 1 : 0;
        }
        return (double) hit / tgt.length;
    }

    private static double[] saliency(float[] grad, int i, int dim) {
        double[] a = new double[SEQ];
        for (int t = 0; t < SEQ; t++) {
            double s = 0;
            for (int d = 0; d < dim; d++) {
                double g = grad[(i * SEQ + t) * dim + d];
                s += g * g;
            }
            a[t] = Math.sqrt(s);
        }
        return a;
    }

    private static double[][] integratedGradients(CopyAttention m, int[][] seqs, float[] onehot, int steps) {
        int n = seqs.length;
        Tensor e = Var.noGrad(() -> m.embeddings(seqs)).value();
        float[] ev = e.toArray();
        double[][] attr = new double[n][SEQ];
        for (int k = 0; k < steps; k++) {
            float alpha = (k + 0.5f) / steps;
            Var x = Var.param(e.mul(alpha));
            m.logits(x, false).mul(Tensor.of(Shape.of(n, VALUES), onehot)).sum().backward();
            float[] g = x.grad().toArray();
            for (int i = 0; i < n; i++) {
                for (int t = 0; t < SEQ; t++) {
                    for (int d = 0; d < m.dim; d++) {
                        int idx = (i * SEQ + t) * m.dim + d;
                        attr[i][t] += g[idx] * ev[idx] / steps;
                    }
                }
            }
        }
        for (Var p : m.parameters()) {
            p.zeroGrad();
        }
        for (double[] row : attr) {
            for (int t = 0; t < SEQ; t++) {
                row[t] = Math.abs(row[t]);
            }
        }
        return attr;
    }

    private static double deletion(CopyAttention m, int[][] seqs, double[][] attr, int[] pred) {
        int n = seqs.length;
        int[][] order = new int[n][];
        for (int i = 0; i < n; i++) {
            double[] a = attr[i];
            order[i] = IntStream.range(0, SEQ - 1).boxed().sorted((x, y) -> Double.compare(a[y], a[x]))
                    .mapToInt(Integer::intValue).toArray();
        }
        double total = 0;
        for (int k = 0; k < SEQ; k++) {
            int[][] cur = new int[n][];
            for (int i = 0; i < n; i++) {
                cur[i] = seqs[i].clone();
                for (int j = 0; j < Math.min(k, SEQ - 1); j++) {
                    cur[i][order[i][j]] = PAD;
                }
            }
            float[] l = Var.noGrad(() -> m.logits(m.embeddings(cur), false)).value().toArray();
            for (int i = 0; i < n; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < VALUES; c++) {
                    max = Math.max(max, l[i * VALUES + c]);
                }
                double sum = 0;
                for (int c = 0; c < VALUES; c++) {
                    sum += Math.exp(l[i * VALUES + c] - max);
                }
                total += Math.exp(l[i * VALUES + pred[i]] - max) / sum / n / SEQ;
            }
        }
        return total;
    }
}
