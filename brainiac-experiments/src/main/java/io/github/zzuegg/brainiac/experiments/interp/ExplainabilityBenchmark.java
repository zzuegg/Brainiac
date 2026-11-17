package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
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
}
