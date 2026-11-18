package io.github.zzuegg.brainiac.experiments.explainability;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ConceptShapes;
import io.github.zzuegg.brainiac.experimental.data.PlantedFunctions;
import io.github.zzuegg.brainiac.experimental.explain.AttributionEvaluation;
import io.github.zzuegg.brainiac.experimental.explain.NetworkDissection;
import io.github.zzuegg.brainiac.experimental.nn.Attribution;
import io.github.zzuegg.brainiac.experimental.nn.ConceptActivationVectors;
import io.github.zzuegg.brainiac.experimental.nn.GradCam;
import io.github.zzuegg.brainiac.experimental.nn.LayerHooks;
import io.github.zzuegg.brainiac.experimental.train.ModelAgnosticExplainer;
import io.github.zzuegg.brainiac.experiments.interp.ExplainabilityBenchmark;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Reshape;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Wave 13 items 13.1.14 and 13.1.15: evaluating attributions, and dissecting units.
 *
 * <ul>
 *   <li>{@link #roar}: remove-and-retrain and keep-and-retrain ({@link AttributionEvaluation}) of
 *       saliency, gradient times input, integrated gradients, SmoothGrad, KernelSHAP, LIME and a
 *       random attribution of an MLP on balanced planted labels {@code x1 + sin(2 x3) + 0.5 x5 > 0}
 *       over 10 Gaussian features ({@link PlantedFunctions#gaussian}), against the deletion AUC
 *       without retraining (the {@link ExplainabilityBenchmark} definition) and the recovery of the three relevant
 *       features.</li>
 *   <li>{@link #fidelity}: infidelity and max-sensitivity of the same methods.</li>
 *   <li>{@link #sanity}: the randomisation sanity checks of saliency, gradient times input,
 *       integrated gradients, SmoothGrad, {@link GradCam}, random maps and an "edge detector"
 *       (gradient times input of an untrained network), on a small CNN trained on
 *       {@link ConceptShapes} (six classes); with the pointing score (the top pixel on the object
 *       mask) and the image deletion AUC as the baselines the checks are compared with.</li>
 *   <li>{@link #dissection}: {@link NetworkDissection} of both convolutional layers of a two-layer
 *       CNN trained on the boolean {@link ConceptShapes} task (with or without a spurious
 *       background patch), against random directions in the same layers and
 *       {@link ConceptActivationVectors} (TCAV) relative to random CAVs.</li>
 * </ul>
 */
public final class ExplanationEvaluation {

    private ExplanationEvaluation() {
    }

    /** The attribution methods compared. */
    public enum Method {
        /** The input gradient. */
        SALIENCY,
        /** The gradient times the input. */
        GRAD_X_INPUT,
        /** Integrated gradients from the zero baseline, 16 steps. */
        INTEGRATED_GRADIENTS,
        /** SmoothGrad, 16 noisy copies. */
        SMOOTHGRAD,
        /** KernelSHAP (tabular only). */
        KERNEL_SHAP,
        /** LIME (tabular only). */
        LIME,
        /** Grad-CAM (images only). */
        GRAD_CAM,
        /** Gradient times input of an untrained network of the same architecture (images only). */
        EDGE_DETECTOR,
        /** Gaussian noise. */
        RANDOM
    }

    private static final Method[] TABULAR = {Method.SALIENCY, Method.GRAD_X_INPUT, Method.INTEGRATED_GRADIENTS,
        Method.SMOOTHGRAD, Method.KERNEL_SHAP, Method.LIME, Method.RANDOM};

    private static final Method[] IMAGE = {Method.SALIENCY, Method.GRAD_X_INPUT, Method.INTEGRATED_GRADIENTS,
        Method.SMOOTHGRAD, Method.GRAD_CAM, Method.EDGE_DETECTOR, Method.RANDOM};

    // ------------------------------------------------------------------ tabular

    private record Tabular(PlantedFunctions.Table train, PlantedFunctions.Table test, Model model, double accuracy) {
    }

    /** The relevant features of the tabular task. */
    static final int[] RELEVANT = {1, 3, 5};

    /**
     * Balanced labels {@code x1 + sin(2 x3) + 0.5 x5 > 0} on 10 standard normal features
     * ({@link PlantedFunctions#gaussian}), 2% flipped.
     */
    private static PlantedFunctions.Table balanced(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = PlantedFunctions.gaussian(n, 10, 0, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            boolean v = x[i][1] + Math.sin(2 * x[i][3]) + 0.5 * x[i][5] > 0;
            y[i] = v ^ r.nextDouble() < 0.02 ? 1 : 0;
        }
        return new PlantedFunctions.Table(x, y);
    }

    private static Tabular tabular(long seed) {
        PlantedFunctions.Table train = balanced(400, seed);
        PlantedFunctions.Table test = balanced(200, seed + 1000);
        Model m = Models.mlpClassifier(train.x(), train.y(), 2, 16, 400, seed);
        return new Tabular(train, test, m, Models.accuracy(Models.probability(m), test.x(), test.y()));
    }

    /** The logit margin {@code z1 - z0} of a two-logit classifier, the explained output. */
    private static Function<double[][], double[]> margin(Model m) {
        return rows -> {
            double[] v = m.forward(Var.of(Models.tensor(rows))).value().toDoubleArray();
            double[] out = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                out[i] = v[2 * i + 1] - v[2 * i];
            }
            return out;
        };
    }

    private static double[][] marginGradient(Model m, double[][] rows) {
        Tensor t = Models.tensor(rows);
        float[] g1 = Attribution.gradient(m, t, 1);
        float[] g0 = Attribution.gradient(m, t, 0);
        int d = rows[0].length;
        double[][] out = new double[rows.length][d];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = g1[i * d + j] - g0[i * d + j];
            }
        }
        return out;
    }

    /** A batch attribution function of a tabular method. */
    private static Function<double[][], double[][]> explainer(Method method, Model m, double[][] background,
            long seed) {
        Function<double[][], double[]> f = margin(m);
        ModelAgnosticExplainer black = new ModelAgnosticExplainer(f);
        return rows -> {
            int d = rows[0].length;
            SplittableRandom r = new SplittableRandom(seed ^ Arrays.hashCode(rows[0]));
            double[][] out = new double[rows.length][];
            switch (method) {
                case SALIENCY -> {
                    return marginGradient(m, rows);
                }
                case GRAD_X_INPUT -> {
                    double[][] g = marginGradient(m, rows);
                    for (int i = 0; i < rows.length; i++) {
                        out[i] = new double[d];
                        for (int j = 0; j < d; j++) {
                            out[i][j] = g[i][j] * rows[i][j];
                        }
                    }
                }
                case INTEGRATED_GRADIENTS, SMOOTHGRAD -> {
                    for (int i = 0; i < rows.length; i++) {
                        out[i] = new double[d];
                    }
                    int steps = 16;
                    for (int s = 0; s < steps; s++) {
                        double[][] z = new double[rows.length][d];
                        for (int i = 0; i < rows.length; i++) {
                            for (int j = 0; j < d; j++) {
                                z[i][j] = method == Method.SMOOTHGRAD ? rows[i][j] + 0.3 * r.nextGaussian()
                                        : rows[i][j] * (s + 0.5) / steps;
                            }
                        }
                        double[][] g = marginGradient(m, z);
                        for (int i = 0; i < rows.length; i++) {
                            for (int j = 0; j < d; j++) {
                                double scale = method == Method.SMOOTHGRAD ? 1 : rows[i][j];
                                out[i][j] += scale * g[i][j] / steps;
                            }
                        }
                    }
                }
                case KERNEL_SHAP -> {
                    for (int i = 0; i < rows.length; i++) {
                        out[i] = black.kernelShap(rows[i], background, 96, r);
                    }
                }
                case LIME -> {
                    double[] scale = new double[d];
                    Arrays.fill(scale, 1);
                    for (int i = 0; i < rows.length; i++) {
                        out[i] = black.lime(rows[i], scale, 150, 0.75 * Math.sqrt(d), r);
                    }
                }
                default -> {
                    for (int i = 0; i < rows.length; i++) {
                        out[i] = new double[d];
                        for (int j = 0; j < d; j++) {
                            out[i][j] = r.nextGaussian();
                        }
                    }
                }
            }
            return out;
        };
    }

    /**
     * One row of the ROAR table.
     *
     * @param method the attribution method
     * @param roar test accuracy after removing the top features and retraining, per fraction
     * @param kar test accuracy after keeping only the top features and retraining, per fraction
     * @param deletionAuc the deletion AUC without retraining (lower is better)
     * @param recovery the share of test rows whose top three features are the planted {1, 3, 5}
     */
    public record RoarRow(Method method, double[] roar, double[] kar, double deletionAuc, double recovery) {
    }

    /**
     * The ROAR study.
     *
     * @param fractions the shares of features removed or kept
     * @param modelAccuracy the explained MLP's test accuracy
     * @param rows one row per method
     * @param discordantPairs method pairs ordered differently by deletion AUC and by ROAR at the
     *     second fraction
     */
    public record RoarStudy(double[] fractions, double modelAccuracy, List<RoarRow> rows, int discordantPairs) {
    }

    /**
     * Remove-and-retrain against deletion.
     *
     * @param fractions the shares of features removed or kept
     * @param seed the seed
     * @return the study
     */
    public static RoarStudy roar(double[] fractions, long seed) {
        Tabular t = tabular(seed);
        double[][] background = Arrays.copyOf(t.train().x(), 16);
        double[] fill = new double[10];
        for (double[] row : t.train().x()) {
            for (int j = 0; j < fill.length; j++) {
                fill[j] += row[j] / t.train().rows();
            }
        }
        Function<double[][], double[]> prob = Models.probability(t.model());
        AttributionEvaluation.Retrainer retrain = (x, y) -> Models.probability(
                Models.mlpClassifier(x, y, 2, 16, 300, seed + 7));
        List<RoarRow> rows = new ArrayList<>();
        for (Method method : TABULAR) {
            Function<double[][], double[][]> ex = explainer(method, t.model(), background, seed);
            double[][] trainAttr = ex.apply(t.train().x());
            double[][] testAttr = ex.apply(t.test().x());
            double[] roar = AttributionEvaluation.removeAndRetrain(AttributionEvaluation.Metric.ROAR, t.train().x(),
                    t.train().y(), trainAttr, t.test().x(), t.test().y(), testAttr, fill, fractions, retrain);
            double[] kar = AttributionEvaluation.removeAndRetrain(AttributionEvaluation.Metric.KAR, t.train().x(),
                    t.train().y(), trainAttr, t.test().x(), t.test().y(), testAttr, fill, fractions, retrain);
            double deletion = 0;
            int hits = 0;
            for (int i = 0; i < t.test().rows(); i++) {
                double[] x = t.test().x()[i];
                boolean positive = prob.apply(new double[][] {x})[0] > 0.5;
                Function<double[][], double[]> score = rs -> Arrays.stream(prob.apply(rs))
                        .map(v -> positive ? v : 1 - v).toArray();
                deletion += AttributionEvaluation.deletionAuc(score, x, testAttr[i], fill, 10) / t.test().rows();
                hits += topIsRelevant(testAttr[i]) ? 1 : 0;
            }
            rows.add(new RoarRow(method, roar, kar, deletion, hits / (double) t.test().rows()));
        }
        int discordant = 0;
        int f = Math.min(1, fractions.length - 1);
        for (int a = 0; a < rows.size(); a++) {
            for (int b = a + 1; b < rows.size(); b++) {
                double dDel = rows.get(a).deletionAuc() - rows.get(b).deletionAuc();
                double dRoar = rows.get(a).roar()[f] - rows.get(b).roar()[f];
                if (dDel * dRoar < 0) {
                    discordant++;
                }
            }
        }
        return new RoarStudy(fractions, t.accuracy(), rows, discordant);
    }

    /** Whether the {@link #RELEVANT}.length largest absolute attributions are the relevant features. */
    private static boolean topIsRelevant(double[] a) {
        double least = Double.POSITIVE_INFINITY;
        for (int j : RELEVANT) {
            least = Math.min(least, Math.abs(a[j]));
        }
        int above = 0;
        for (double v : a) {
            above += Math.abs(v) >= least ? 1 : 0;
        }
        return above == RELEVANT.length;
    }

    /**
     * One row of the fidelity table.
     *
     * @param method the method
     * @param infidelity the mean relative infidelity (lower is better)
     * @param maxSensitivity the mean relative max-sensitivity (lower is better)
     */
    public record FidelityRow(Method method, double infidelity, double maxSensitivity) {
    }

    /**
     * Infidelity and max-sensitivity of the tabular methods.
     *
     * @param rowsToExplain test rows to evaluate
     * @param seed the seed
     * @return one row per method
     */
    public static List<FidelityRow> fidelity(int rowsToExplain, long seed) {
        Tabular t = tabular(seed);
        double[][] background = Arrays.copyOf(t.train().x(), 16);
        Function<double[][], double[]> f = margin(t.model());
        List<FidelityRow> out = new ArrayList<>();
        for (Method method : TABULAR) {
            Function<double[][], double[][]> ex = explainer(method, t.model(), background, seed);
            SplittableRandom r = new SplittableRandom(seed);
            double inf = 0;
            double sens = 0;
            for (int i = 0; i < rowsToExplain; i++) {
                double[] x = t.test().x()[i];
                double[] a = ex.apply(new double[][] {x})[0];
                inf += AttributionEvaluation.infidelity(f, x, a, 0.3, 64, r) / rowsToExplain;
                sens += AttributionEvaluation.maxSensitivity(z -> ex.apply(new double[][] {z})[0], x, 0.1, 6, r)
                        / rowsToExplain;
            }
            out.add(new FidelityRow(method, inf, sens));
        }
        return out;
    }

    // ------------------------------------------------------------------ images: sanity checks

    private static final int SIDE = 12;
    private static final ConceptShapes SHAPES = new ConceptShapes(SIDE, 0.05f);

    private static Supplier<Model> cnn(long seed) {
        return () -> Model.sequential().input(2 * SIDE * SIDE).layer(new Reshape(2, SIDE, SIDE))
                .conv2d(8, 3, Activation.RELU).maxPool(2, 2).flatten().dense(6, Activation.IDENTITY).seed(seed)
                .build();
    }

    private static Tensor ids(int[] labels) {
        float[] v = new float[labels.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = labels[i];
        }
        return Tensor.of(Shape.of(v.length), v);
    }

    private static Model train(Model m, Tensor images, int[] labels, int steps, float lr, long seed) {
        Tensor ty = ids(labels);
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int n = labels.length;
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(n - 32 + 1);
            Functional.crossEntropy(m.forward(Var.of(images.slice(0, from, from + 32)), true),
                    ty.slice(0, from, from + 32)).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    private static double[][] softmax(Model m, Tensor images) {
        float[] z = m.forward(Var.of(images)).value().toArray();
        int n = images.shape().dim(0);
        int c = z.length / n;
        double[][] out = new double[n][c];
        for (int i = 0; i < n; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < c; k++) {
                max = Math.max(max, z[i * c + k]);
            }
            double sum = 0;
            for (int k = 0; k < c; k++) {
                out[i][k] = Math.exp(z[i * c + k] - max);
                sum += out[i][k];
            }
            for (int k = 0; k < c; k++) {
                out[i][k] /= sum;
            }
        }
        return out;
    }

    /** Input-space attribution {@code [n][2 * area]} of each image for its own class. */
    private static double[][] inputMaps(Method method, Model m, Tensor images, int[] cls, long seed) {
        int n = cls.length;
        int d = 2 * SIDE * SIDE;
        float[] x = images.toArray();
        double[][] out = new double[n][d];
        SplittableRandom r = new SplittableRandom(seed);
        int steps = method == Method.SALIENCY || method == Method.GRAD_X_INPUT || method == Method.EDGE_DETECTOR ? 1
                : 16;
        for (int s = 0; s < steps; s++) {
            float[] z = x.clone();
            if (method == Method.SMOOTHGRAD) {
                for (int p = 0; p < z.length; p++) {
                    z[p] += (float) (0.15 * r.nextGaussian());
                }
            } else if (method == Method.INTEGRATED_GRADIENTS) {
                for (int p = 0; p < z.length; p++) {
                    z[p] *= (float) ((s + 0.5) / steps);
                }
            }
            Tensor batch = Tensor.of(images.shape(), z);
            for (int c = 0; c < 6; c++) {
                boolean any = false;
                for (int k : cls) {
                    any |= k == c;
                }
                if (!any) {
                    continue;
                }
                float[] g = Attribution.gradient(m, batch, c);
                for (int i = 0; i < n; i++) {
                    if (cls[i] != c) {
                        continue;
                    }
                    for (int p = 0; p < d; p++) {
                        double scale = method == Method.SALIENCY || method == Method.SMOOTHGRAD ? 1 : x[i * d + p];
                        out[i][p] += scale * g[i * d + p] / steps;
                    }
                }
            }
        }
        return out;
    }

    /** A pixel map {@code [n][area]} of a method (absolute values summed over the channels). */
    private static double[][] pixelMaps(Method method, Model m, Model untrained, Tensor images, int[] cls, long seed) {
        int n = cls.length;
        int area = SIDE * SIDE;
        double[][] out = new double[n][area];
        if (method == Method.GRAD_CAM) {
            for (int i = 0; i < n; i++) {
                float[] cam = GradCam.upsample(GradCam.map(m, 2, images.slice(0, i, i + 1), cls[i]), SIDE, 2)
                        .toArray();
                for (int p = 0; p < area; p++) {
                    out[i][p] = cam[p];
                }
            }
            return out;
        }
        if (method == Method.RANDOM) {
            SplittableRandom r = new SplittableRandom(seed);
            for (double[] row : out) {
                for (int p = 0; p < area; p++) {
                    row[p] = Math.abs(r.nextGaussian());
                }
            }
            return out;
        }
        double[][] a = inputMaps(method, method == Method.EDGE_DETECTOR ? untrained : m, images, cls, seed);
        for (int i = 0; i < n; i++) {
            for (int p = 0; p < area; p++) {
                out[i][p] = Math.abs(a[i][p]) + Math.abs(a[i][area + p]);
            }
        }
        return out;
    }

    /**
     * One row of the sanity-check table.
     *
     * @param method the method
     * @param pointing the share of images whose top pixel lies on the object
     * @param deletionAuc the image deletion AUC (pixels removed in ten steps; lower is better)
     * @param topLayerRank the rank correlation with the trained model's maps after re-drawing the
     *     output layer
     * @param allLayersRank after re-drawing every layer
     * @param allLayersSsim the SSIM after re-drawing every layer
     * @param dataRank the rank correlation with the maps of a model trained on permuted labels
     */
    public record SanityRow(Method method, double pointing, double deletionAuc, double topLayerRank,
            double allLayersRank, double allLayersSsim, double dataRank) {
    }

    /**
     * The sanity-check study.
     *
     * @param accuracy the CNN's test accuracy
     * @param shuffledAccuracy the permuted-label model's test accuracy
     * @param rows one row per method
     */
    public record SanityStudy(double accuracy, double shuffledAccuracy, List<SanityRow> rows) {
    }

    /**
     * Randomisation sanity checks of the image attributions.
     *
     * @param steps training steps of each CNN
     * @param images test images explained
     * @param seed the seed
     * @return the study
     */
    public static SanityStudy sanity(int steps, int images, long seed) {
        ConceptShapes.Batch train = SHAPES.generate(ConceptShapes.Task.SIX_CLASS, 1200, -1, seed);
        ConceptShapes.Batch test = SHAPES.generate(ConceptShapes.Task.SIX_CLASS, images, -1, seed + 1);
        Supplier<Model> arch = cnn(seed);
        Model m = train(arch.get(), train.images(), train.labels(), steps, 1e-2f, seed);
        Model shuffled = train(arch.get(), train.images(), AttributionEvaluation.shuffledLabels(train.labels(), seed),
                steps, 1e-2f, seed);
        Model untrained = cnn(seed + 99).get();
        Model top = AttributionEvaluation.cascade(m, arch, 2, seed + 5);
        Model all = AttributionEvaluation.cascade(m, arch, 4, seed + 6);
        double[][] p = softmax(m, test.images());
        int[] cls = new int[images];
        int ok = 0;
        for (int i = 0; i < images; i++) {
            for (int k = 1; k < 6; k++) {
                if (p[i][k] > p[i][cls[i]]) {
                    cls[i] = k;
                }
            }
            ok += cls[i] == test.labels()[i] ? 1 : 0;
        }
        double[][] ps = softmax(shuffled, test.images());
        int okS = 0;
        for (int i = 0; i < images; i++) {
            int best = 0;
            for (int k = 1; k < 6; k++) {
                if (ps[i][k] > ps[i][best]) {
                    best = k;
                }
            }
            okS += best == test.labels()[i] ? 1 : 0;
        }
        int area = SIDE * SIDE;
        float[] raw = test.images().toArray();
        List<SanityRow> rows = new ArrayList<>();
        for (Method method : IMAGE) {
            double[][] base = pixelMaps(method, m, untrained, test.images(), cls, seed);
            double[][] mt = pixelMaps(method, top, untrained, test.images(), cls, seed + 1);
            double[][] ma = pixelMaps(method, all, untrained, test.images(), cls, seed + 2);
            double[][] md = pixelMaps(method, shuffled, untrained, test.images(), cls, seed + 3);
            double point = 0;
            double del = 0;
            double rt = 0;
            double ra = 0;
            double sa = 0;
            double rd = 0;
            for (int i = 0; i < images; i++) {
                int best = 0;
                for (int q = 1; q < area; q++) {
                    if (base[i][q] > base[i][best]) {
                        best = q;
                    }
                }
                boolean[][] mk = test.masks()[i];
                point += mk[0][best] || mk[1][best] || mk[2][best] ? 1 : 0;
                double[] x = new double[2 * area];
                double[] attr = new double[2 * area];
                for (int q = 0; q < 2 * area; q++) {
                    x[q] = raw[i * 2 * area + q];
                    attr[q] = base[i][q % area];
                }
                int c = cls[i];
                Function<double[][], double[]> score = rs -> {
                    double[][] sm = softmax(m, Models.tensor(rs));
                    return Arrays.stream(sm).mapToDouble(v -> v[c]).toArray();
                };
                del += AttributionEvaluation.deletionAuc(score, x, attr, new double[2 * area], 10);
                rt += AttributionEvaluation.rankCorrelation(base[i], mt[i]);
                ra += AttributionEvaluation.rankCorrelation(base[i], ma[i]);
                sa += AttributionEvaluation.ssim(base[i], ma[i]);
                rd += AttributionEvaluation.rankCorrelation(base[i], md[i]);
            }
            rows.add(new SanityRow(method, point / images, del / images, rt / images, ra / images, sa / images,
                    rd / images));
        }
        return new SanityStudy(ok / (double) images, okS / (double) images, rows);
    }

    // ------------------------------------------------------------------ dissection

    /**
     * The dissection of one layer.
     *
     * @param layer the layer name
     * @param dissection the unit labels
     * @param random the dissection of random directions in the layer
     */
    public record LayerReport(String layer, NetworkDissection.Dissection dissection,
            NetworkDissection.Dissection random) {
    }

    /**
     * The dissection study.
     *
     * @param accuracy test accuracy on the training distribution
     * @param brokenAccuracy test accuracy with the patch uninformative
     * @param layers per convolutional layer
     * @param tcavZ per concept of {@link #TCAV_CONCEPTS}, the TCAV sensitivity's z-score against
     *     random CAVs
     * @param agreement the share of concepts on which dissection (some unit detects the concept)
     *     and TCAV ({@code |z| > 2}) agree
     */
    public record DissectionStudy(double accuracy, double brokenAccuracy, List<LayerReport> layers, double[] tcavZ,
            double agreement) {
    }

    /** The concepts TCAV tests: the five binary concepts and the patch. */
    public static final String[] TCAV_CONCEPTS = {"plus", "colour1", "large", "top", "striped", "patch"};

    /** The dissection mask of each TCAV concept. */
    private static final String[] TCAV_MASKS = {"plus", "channel1", "large", "top-half", "striped", "patch"};

    /** The detector threshold of the paper. */
    public static final double DETECTOR_IOU = 0.04;

    /**
     * Dissects a two-layer CNN trained on the boolean concept task.
     *
     * @param spurious whether the background patch predicts the label (95%) in training
     * @param steps training steps
     * @param topShare the share of each unit's activations kept as its mask
     * @param seed the seed
     * @return the study
     */
    public static DissectionStudy dissection(boolean spurious, int steps, double topShare, long seed) {
        double rate = spurious ? 0.95 : 0.5;
        ConceptShapes.Batch train = SHAPES.generate(ConceptShapes.Task.BOOLEAN, 1200, rate, seed);
        ConceptShapes.Batch test = SHAPES.generate(ConceptShapes.Task.BOOLEAN, 300, rate, seed + 1);
        ConceptShapes.Batch broken = SHAPES.generate(ConceptShapes.Task.BOOLEAN, 300, 0.5, seed + 2);
        Model m = Model.sequential().input(2 * SIDE * SIDE).layer(new Reshape(2, SIDE, SIDE))
                .conv2d(8, 3, Activation.RELU).conv2d(8, 3, Activation.RELU).maxPool(2, 2).flatten()
                .dense(2, Activation.IDENTITY).seed(seed).build();
        train(m, train.images(), train.labels(), steps, 1e-2f, seed);
        double acc = accuracy(m, test);
        double brokenAcc = accuracy(m, broken);
        List<Tensor> acts = LayerHooks.activations(m, test.images());
        List<LayerReport> layers = new ArrayList<>();
        boolean[] dissected = new boolean[TCAV_CONCEPTS.length];
        for (int layer = 1; layer <= 2; layer++) {
            Tensor a = acts.get(layer);
            NetworkDissection.Dissection d = NetworkDissection.dissect(NetworkDissection.unitMaps(a, SIDE, 1),
                    test.masks(), ConceptShapes.MASK_NAMES, topShare, 3, 5);
            NetworkDissection.Dissection rnd = NetworkDissection.dissect(
                    NetworkDissection.randomDirections(a, 8, SIDE, 1, seed + layer), test.masks(),
                    ConceptShapes.MASK_NAMES, topShare, 1, 1);
            layers.add(new LayerReport("conv" + layer, d, rnd));
            for (int k = 0; k < TCAV_MASKS.length; k++) {
                dissected[k] |= d.detectors(TCAV_MASKS[k], DETECTOR_IOU) > 0;
            }
        }
        double[] z = tcav(m, 2, train, test, seed);
        int agree = 0;
        for (int k = 0; k < z.length; k++) {
            agree += dissected[k] == (Math.abs(z[k]) > 2) ? 1 : 0;
        }
        return new DissectionStudy(acc, brokenAcc, layers, z, agree / (double) z.length);
    }

    private static double accuracy(Model m, ConceptShapes.Batch b) {
        float[] pred = m.forward(Var.of(b.images())).value().argmax(1).toArray();
        int ok = 0;
        for (int i = 0; i < pred.length; i++) {
            ok += (int) pred[i] == b.labels()[i] ? 1 : 0;
        }
        return ok / (double) pred.length;
    }

    private static Tensor rows(Tensor images, List<Integer> idx) {
        int d = images.shape().dim(1);
        float[] v = images.toArray();
        float[] out = new float[idx.size() * d];
        for (int i = 0; i < idx.size(); i++) {
            System.arraycopy(v, idx.get(i) * d, out, i * d, d);
        }
        return Tensor.of(Shape.of(idx.size(), d), out);
    }

    /** TCAV sensitivity z-scores of the concepts at a layer, against random concept directions. */
    private static double[] tcav(Model m, int layer, ConceptShapes.Batch train, ConceptShapes.Batch test, long seed) {
        int per = 60;
        Tensor pool = train.images().slice(0, 0, 2 * per);
        List<Integer> positives = new ArrayList<>();
        for (int i = 0; i < test.size() && positives.size() < 100; i++) {
            if (test.labels()[i] == 1) {
                positives.add(i);
            }
        }
        Tensor inputs = rows(test.images(), positives);
        List<float[]> random = ConceptActivationVectors.randomCavs(m, layer, pool, 10, seed);
        double[] rs = random.stream().mapToDouble(c -> ConceptActivationVectors.sensitivity(m, layer, c, inputs, 1))
                .toArray();
        double mean = Arrays.stream(rs).average().orElse(0);
        double sd = Math.sqrt(Arrays.stream(rs).map(v -> (v - mean) * (v - mean)).sum() / Math.max(1, rs.length - 1));
        double[] z = new double[TCAV_CONCEPTS.length];
        for (int k = 0; k < z.length; k++) {
            List<Integer> with = new ArrayList<>();
            List<Integer> without = new ArrayList<>();
            for (int i = 2 * per; i < train.size(); i++) {
                boolean has = k < ConceptShapes.CONCEPTS ? train.concepts()[i][k] == 1 : train.patch()[i];
                if (has && with.size() < per) {
                    with.add(i);
                } else if (!has && without.size() < per) {
                    without.add(i);
                }
            }
            float[] cav = ConceptActivationVectors.cav(m, layer, rows(train.images(), with),
                    rows(train.images(), without));
            double s = ConceptActivationVectors.sensitivity(m, layer, cav, inputs, 1);
            z[k] = sd > 0 ? (s - mean) / sd : 0;
        }
        return z;
    }
}
