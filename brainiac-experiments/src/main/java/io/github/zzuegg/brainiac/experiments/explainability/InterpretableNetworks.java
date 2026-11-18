package io.github.zzuegg.brainiac.experiments.explainability;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.boost.ExplainableBoosting;
import io.github.zzuegg.brainiac.experimental.data.ConceptShapes;
import io.github.zzuegg.brainiac.experimental.data.PlantedFunctions;
import io.github.zzuegg.brainiac.experimental.explain.ConceptBottleneck;
import io.github.zzuegg.brainiac.experimental.explain.ConceptLeakage;
import io.github.zzuegg.brainiac.experimental.explain.ExplanationRegularizer;
import io.github.zzuegg.brainiac.experimental.explain.PrototypeLayer;
import io.github.zzuegg.brainiac.experimental.explain.ProtoPNet;
import io.github.zzuegg.brainiac.experimental.explain.SelfExplainingNetwork;
import io.github.zzuegg.brainiac.experimental.nn.Attribution;
import io.github.zzuegg.brainiac.experimental.nn.ConceptActivationVectors;
import io.github.zzuegg.brainiac.experimental.nn.GradCam;
import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.experimental.nn.NeuralAdditiveModel;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;
import io.github.zzuegg.brainiac.experimental.train.GroupRobustTraining;
import io.github.zzuegg.brainiac.experimental.train.ModelAgnosticExplainer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Reshape;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 13 items 13.1.7-13.1.9 and 13.1.16: interpretable-by-design networks and
 * explanation-guided training on {@link ConceptShapes} (12x12, two colour channels) and
 * {@link PlantedFunctions}.
 *
 * <ul>
 *   <li>{@link #prototypes} (13.1.7): {@link ProtoPNet} with L2-log and cosine similarity at
 *       three learning rates on the six-class shape-by-texture task, against the same backbone
 *       with a linear head (explained by {@link GradCam}) and a k-NN on its features
 *       ({@link KnnIndex}). Scores: accuracy, its spread over learning rates, and purity (the
 *       share of images whose most activated prototype patch covers the object).</li>
 *   <li>{@link #concepts} (13.1.8): every {@link ConceptBottleneck} mode on the boolean concept
 *       task, against an end-to-end MLP with linear concept probes on its hidden layer (the
 *       TCAV-style baseline of concept accuracy); scores: task and concept accuracy, accuracy
 *       after intervening on 0-5 concepts, {@link ConceptLeakage} CTL and ICL, and the same on
 *       an incomplete-concept variant.</li>
 *   <li>{@link #senn} (13.1.9): {@link SelfExplainingNetwork} on a varying-coefficient model,
 *       against an MLP explained by input gradients and by LIME, and the additive
 *       {@link NeuralAdditiveModel} and {@link ExplainableBoosting} (local slopes of their shape
 *       functions); scores: test MSE, correlation of the explanation with the true local
 *       coefficients, and a local Lipschitz estimate of the explanation.</li>
 *   <li>{@link #guided} (13.1.16): a background patch predicts the label 95% of the time in
 *       training and 50% at test. An MLP trained by ERM, by {@link ExplanationRegularizer}
 *       (input gradients with masks on 10% or all images, and the integrated-gradients variant),
 *       by GroupDRO and JTT ({@link GroupRobustTraining}, with group labels), by last-layer
 *       retraining on group-balanced data, and with patch-randomising augmentation; scores:
 *       accuracy when the shortcut breaks, on the shortcut-aligned split, and the share of
 *       saliency mass on the patch (Grad-CAM needs a convolutional layer, and conv and pooling
 *       ops have no second derivatives in core autograd, so the guided models are MLPs).</li>
 * </ul>
 */
public final class InterpretableNetworks {

    private InterpretableNetworks() {
    }

    private static final int SIDE = 12;
    private static final ConceptShapes SHAPES = new ConceptShapes(SIDE, 0.05f);

    private static Tensor ids(int[] labels) {
        float[] v = new float[labels.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = labels[i];
        }
        return Tensor.of(Shape.of(v.length), v);
    }

    private static double accuracy(int[] pred, int[] labels) {
        int ok = 0;
        for (int i = 0; i < labels.length; i++) {
            ok += pred[i] == labels[i] ? 1 : 0;
        }
        return ok / (double) labels.length;
    }

    private static int[] argmax(Tensor logits) {
        float[] a = logits.argmax(1).toArray();
        int[] out = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = (int) a[i];
        }
        return out;
    }

    // ------------------------------------------------------------------ 13.1.7 prototypes

    private static Model backbone(long seed) {
        return Model.sequential().input(2 * SIDE * SIDE).layer(new Reshape(2, SIDE, SIDE))
                .conv2d(8, 3, Activation.RELU).maxPool(2, 2).seed(seed).build();
    }

    /**
     * One prototype-table row.
     *
     * @param model the model
     * @param accuracy accuracy per learning rate
     * @param purity the share of test images whose explanation patch covers the object (per learning rate)
     */
    public record PrototypeRow(String model, double[] accuracy, double[] purity) {

        /** @return the accuracy spread (max - min) over learning rates */
        public double spread() {
            return Arrays.stream(accuracy).max().orElse(0) - Arrays.stream(accuracy).min().orElse(0);
        }

        /** @return the mean accuracy */
        public double meanAccuracy() {
            return Arrays.stream(accuracy).average().orElse(0);
        }

        /** @return the mean purity */
        public double meanPurity() {
            return Arrays.stream(purity).average().orElse(0);
        }
    }

    /** Whether the 4x4 input window of pooled cell {@code p} (5x5 map) touches the object. */
    private static boolean windowHitsObject(boolean[][] masks, int p) {
        int ci = p / 5;
        int cj = p % 5;
        for (int r = 2 * ci; r < 2 * ci + 4; r++) {
            for (int c = 2 * cj; c < 2 * cj + 4; c++) {
                int px = r * SIDE + c;
                if (masks[0][px] || masks[1][px] || masks[2][px]) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * The prototype table.
     *
     * @param learningRates the learning rates
     * @param steps training steps per model
     * @param seed the seed
     * @return rows for ProtoPNet (L2 and cosine), the black-box CNN and k-NN, plus the chance purity
     */
    public static List<PrototypeRow> prototypes(float[] learningRates, int steps, long seed) {
        ConceptShapes.Batch train = SHAPES.generate(ConceptShapes.Task.SIX_CLASS, 1200, -1, seed);
        ConceptShapes.Batch test = SHAPES.generate(ConceptShapes.Task.SIX_CLASS, 300, -1, seed + 1);
        Tensor ty = ids(train.labels());
        List<PrototypeRow> rows = new ArrayList<>();
        int k = learningRates.length;
        for (PrototypeLayer.Similarity sim : PrototypeLayer.Similarity.values()) {
            double[] acc = new double[k];
            double[] pur = new double[k];
            for (int i = 0; i < k; i++) {
                ProtoPNet net = new ProtoPNet(backbone(seed), 6, 2, sim, seed);
                net.train(train.images(), ty, steps, learningRates[i], 32, seed);
                net.project(train.images().slice(0, 0, 400), Arrays.copyOf(train.labels(), 400));
                net.fitLastLayer(train.images().slice(0, 0, 400), ty.slice(0, 0, 400), 60, 1e-2f, 1e-3f);
                acc[i] = accuracy(net.predict(test.images()), test.labels());
                int[][] top = net.topPatches(test.images());
                int hits = 0;
                for (int t = 0; t < test.size(); t++) {
                    hits += windowHitsObject(test.masks()[t], top[t][1]) ? 1 : 0;
                }
                pur[i] = hits / (double) test.size();
            }
            rows.add(new PrototypeRow("ProtoPNet-" + sim, acc, pur));
        }
        double[] acc = new double[k];
        double[] pur = new double[k];
        double[] knn = new double[k];
        for (int i = 0; i < k; i++) {
            Model m = Model.sequential().input(2 * SIDE * SIDE).layer(new Reshape(2, SIDE, SIDE))
                    .conv2d(8, 3, Activation.RELU).maxPool(2, 2).flatten().dense(6, Activation.IDENTITY).seed(seed)
                    .build();
            Optimizer opt = Optimizer.adam(learningRates[i]);
            SplittableRandom r = new SplittableRandom(seed);
            for (int s = 0; s < steps; s++) {
                int from = r.nextInt(train.size() - 32 + 1);
                Functional.crossEntropy(m.forward(Var.of(train.images().slice(0, from, from + 32)), true),
                        ty.slice(0, from, from + 32)).backward();
                opt.step(m.parameters());
            }
            int[] pred = argmax(m.forward(Var.of(test.images())).value());
            acc[i] = accuracy(pred, test.labels());
            int hits = 0;
            for (int t = 0; t < test.size(); t++) {
                float[] cam = GradCam.upsample(GradCam.map(m, 2, test.images().slice(0, t, t + 1), pred[t]), SIDE, 2)
                        .toArray();
                int best = 0;
                for (int p = 1; p < cam.length; p++) {
                    if (cam[p] > cam[best]) {
                        best = p;
                    }
                }
                boolean[][] mk = test.masks()[t];
                hits += mk[0][best] || mk[1][best] || mk[2][best] || near(mk, best) ? 1 : 0;
            }
            pur[i] = hits / (double) test.size();
            float[][] featTrain = ConceptActivationVectors.activations(m, 3, train.images());
            float[][] featTest = ConceptActivationVectors.activations(m, 3, test.images());
            KnnIndex index = KnnIndex.create(featTrain[0].length, 1, KnnIndex.Metric.L2);
            for (int t = 0; t < featTrain.length; t++) {
                index.add(featTrain[t], new float[] {train.labels()[t]});
            }
            int ok = 0;
            for (int t = 0; t < featTest.length; t++) {
                int[] votes = new int[6];
                for (KnnIndex.Hit h : index.search(featTest[t], 5)) {
                    votes[(int) index.value(h.row())[0]]++;
                }
                int best = 0;
                for (int c = 1; c < 6; c++) {
                    if (votes[c] > votes[best]) {
                        best = c;
                    }
                }
                ok += best == test.labels()[t] ? 1 : 0;
            }
            knn[i] = ok / (double) featTest.length;
        }
        rows.add(new PrototypeRow("CNN+GradCAM", acc, pur));
        rows.add(new PrototypeRow("kNN-features", knn, new double[k]));
        double chance = 0;
        for (int t = 0; t < test.size(); t++) {
            int hits = 0;
            for (int p = 0; p < 25; p++) {
                hits += windowHitsObject(test.masks()[t], p) ? 1 : 0;
            }
            chance += hits / 25.0 / test.size();
        }
        double[] c = new double[k];
        Arrays.fill(c, chance);
        rows.add(new PrototypeRow("chance-purity", new double[k], c));
        return rows;
    }

    private static boolean near(boolean[][] mk, int p) {
        int r = p / SIDE;
        int c = p % SIDE;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int rr = r + dr;
                int cc = c + dc;
                if (rr >= 0 && rr < SIDE && cc >= 0 && cc < SIDE) {
                    int q = rr * SIDE + cc;
                    if (mk[0][q] || mk[1][q] || mk[2][q]) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ 13.1.8 concepts

    /**
     * One concept-table row.
     *
     * @param model the model
     * @param taskAccuracy test task accuracy
     * @param conceptAccuracy mean test concept accuracy
     * @param intervened accuracy after correcting the first 0..k concepts
     * @param ctl concepts-task leakage
     * @param icl inter-concept leakage
     */
    public record ConceptRow(String model, double taskAccuracy, double conceptAccuracy, double[] intervened,
            double ctl, double icl) {

        /** @return accuracy with all concepts corrected minus accuracy with none */
        public double interventionGain() {
            return intervened[intervened.length - 1] - intervened[0];
        }
    }

    private static Tensor conceptTensor(ConceptShapes.Batch b, int k) {
        float[] v = new float[b.size() * k];
        for (int i = 0; i < b.size(); i++) {
            for (int c = 0; c < k; c++) {
                v[i * k + c] = b.concepts()[i][c];
            }
        }
        return Tensor.of(Shape.of(b.size(), k), v);
    }

    /**
     * The concept table.
     *
     * @param incomplete use the variant whose label depends on the unlabelled texture concept
     * @param steps training steps
     * @param seed the seed
     * @return a row per mode plus the end-to-end MLP with linear probes
     */
    public static List<ConceptRow> concepts(boolean incomplete, int steps, long seed) {
        ConceptShapes.Task task = incomplete ? ConceptShapes.Task.BOOLEAN_TEXTURE : ConceptShapes.Task.BOOLEAN;
        int k = incomplete ? 4 : 5;
        ConceptShapes.Batch train = SHAPES.generate(task, 1200, -1, seed);
        ConceptShapes.Batch test = SHAPES.generate(task, 400, -1, seed + 1);
        Tensor c = conceptTensor(train, k);
        Tensor ct = conceptTensor(test, k);
        Tensor y = ids(train.labels());
        List<ConceptRow> rows = new ArrayList<>();
        for (ConceptBottleneck.Mode mode : ConceptBottleneck.Mode.values()) {
            ConceptBottleneck cbm = new ConceptBottleneck(mode, 2 * SIDE * SIDE, 64, k, 2, seed);
            cbm.fit(train.images(), c, y, steps, 1f, seed);
            double[] curve = new double[k + 1];
            for (int m = 0; m <= k; m++) {
                int[] which = new int[m];
                for (int q = 0; q < m; q++) {
                    which[q] = q;
                }
                curve[m] = accuracy(cbm.intervene(test.images(), which, ct), test.labels());
            }
            double[][] probs = toDouble(cbm.concepts(test.images()));
            double[][] truth = toDouble(ct);
            double conceptAcc = 0;
            for (int i = 0; i < probs.length; i++) {
                for (int q = 0; q < k; q++) {
                    conceptAcc += (probs[i][q] > 0.5 ? 1 : 0) == truth[i][q] ? 1 : 0;
                }
            }
            rows.add(new ConceptRow(mode.name(), curve[0], conceptAcc / (probs.length * k), curve,
                    ConceptLeakage.ctl(probs, truth, test.labels(), 8),
                    ConceptLeakage.icl(probs, truth, 8, ConceptLeakage.Estimator.BINNED)));
        }
        // the end-to-end black box with linear concept probes on its hidden layer
        double[][] xTrain = toDouble(train.images());
        double[] yTrain = Arrays.stream(train.labels()).asDoubleStream().toArray();
        Model mlp = Models.mlpClassifier(xTrain, yTrain, 2, 64, steps, seed);
        int[] pred = argmax(mlp.forward(Var.of(test.images())).value());
        double[][] hTrain = toDouble(ConceptActivationVectors.activations(mlp, 0, train.images()));
        double[][] hTest = toDouble(ConceptActivationVectors.activations(mlp, 0, test.images()));
        double probeAcc = 0;
        double[][] probeScores = new double[test.size()][k];
        for (int q = 0; q < k; q++) {
            double[] target = new double[train.size()];
            for (int i = 0; i < train.size(); i++) {
                target[i] = train.concepts()[i][q];
            }
            PenalizedRegression.Fit probe = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                    .family(PenalizedRegression.Family.LOGISTIC).fit(hTrain, target, 1e-3);
            double[] z = probe.predict(hTest);
            for (int i = 0; i < test.size(); i++) {
                probeScores[i][q] = 1 / (1 + Math.exp(-z[i]));
                probeAcc += (z[i] > 0 ? 1 : 0) == test.concepts()[i][q] ? 1 : 0;
            }
        }
        double acc = accuracy(pred, test.labels());
        double[] flat = new double[k + 1];
        Arrays.fill(flat, acc);
        rows.add(new ConceptRow("MLP+probes", acc, probeAcc / (test.size() * k), flat,
                ConceptLeakage.ctl(probeScores, toDouble(ct), test.labels(), 8),
                ConceptLeakage.icl(probeScores, toDouble(ct), 8, ConceptLeakage.Estimator.BINNED)));
        return rows;
    }

    private static double[][] toDouble(Tensor t) {
        int n = t.shape().dim(0);
        int d = (int) (t.shape().size() / n);
        double[] v = t.toDoubleArray();
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    private static double[][] toDouble(float[][] f) {
        double[][] out = new double[f.length][];
        for (int i = 0; i < f.length; i++) {
            out[i] = new double[f[i].length];
            for (int j = 0; j < f[i].length; j++) {
                out[i][j] = f[i][j];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ 13.1.9 SENN

    /**
     * One SENN-table row.
     *
     * @param model the model and explanation
     * @param mse test mean squared error
     * @param coefficientCorrelation mean correlation of the explanation with the true local coefficients
     * @param lipschitz the mean local Lipschitz estimate of the explanation (lower is more stable)
     */
    public record SennRow(String model, double mse, double coefficientCorrelation, double lipschitz) {
    }

    /**
     * The SENN table on {@link PlantedFunctions#varyingCoefficients}.
     *
     * @param steps training steps of the networks
     * @param seed the seed
     * @return rows for SENN (lambda 0 and 0.02, and autoencoder concepts), the MLP with gradients and with LIME, NAM and EBM
     */
    public static List<SennRow> senn(int steps, long seed) {
        PlantedFunctions.Table train = PlantedFunctions.varyingCoefficients(1500, seed);
        PlantedFunctions.Table test = PlantedFunctions.varyingCoefficients(300, seed + 1);
        Tensor tx = Models.tensor(train.x());
        Tensor ty = Models.labels(train.y()).reshape(train.rows(), 1);
        double[][] ex = Arrays.copyOf(test.x(), 60);
        List<SennRow> rows = new ArrayList<>();
        for (float lambda : new float[] {0f, 0.02f}) {
            SelfExplainingNetwork senn = new SelfExplainingNetwork(5, 0, 32, seed);
            senn.fit(tx, ty, false, lambda, steps, seed);
            Function<double[][], double[]> f = rowsIn -> senn.forward(Var.of(Models.tensor(rowsIn))).value()
                    .toDoubleArray();
            Function<double[], double[]> theta = x -> senn.relevances(Var.of(Models.tensor(new double[][] {x})))
                    .value().toDoubleArray();
            rows.add(new SennRow("SENN lambda=" + lambda, mse(f, test), correlation(theta, ex),
                    lipschitz(theta, ex, seed)));
        }
        SelfExplainingNetwork ae = new SelfExplainingNetwork(5, 3, 32, seed);
        ae.fit(tx, ty, false, 0.02f, steps, seed);
        rows.add(new SennRow("SENN autoencoder k=3", mse(rowsIn -> ae.forward(Var.of(Models.tensor(rowsIn))).value()
                .toDoubleArray(), test), Double.NaN, Double.NaN));
        Model mlp = Models.mlpRegressor(train.x(), train.y(), 32, steps, seed);
        Function<double[][], double[]> fm = Models.output(mlp);
        Function<double[], double[]> grad = x -> gradient(fm, x, 1e-3);
        rows.add(new SennRow("MLP+gradient", mse(fm, test), correlation(grad, ex), lipschitz(grad, ex, seed)));
        ModelAgnosticExplainer lime = new ModelAgnosticExplainer(fm);
        double[] scale = {0.3, 0.3, 0.3, 0.3, 0.3};
        Function<double[], double[]> limeCoef = x -> {
            double[] a = lime.lime(x, scale, 300, 0.75 * Math.sqrt(5), new SplittableRandom(seed));
            double[] out = new double[5];
            for (int j = 0; j < 5; j++) {
                out[j] = a[j] / scale[j];
            }
            return out;
        };
        double[][] few = Arrays.copyOf(ex, 15);
        rows.add(new SennRow("MLP+LIME", mse(fm, test), correlation(limeCoef, ex), lipschitz(limeCoef, few, seed)));
        Model nam = Model.sequential().input(5).add(NeuralAdditiveModel.spec(16)).seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(train.rows() - 64 + 1);
            nam.forward(Var.of(tx.slice(0, from, from + 64)), true).sub(ty.slice(0, from, from + 64)).square().mean()
                    .backward();
            opt.step(nam.parameters());
        }
        Function<double[][], double[]> fn = Models.output(nam);
        Function<double[], double[]> namSlope = x -> gradient(fn, x, 1e-2);
        rows.add(new SennRow("NAM slope", mse(fn, test), correlation(namSlope, ex), lipschitz(namSlope, ex, seed)));
        ExplainableBoosting ebm = ExplainableBoosting.fit(train.x(), train.y(),
                GradientBoostedTrees.Objective.REGRESSION.loss(), ExplainableBoosting.Settings.defaults());
        Function<double[][], double[]> fe = rowsIn -> Arrays.stream(rowsIn).mapToDouble(ebm::predict).toArray();
        Function<double[], double[]> ebmSlope = x -> gradient(fe, x, 0.25);
        rows.add(new SennRow("EBM slope", mse(fe, test), correlation(ebmSlope, ex), lipschitz(ebmSlope, ex, seed)));
        return rows;
    }

    private static double mse(Function<double[][], double[]> f, PlantedFunctions.Table t) {
        double[] p = f.apply(t.x());
        double s = 0;
        for (int i = 0; i < p.length; i++) {
            s += (p[i] - t.y()[i]) * (p[i] - t.y()[i]);
        }
        return s / p.length;
    }

    private static double[] gradient(Function<double[][], double[]> f, double[] x, double h) {
        double[][] probe = new double[2 * x.length][];
        for (int j = 0; j < x.length; j++) {
            probe[2 * j] = x.clone();
            probe[2 * j][j] += h;
            probe[2 * j + 1] = x.clone();
            probe[2 * j + 1][j] -= h;
        }
        double[] v = f.apply(probe);
        double[] g = new double[x.length];
        for (int j = 0; j < x.length; j++) {
            g[j] = (v[2 * j] - v[2 * j + 1]) / (2 * h);
        }
        return g;
    }

    /** Mean over the four modulated features of the correlation between explanation and true coefficient. */
    private static double correlation(Function<double[], double[]> explain, double[][] rows) {
        double[][] e = new double[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            e[i] = explain.apply(rows[i]);
        }
        double s = 0;
        int counted = 0;
        for (int j = 0; j < 4; j++) {
            if (j == 2) {
                continue;   // beta_2 is constant, so its correlation is undefined
            }
            double[] a = new double[rows.length];
            double[] b = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                a[i] = e[i][j + 1];
                b[i] = PlantedFunctions.coefficients(rows[i][0])[j];
            }
            s += Models.pearson(a, b);
            counted++;
        }
        return s / counted;
    }

    /** The mean over rows of the largest {@code |e(x + d) - e(x)| / |d|} over 10 perturbations of radius 0.1. */
    private static double lipschitz(Function<double[], double[]> explain, double[][] rows, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        for (double[] x : rows) {
            double[] e0 = explain.apply(x);
            double worst = 0;
            for (int k = 0; k < 10; k++) {
                double[] z = x.clone();
                double norm = 0;
                for (int j = 0; j < z.length; j++) {
                    double d = 0.1 * r.nextGaussian() / Math.sqrt(z.length);
                    z[j] += d;
                    norm += d * d;
                }
                double[] e1 = explain.apply(z);
                double diff = 0;
                for (int j = 0; j < e0.length; j++) {
                    diff += (e1[j] - e0[j]) * (e1[j] - e0[j]);
                }
                worst = Math.max(worst, Math.sqrt(diff / norm));
            }
            total += worst / rows.length;
        }
        return total;
    }

    // ------------------------------------------------------------------ 13.1.16 guided training

    /** The guided-training methods. */
    public enum Guided {
        /** Plain cross-entropy. */
        ERM,
        /** Right for the right reasons, masks on 10% of the training images. */
        RRR_10_PERCENT,
        /** Right for the right reasons, masks on every image. */
        RRR_ALL,
        /** The integrated-gradients penalty, masks on every image. */
        RRR_INTEGRATED,
        /** GroupDRO with label-by-patch groups. */
        GROUP_DRO,
        /** Just train twice. */
        JTT,
        /** ERM, then the last layer refitted on group-balanced data. */
        LAST_LAYER,
        /** Patch-randomising augmentation. */
        AUGMENT
    }

    /**
     * Scores of one guided-training method.
     *
     * @param brokenAccuracy test accuracy when the patch is independent of the label
     * @param alignedAccuracy test accuracy when the patch always matches the label
     * @param patchSaliency mean share of input-gradient mass on the patch
     */
    public record GuidedScores(double brokenAccuracy, double alignedAccuracy, double patchSaliency) {
    }

    private static Model guidedMlp(long seed) {
        return Model.sequential().input(2 * SIDE * SIDE).dense(64, Activation.RELU).dense(2, Activation.IDENTITY)
                .seed(seed).build();
    }

    /**
     * Runs the guided-training table.
     *
     * @param methods the methods to run
     * @param steps training steps
     * @param seed the seed
     * @return scores per method
     */
    public static Map<Guided, GuidedScores> guided(Guided[] methods, int steps, long seed) {
        ConceptShapes.Task task = ConceptShapes.Task.BOOLEAN;
        ConceptShapes.Batch train = SHAPES.generate(task, 2000, 0.95, seed);
        ConceptShapes.Batch broken = SHAPES.generate(task, 600, 0.5, seed + 1);
        ConceptShapes.Batch aligned = SHAPES.generate(task, 300, 1.0, seed + 2);
        Tensor y = ids(train.labels());
        int n = train.size();
        int d = 2 * SIDE * SIDE;
        float[] annotation = SHAPES.patchAnnotation();
        int[] group = new int[n];
        for (int i = 0; i < n; i++) {
            group[i] = train.labels()[i] * 2 + (train.patch()[i] ? 1 : 0);
        }
        Map<Guided, GuidedScores> out = new EnumMap<>(Guided.class);
        for (Guided method : methods) {
            Model m = guidedMlp(seed);
            Optimizer opt = Optimizer.adam(2e-3f);
            SplittableRandom r = new SplittableRandom(seed);
            Tensor images = method == Guided.AUGMENT ? randomisePatch(train.images(), r) : train.images();
            double share = method == Guided.RRR_10_PERCENT ? 0.1 : 1.0;
            ExplanationRegularizer reg = new ExplanationRegularizer(method == Guided.RRR_INTEGRATED
                    ? ExplanationRegularizer.Penalty.INTEGRATED_GRADIENTS : ExplanationRegularizer.Penalty.INPUT_GRADIENT,
                    method == Guided.RRR_INTEGRATED ? 100f : 1000f, 4);
            GroupRobustTraining.GroupDro dro = new GroupRobustTraining.GroupDro(4, 0.01);
            int batch = 64;
            int stage = method == Guided.JTT || method == Guided.LAST_LAYER ? steps / 2 : steps;
            for (int s = 0; s < stage; s++) {
                int from = r.nextInt(n - batch + 1);
                Tensor xb = images.slice(0, from, from + batch);
                Tensor yb = y.slice(0, from, from + batch);
                Var loss;
                switch (method) {
                    case RRR_10_PERCENT, RRR_ALL, RRR_INTEGRATED -> {
                        float[] mask = new float[batch * d];
                        for (int i = 0; i < batch; i++) {
                            if ((from + i) % 10 < share * 10) {
                                System.arraycopy(annotation, 0, mask, i * d, d);
                            }
                        }
                        loss = reg.loss(m, xb, yb, Tensor.of(Shape.of(batch, d), mask));
                    }
                    case GROUP_DRO -> {
                        Var per = Functional.crossEntropyPerRow(m.forward(Var.of(xb), true), yb);
                        List<Var> losses = new ArrayList<>();
                        for (int g = 0; g < 4; g++) {
                            float[] sel = new float[batch];
                            int count = 0;
                            for (int i = 0; i < batch; i++) {
                                if (group[from + i] == g) {
                                    sel[i] = 1;
                                    count++;
                                }
                            }
                            losses.add(count == 0 ? null : per.mul(Tensor.of(Shape.of(batch), sel)).sum().div(count));
                        }
                        loss = dro.loss(losses);
                    }
                    default -> loss = Functional.crossEntropy(m.forward(Var.of(xb), true), yb);
                }
                loss.backward();
                opt.step(m.parameters());
            }
            if (method == Guided.JTT) {
                int[] pred = argmax(m.forward(Var.of(train.images())).value());
                GroupRobustTraining.JustTrainTwice jtt = new GroupRobustTraining.JustTrainTwice(
                        GroupRobustTraining.JustTrainTwice.errorSet(pred, train.labels()), 20);
                m = guidedMlp(seed + 1);
                opt = Optimizer.adam(2e-3f);
                for (int s = 0; s < steps; s++) {
                    int[] idx = jtt.sample(batch, r);
                    Functional.crossEntropy(m.forward(Var.of(select(train.images(), idx)), true),
                            select(y, idx)).backward();
                    opt.step(m.parameters());
                }
            } else if (method == Guided.LAST_LAYER) {
                // retrain the last layer on group-balanced batches
                List<List<Integer>> byGroup = new ArrayList<>();
                for (int g = 0; g < 4; g++) {
                    byGroup.add(new ArrayList<>());
                }
                for (int i = 0; i < n; i++) {
                    byGroup.get(group[i]).add(i);
                }
                List<Var> last = m.layers().get(m.layers().size() - 1).parameters();
                Optimizer lastOpt = Optimizer.adam(5e-3f);
                for (int s = 0; s < steps; s++) {
                    int[] idx = new int[batch];
                    for (int i = 0; i < batch; i++) {
                        List<Integer> pool = byGroup.get(i % 4);
                        idx[i] = pool.get(r.nextInt(pool.size()));
                    }
                    Functional.crossEntropy(m.forward(Var.of(select(train.images(), idx)), true),
                            select(y, idx)).backward();
                    lastOpt.step(last);
                    m.parameters().forEach(Var::zeroGrad);
                }
            }
            double brokenAcc = accuracy(argmax(m.forward(Var.of(broken.images())).value()), broken.labels());
            double alignedAcc = accuracy(argmax(m.forward(Var.of(aligned.images())).value()), aligned.labels());
            double mass = 0;
            for (int i = 0; i < 50; i++) {
                float[] g = Attribution.saliency(m, broken.images().slice(0, i, i + 1).toArray(), broken.labels()[i]);
                double on = 0;
                double all = 0;
                for (int p = 0; p < d; p++) {
                    all += Math.abs(g[p]);
                    on += annotation[p] > 0 ? Math.abs(g[p]) : 0;
                }
                mass += all > 0 ? on / all / 50 : 0;
            }
            out.put(method, new GuidedScores(brokenAcc, alignedAcc, mass));
        }
        return out;
    }

    private static Tensor select(Tensor t, int[] idx) {
        int width = (int) (t.shape().size() / t.shape().dim(0));
        float[] v = t.toArray();
        float[] out = new float[idx.length * width];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(v, idx[i] * width, out, i * width, width);
        }
        int[] dims = t.shape().toArray();
        dims[0] = idx.length;
        return Tensor.of(Shape.of(dims), out);
    }

    /** Adds or removes the patch independently of the label (each with probability one half). */
    private static Tensor randomisePatch(Tensor images, SplittableRandom r) {
        int n = images.shape().dim(0);
        int area = SIDE * SIDE;
        float[] v = images.toArray();
        int[] cells = {SIDE - 2, SIDE - 1, 2 * SIDE - 2, 2 * SIDE - 1};
        for (int i = 0; i < n; i++) {
            boolean on = r.nextBoolean();
            for (int ch = 0; ch < 2; ch++) {
                for (int p : cells) {
                    v[(i * 2 + ch) * area + p] = on ? 1f : (float) (0.05 * r.nextGaussian());
                }
            }
        }
        return Tensor.of(images.shape(), v);
    }
}
