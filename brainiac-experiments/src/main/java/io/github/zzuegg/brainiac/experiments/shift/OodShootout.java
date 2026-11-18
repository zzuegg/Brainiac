package io.github.zzuegg.brainiac.experiments.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.BatchMixing;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.ood.ActivationShaping;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.ood.FeatureScores;
import io.github.zzuegg.brainiac.experimental.ood.LogitScores;
import io.github.zzuegg.brainiac.experimental.ood.OodMetrics;
import io.github.zzuegg.brainiac.experimental.ood.OpenMax;
import io.github.zzuegg.brainiac.experimental.ood.OutlierExposure;
import io.github.zzuegg.brainiac.experimental.train.ConformalClassifier;
import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 10 items 10.2.1-10.2.5: out-of-distribution detection and open-set recognition on
 * {@link ShiftSuites}. One MLP per family and seed is trained once ({@link ShiftHarness#ood})
 * and every post-hoc score reuses it:
 *
 * <ul>
 *   <li>{@link #postHoc}: logit scores (MSP, max logit, energy, ODIN), feature scores
 *       (Mahalanobis, deep k-NN, ViM), activation shaping (ReAct, ASH-P/B/S, SCALE,
 *       AdaSCALE-A/L) and a {@link GaussianMixture} on raw pixels, each scored by AUROC and FPR
 *       at 95% TPR on near OOD (held-out classes) and far OOD (noise and stripes), with the
 *       in-distribution accuracy after shaping and the forward passes per input.</li>
 *   <li>{@link #uncertaintyBaseline}: {@link UncertaintyBenchmark}'s OOD AUROC on the same
 *       split (shapes 0-3 against 4-5) for deep ensembles, MC dropout, SWAG, SNGP, DUQ and
 *       last-layer Laplace.</li>
 *   <li>{@link #outlierExposure}: retraining with auxiliary outliers (blobs and checkerboards,
 *       disjoint from the test OOD sets) under the uniform-target and energy-margin losses,
 *       against plain and mixup training at equal steps.</li>
 *   <li>{@link #openSet}: 6 known and 4 unknown digits; OpenMax against the max logit of a
 *       briefly and a well-trained classifier (4x steps, label smoothing, augmentation) and
 *       conformal set size.</li>
 * </ul>
 */
public final class OodShootout {

    /**
     * One score's detection quality.
     *
     * @param name the score
     * @param nearAuroc AUROC against held-out classes
     * @param nearFpr95 FPR at 95% TPR against held-out classes
     * @param farAuroc AUROC against noise and stripes
     * @param farFpr95 FPR at 95% TPR against noise and stripes
     * @param accuracy in-distribution accuracy of the (shaped) classifier, NaN if it does not classify
     * @param passes forward passes per input (a backward counts as one)
     */
    public record Row(String name, double nearAuroc, double nearFpr95, double farAuroc, double farFpr95,
            double accuracy, int passes) {
    }

    /**
     * Open-set recognition scores.
     *
     * @param closedAccuracy accuracy on the known classes
     * @param maxLogitAuroc known-vs-unknown AUROC of the max logit
     * @param openMaxAuroc known-vs-unknown AUROC of {@code 1 - p(unknown)}
     * @param openMaxAccuracy accuracy over known and unknown inputs with OpenMax's unknown class
     * @param maxLogitAccuracy the same with "unknown" below the max-logit threshold at 95% TPR
     * @param conformalAuroc AUROC of minus the conformal set size (empty sets count as size 10)
     */
    public record OpenSetScores(double closedAccuracy, double maxLogitAuroc, double openMaxAuroc,
            double openMaxAccuracy, double maxLogitAccuracy, double conformalAuroc) {
    }

    private OodShootout() {
    }

    /**
     * Every post-hoc score on the cached classifier.
     *
     * @param family shapes or digits
     * @param seed the seed
     * @return one row per score
     */
    public static List<Row> postHoc(ShiftSuites.Family family, long seed) {
        ShiftHarness.OodSetup setup = ShiftHarness.ood(family, seed);
        ShiftSuites.OodSuite s = setup.suite();
        FeatureClassifier m = setup.model();
        Tensor test = s.test().x();
        List<Row> rows = new ArrayList<>();
        double acc = m.accuracy(test, s.test().y());
        rows.add(row("MSP", x -> LogitScores.msp(m.logits(x)), s, acc, 1));
        rows.add(row("max logit", x -> LogitScores.maxLogit(m.logits(x)), s, acc, 1));
        rows.add(row("energy", x -> LogitScores.energy(m.logits(x), 1f), s, acc, 1));
        rows.add(row("ODIN", x -> LogitScores.odin(v -> m.logits(v, false), x, 1000f, 0.01f), s, acc, 3));
        FeatureScores fs = FeatureScores.fit(m.features(s.train().x()), s.train().y(), m.head(), 10, 16);
        rows.add(row("Mahalanobis", x -> fs.mahalanobis(m.features(x)), s, acc, 1));
        rows.add(row("deep k-NN", x -> fs.knn(m.features(x)), s, acc, 1));
        rows.add(row("ViM", x -> fs.vim(m.features(x)), s, acc, 1));
        ActivationShaping shaping = ActivationShaping.fit(m, s.train().x(), s.validation().x().slice(0, 0, 20), 0.9);
        for (ActivationShaping.Method method : ActivationShaping.Method.values()) {
            if (method == ActivationShaping.Method.NONE) {
                continue;
            }
            boolean ada = method == ActivationShaping.Method.ADASCALE_A || method == ActivationShaping.Method.ADASCALE_L;
            double shapedAcc = ShiftHarness.accuracy(shaping.logits(method, test).argmax(1).toIntArray(), s.test().y());
            rows.add(row(method.name(), x -> shaping.score(method, x), s, shapedAcc, ada ? 3 : 1));
        }
        GaussianMixture gmm = GaussianMixture.fit(ShiftHarness.rows(s.train().x()), 8, 50, 1e-2, seed);
        rows.add(row("GMM on pixels", x -> {
            double[][] r = ShiftHarness.rows(x);
            double[] out = new double[r.length];
            for (int i = 0; i < r.length; i++) {
                out[i] = gmm.logDensity(r[i]);
            }
            return out;
        }, s, Double.NaN, 0));
        return rows;
    }

    private static Row row(String name, Function<Tensor, double[]> score, ShiftSuites.OodSuite s, double acc,
            int passes) {
        double[] in = score.apply(s.test().x());
        double[] near = score.apply(s.near());
        double[] far = score.apply(s.far());
        return new Row(name, OodMetrics.auroc(in, near), OodMetrics.fpr95(in, near), OodMetrics.auroc(in, far),
                OodMetrics.fpr95(in, far), acc, passes);
    }

    /**
     * The wave 4 uncertainty methods' OOD AUROC on shapes 0-3 against 4-5.
     *
     * @param method the method
     * @param seed the seed
     * @return its AUROC (predictive entropy unless the method has its own score)
     */
    public static double uncertaintyBaseline(UncertaintyBenchmark.Method method, long seed) {
        return UncertaintyBenchmark.classification(method, seed).oodAuroc();
    }

    /** The training regime of {@link #outlierExposure}. */
    public enum Training {
        /** cross-entropy only */
        PLAIN,
        /** mixup (alpha 0.2) */
        MIXUP,
        /** outlier exposure with the uniform target (weight 0.5) */
        OE_UNIFORM,
        /** outlier exposure with energy margins (weight 0.1) */
        OE_ENERGY
    }

    /**
     * Trains a shape classifier under a regime and scores MSP and energy.
     *
     * @param training the regime
     * @param steps Adam steps
     * @param seed the seed
     * @return the MSP row and the energy row
     */
    public static List<Row> outlierExposure(Training training, int steps, long seed) {
        ShiftSuites.OodSuite s = ShiftSuites.ood(ShiftSuites.Family.SHAPES, 1200, 400, seed);
        FeatureClassifier m = ShiftHarness.mlp(ShiftSuites.PIXELS, s.classes(), seed);
        Tensor aux = ShiftSuites.auxiliaryOutliers(2000, seed + 77);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        Loss oe = training == Training.OE_ENERGY ? OutlierExposure.loss(OutlierExposure.Variant.ENERGY, 0.1f, -7f, -2f)
                : OutlierExposure.loss(OutlierExposure.Variant.UNIFORM, 0.5f, 0, 0);
        int k = s.classes();
        for (int step = 0; step < steps; step++) {
            ShiftSuites.Split b = ShiftHarness.batch(s.train(), 64, r);
            Var loss;
            switch (training) {
                case PLAIN -> loss = Loss.crossEntropyFromLogits(0).compute(m.logits(Var.of(b.x()), true), b.ids());
                case MIXUP -> {
                    Tensor[] mixed = BatchMixing.mixup(b.x(), oneHot(b.y(), k), 0.2f, r);
                    loss = Loss.crossEntropyFromLogits(0).compute(m.logits(Var.of(mixed[0]), true), mixed[1]);
                }
                default -> {
                    float[] rows = new float[64];
                    for (int i = 0; i < rows.length; i++) {
                        rows[i] = r.nextInt(aux.shape().dim(0));
                    }
                    Tensor outs = aux.gather(0, Tensor.of(Shape.of(64), rows));
                    float[] ids = new float[128];
                    float[] bi = b.ids().toArray();
                    for (int i = 0; i < 128; i++) {
                        ids[i] = i < 64 ? bi[i] : -1;
                    }
                    loss = oe.compute(m.logits(Var.of(Tensor.concat(0, b.x(), outs)), true),
                            Tensor.of(Shape.of(128), ids));
                }
            }
            loss.backward();
            opt.step(m.parameters());
        }
        double acc = m.accuracy(s.test().x(), s.test().y());
        return List.of(row(training + " MSP", x -> LogitScores.msp(m.logits(x)), s, acc, 1),
                row(training + " energy", x -> LogitScores.energy(m.logits(x), 1f), s, acc, 1));
    }

    private static Tensor oneHot(int[] y, int k) {
        float[] v = new float[y.length * k];
        for (int i = 0; i < y.length; i++) {
            v[i * k + y[i]] = 1;
        }
        return Tensor.of(Shape.of(y.length, k), v);
    }

    /**
     * Open-set recognition on digits: 6 known classes, the 4 others unknown.
     *
     * @param wellTrained 4x steps, label smoothing 0.1 and shift/brightness augmentation
     * @param steps the brief budget in Adam steps
     * @param seed the seed
     * @return the scores
     */
    public static OpenSetScores openSet(boolean wellTrained, int steps, long seed) {
        ShiftSuites.OodSuite s = ShiftSuites.ood(ShiftSuites.Family.DIGITS, 1200, 400, seed + 5);
        FeatureClassifier m = ShiftHarness.mlp(ShiftSuites.PIXELS, s.classes(), seed);
        ShapeImages augmenter = new ShapeImages(ShiftSuites.SIDE, 0.1f);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        Loss loss = Loss.crossEntropyFromLogits(wellTrained ? 0.1f : 0f);
        int total = wellTrained ? 4 * steps : steps;
        for (int step = 0; step < total; step++) {
            ShiftSuites.Split b = ShiftHarness.batch(s.train(), 64, r);
            Tensor x = b.x();
            if (wellTrained && step % 2 == 1) {
                x = augmenter.augment(x.reshape(64, 1, 8, 8), 1, r).reshape(64, ShiftSuites.PIXELS);
            }
            loss.compute(m.logits(Var.of(x), true), b.ids()).backward();
            opt.step(m.parameters());
        }
        Tensor known = s.test().x();
        Tensor unknown = s.near();
        Tensor lk = m.logits(known);
        Tensor lu = m.logits(unknown);
        double closed = m.accuracy(known, s.test().y());
        double[] mk = LogitScores.maxLogit(lk);
        double[] mu = LogitScores.maxLogit(lu);
        OpenMax om = OpenMax.fit(m.logits(s.train().x()), s.train().y(), 20, 2);
        double[][] pk = om.probabilities(lk);
        double[][] pu = om.probabilities(lu);
        int k = s.classes();
        int ok = 0;
        for (int i = 0; i < pk.length; i++) {
            ok += argmax(pk[i]) == s.test().y()[i] ? 1 : 0;
        }
        for (double[] p : pu) {
            ok += argmax(p) == k ? 1 : 0;
        }
        double openMaxAccuracy = ok / (double) (pk.length + pu.length);
        double[] sorted = mk.clone();
        java.util.Arrays.sort(sorted);
        double threshold = sorted[(int) (0.05 * sorted.length)];
        int[] predK = lk.argmax(1).toIntArray();
        int ok2 = 0;
        for (int i = 0; i < mk.length; i++) {
            ok2 += mk[i] >= threshold && predK[i] == s.test().y()[i] ? 1 : 0;
        }
        for (double v : mu) {
            ok2 += v < threshold ? 1 : 0;
        }
        double maxLogitAccuracy = ok2 / (double) (mk.length + mu.length);
        ConformalClassifier cc = ConformalClassifier.calibrate(ShiftHarness.probabilities(m.logits(s.validation().x())),
                s.validation().y(), 0.05, ConformalClassifier.Score.LAC);
        double[] ck = setScores(cc, ShiftHarness.probabilities(lk), k);
        double[] cu = setScores(cc, ShiftHarness.probabilities(lu), k);
        return new OpenSetScores(closed, OodMetrics.auroc(mk, mu), OodMetrics.auroc(om.knownScore(lk),
                om.knownScore(lu)), openMaxAccuracy, maxLogitAccuracy, OodMetrics.auroc(ck, cu));
    }

    private static double[] setScores(ConformalClassifier cc, double[][] probs, int k) {
        double[] out = new double[probs.length];
        for (int i = 0; i < probs.length; i++) {
            int size = cc.predictSet(probs[i]).length;
            out[i] = -(size == 0 ? k : size);
        }
        return out;
    }

    private static int argmax(double[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[b]) {
                b = i;
            }
        }
        return b;
    }

    /**
     * The long-budget run: post-hoc scores on both families over three seeds, the wave 4
     * uncertainty baselines, outlier exposure at 1,200 steps and open set at both budgets.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (ShiftSuites.Family f : ShiftSuites.Family.values()) {
            for (long seed = 1; seed <= 3; seed++) {
                for (Row r : postHoc(f, seed)) {
                    System.out.printf("%s seed %d %-14s near %.3f/%.3f far %.3f/%.3f acc %.3f passes %d%n", f, seed,
                            r.name(), r.nearAuroc(), r.nearFpr95(), r.farAuroc(), r.farFpr95(), r.accuracy(), r.passes());
                }
            }
        }
        for (UncertaintyBenchmark.Method m : UncertaintyBenchmark.Method.values()) {
            System.out.printf("UncertaintyBenchmark %s OOD AUROC %.3f%n", m, uncertaintyBaseline(m, 1));
        }
        for (Training t : Training.values()) {
            for (Row r : outlierExposure(t, 1200, 1)) {
                System.out.printf("%-20s near %.3f far %.3f acc %.3f%n", r.name(), r.nearAuroc(), r.farAuroc(),
                        r.accuracy());
            }
        }
        for (boolean well : new boolean[] {false, true}) {
            System.out.println((well ? "well-trained " : "brief ") + openSet(well, 300, 1));
        }
    }
}
