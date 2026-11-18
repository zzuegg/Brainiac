package io.github.zzuegg.brainiac.experiments.uncertainty;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.SetPredictor;
import io.github.zzuegg.brainiac.experimental.calibrate.ConformalTraining;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.labels.RobustLosses;
import io.github.zzuegg.brainiac.experimental.nn.SmoothedCrossEntropy;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.train.ConformalClassifier;
import io.github.zzuegg.brainiac.experiments.shift.ShiftHarness;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Conformal sets beyond LAC and APS (wave 11 item 11.4.4) and conformal training (11.4.5):
 *
 * <ul>
 *   <li>{@link #sets}: LAC and APS ({@link ConformalClassifier}) against tuned RAPS,
 *       Mondrian (class-conditional) LAC/APS and size-matched top-k sets on a fixture model:
 *       marginal and worst-class coverage, mean size and size-stratified coverage;</li>
 *   <li>{@link #longTail}: the same on long-tailed digits (training, calibration and test drawn
 *       with the same imbalance), where marginal methods under-cover tail classes;</li>
 *   <li>{@link #riskControl}: conformal risk control of the false-negative rate of multilabel
 *       "which shapes are present" sets on {@link ShapeScenes}, against a fixed 0.5 threshold;</li>
 *   <li>{@link #conformalTraining}: fine-tuning the fixture model with the ConfTr loss against
 *       continued cross-entropy, label smoothing and focal loss, all followed by LAC, APS and
 *       RAPS calibration.</li>
 * </ul>
 */
public final class ConformalSets {

    /**
     * One set method.
     *
     * @param method the method
     * @param coverage the marginal test coverage
     * @param worstClass the lowest per-class coverage
     * @param size the mean set size
     * @param sizeStratified the lowest coverage over set-size groups {1}, {2, 3}, {4+} with at
     *     least 20 samples
     */
    public record Row(String method, double coverage, double worstClass, double size, double sizeStratified) {
    }

    private ConformalSets() {
    }

    /**
     * Evaluates a set predictor.
     *
     * @param name the method
     * @param p the predictor
     * @param probs test probabilities
     * @param y test labels
     * @param classes the number of classes
     * @return the row
     */
    public static Row evaluate(String name, SetPredictor p, double[][] probs, int[] y, int classes) {
        double[] hit = new double[3];
        double[] n = new double[3];
        for (int i = 0; i < y.length; i++) {
            int[] s = p.predictSet(probs[i]);
            int g = s.length <= 1 ? 0 : s.length <= 3 ? 1 : 2;
            n[g]++;
            for (int c : s) {
                if (c == y[i]) {
                    hit[g]++;
                    break;
                }
            }
        }
        double ssc = 1;
        for (int g = 0; g < 3; g++) {
            if (n[g] >= 20) {
                ssc = Math.min(ssc, hit[g] / n[g]);
            }
        }
        return new Row(name, p.coverage(probs, y), p.worstClassCoverage(probs, y, classes), p.meanSetSize(probs), ssc);
    }

    /**
     * All set methods on calibration and test probabilities.
     *
     * @param cal calibration probabilities
     * @param calY calibration labels
     * @param test test probabilities
     * @param testY test labels
     * @param classes the number of classes
     * @param alpha the miscoverage
     * @return LAC, APS, RAPS, Mondrian LAC, Mondrian APS and top-k (k = APS's rounded mean size)
     */
    public static List<Row> allMethods(double[][] cal, int[] calY, double[][] test, int[] testY, int classes,
            double alpha) {
        List<Row> rows = new ArrayList<>();
        ConformalClassifier lac = ConformalClassifier.calibrate(cal, calY, alpha, ConformalClassifier.Score.LAC);
        ConformalClassifier aps = ConformalClassifier.calibrate(cal, calY, alpha, ConformalClassifier.Score.APS);
        rows.add(evaluate("lac", lac::predictSet, test, testY, classes));
        Row apsRow = evaluate("aps", aps::predictSet, test, testY, classes);
        rows.add(apsRow);
        io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.Tuned raps =
                io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.selectRaps(cal, calY, alpha, 0.3);
        rows.add(evaluate("raps", raps.predictor(), test, testY, classes));
        rows.add(evaluate("mondrian lac", io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.mondrian(cal,
                calY, classes, alpha, ConformalClassifier.Score.LAC), test, testY, classes));
        rows.add(evaluate("mondrian aps", io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.mondrian(cal,
                calY, classes, alpha, ConformalClassifier.Score.APS), test, testY, classes));
        int k = (int) Math.max(1, Math.round(apsRow.size()));
        rows.add(evaluate("top-" + k, io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.topK(k), test,
                testY, classes));
        return rows;
    }

    /**
     * The set methods on a fixture model.
     *
     * @param task the task
     * @param alpha the miscoverage
     * @param seed the seed
     * @return one row per method
     */
    public static List<Row> sets(UncertaintyFixture.Task task, double alpha, long seed) {
        UncertaintyFixture.Setup s = UncertaintyFixture.get(task, seed);
        return allMethods(softmax(s.validationLogits()), s.validation().y(), softmax(s.testLogits()), s.test().y(),
                s.classes(), alpha);
    }

    /**
     * The set methods on long-tailed digits.
     *
     * @param imbalance the head-to-tail ratio, such as 10 or 100
     * @param alpha the miscoverage
     * @param steps training steps
     * @param seed the seed
     * @return one row per method
     */
    public static List<Row> longTail(double imbalance, double alpha, int steps, long seed) {
        int[] cls = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        ShiftSuites.Split train = tail(ShiftSuites.digits(6000, cls, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN,
                0.25, seed), 300, imbalance, seed);
        ShiftSuites.Split cal = tail(ShiftSuites.digits(8000, cls, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN,
                0.25, seed + 1), 400, imbalance, seed + 1);
        ShiftSuites.Split test = tail(ShiftSuites.digits(8000, cls, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN,
                0.25, seed + 2), 400, imbalance, seed + 2);
        FeatureClassifier m = ShiftHarness.mlp(ShiftSuites.PIXELS, 10, seed);
        ShiftHarness.train(m, train, steps, 3e-3f, seed);
        return allMethods(softmax(UncertaintyFixture.logits(m, cal.x())), cal.y(),
                softmax(UncertaintyFixture.logits(m, test.x())), test.y(), 10, alpha);
    }

    private static ShiftSuites.Split tail(ShiftSuites.Split pool, int max, double ratio, long seed) {
        return pool.rows(LabelCorruption.longTail(pool.y(), 10, max, ratio, seed));
    }

    /**
     * Risk control of the multilabel false-negative rate.
     *
     * @param alpha the target false-negative rate
     * @param steps training steps of the multilabel MLP
     * @param seed the seed
     * @return {@code {crc FNR, crc mean set size, crc λ, threshold-0.5 FNR, threshold-0.5 size}}
     */
    public static double[] riskControl(double alpha, int steps, long seed) {
        ShapeScenes gen = new ShapeScenes(12, 0.15f);
        ShapeScenes.Scene[] train = gen.scenes(1500, 1, 3, ShapeScenes.Layout.SPARSE, seed);
        ShapeScenes.Scene[] cal = gen.scenes(800, 1, 3, ShapeScenes.Layout.SPARSE, seed + 1);
        ShapeScenes.Scene[] test = gen.scenes(800, 1, 3, ShapeScenes.Layout.SPARSE, seed + 2);
        int k = ShapeScenes.CLASSES;
        int d = 144;
        FeatureClassifier m = FeatureClassifier.mlp(d, new int[] {64, 64}, k, seed);
        float[] x = ShapeScenes.images(train).toArray();
        float[] t = present(train).toArray();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = 64;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[b * d];
            float[] bt = new float[b * k];
            for (int i = 0; i < b; i++) {
                int row = r.nextInt(train.length);
                System.arraycopy(x, row * d, bx, i * d, d);
                System.arraycopy(t, row * k, bt, i * k, k);
            }
            Var z = m.logits(Var.of(Tensor.of(Shape.of(b, d), bx)), true);
            Tensor tb = Tensor.of(Shape.of(b, k), bt);
            // binary cross-entropy with logits: softplus(z) - t z
            z.softplus().sub(z.mul(tb)).mean().backward();
            opt.step(m.parameters());
        }
        double[][] pc = sigmoid(UncertaintyFixture.logits(m, ShapeScenes.images(cal).reshape(cal.length, d)));
        double[][] pt = sigmoid(UncertaintyFixture.logits(m, ShapeScenes.images(test).reshape(test.length, d)));
        boolean[][] yc = truth(cal);
        boolean[][] yt = truth(test);
        double lambda = io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.multilabelFnr(pc, yc, alpha);
        return new double[] {fnr(pt, yt, lambda), size(pt, lambda), lambda, fnr(pt, yt, 0.5), size(pt, 0.5)};
    }

    private static double fnr(double[][] p, boolean[][] y, double lambda) {
        double s = 0;
        for (int i = 0; i < p.length; i++) {
            s += io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets.falseNegativeRate(p[i], y[i], lambda);
        }
        return s / p.length;
    }

    private static double size(double[][] p, double lambda) {
        double s = 0;
        for (double[] row : p) {
            for (double v : row) {
                s += v >= 1 - lambda ? 1 : 0;
            }
        }
        return s / p.length;
    }

    private static Tensor present(ShapeScenes.Scene[] scenes) {
        int k = ShapeScenes.CLASSES;
        float[] v = new float[scenes.length * k];
        for (int i = 0; i < scenes.length; i++) {
            for (int c : scenes[i].classes()) {
                v[i * k + c] = 1;
            }
        }
        return Tensor.of(Shape.of(scenes.length, k), v);
    }

    private static boolean[][] truth(ShapeScenes.Scene[] scenes) {
        boolean[][] out = new boolean[scenes.length][ShapeScenes.CLASSES];
        for (int i = 0; i < scenes.length; i++) {
            for (int c : scenes[i].classes()) {
                out[i][c] = true;
            }
        }
        return out;
    }

    private static double[][] sigmoid(double[][] z) {
        double[][] out = new double[z.length][];
        for (int i = 0; i < z.length; i++) {
            out[i] = new double[z[i].length];
            for (int j = 0; j < z[i].length; j++) {
                out[i][j] = 1 / (1 + Math.exp(-z[i][j]));
            }
        }
        return out;
    }

    /**
     * One fine-tuning loss followed by conformal calibration.
     *
     * @param method the training loss
     * @param accuracy test accuracy
     * @param lacSize LAC mean set size at the target coverage
     * @param apsSize APS mean set size
     * @param rapsSize RAPS mean set size
     * @param lacCoverage LAC test coverage
     */
    public record TrainingRow(String method, double accuracy, double lacSize, double apsSize, double rapsSize,
            double lacCoverage) {
    }

    /**
     * Fine-tunes copies of the fixture model with different losses, then calibrates.
     *
     * @param task the fixture task
     * @param alpha the miscoverage
     * @param steps fine-tuning steps
     * @param seed the seed
     * @return rows for the untouched model, cross-entropy, label smoothing, focal loss and ConfTr
     */
    public static List<TrainingRow> conformalTraining(UncertaintyFixture.Task task, double alpha, int steps,
            long seed) {
        UncertaintyFixture.Setup s = UncertaintyFixture.get(task, seed);
        List<Tensor> start = s.model().snapshot();
        List<TrainingRow> rows = new ArrayList<>();
        rows.add(trainingRow("pretrained", s.model(), s, alpha));
        ConformalTraining.Settings conf = new ConformalTraining.Settings((float) alpha, 0.1f, 1f, 0f, 0.1f, 1f);
        List<String> names = List.of("ce", "label smoothing", "focal", "conftr");
        List<BiFunction<Var, Tensor, Var>> losses = List.of(
                Functional::crossEntropy,
                (z, y) -> new SmoothedCrossEntropy(0.1f).compute(z.softmax(1), y),
                (z, y) -> RobustLosses.focal(2f).compute(z, y),
                (z, y) -> ConformalTraining.loss(z, y, conf));
        FeatureClassifier m = s.model();
        try {
            for (int i = 0; i < names.size(); i++) {
                m.restore(start);
                ShiftHarness.train(m, s.train(), steps, 1e-3f, seed + 31, losses.get(i));
                rows.add(trainingRow(names.get(i), m, s, alpha));
            }
        } finally {
            m.restore(start);
        }
        return rows;
    }

    private static TrainingRow trainingRow(String name, FeatureClassifier m, UncertaintyFixture.Setup s,
            double alpha) {
        double[][] cal = softmax(UncertaintyFixture.logits(m, s.validation().x()));
        double[][] test = softmax(UncertaintyFixture.logits(m, s.test().x()));
        int[] cy = s.validation().y();
        int[] ty = s.test().y();
        ConformalClassifier lac = ConformalClassifier.calibrate(cal, cy, alpha, ConformalClassifier.Score.LAC);
        ConformalClassifier aps = ConformalClassifier.calibrate(cal, cy, alpha, ConformalClassifier.Score.APS);
        SetPredictor raps = io.github.zzuegg.brainiac.experimental.calibrate.ConformalSets
                .selectRaps(cal, cy, alpha, 0.3).predictor();
        return new TrainingRow(name, io.github.zzuegg.brainiac.experimental.train.Calibration.accuracy(test, ty),
                lac.meanSetSize(test), aps.meanSetSize(test), raps.meanSetSize(test), lac.coverage(test, ty));
    }

    static double[][] softmax(double[][] z) {
        return io.github.zzuegg.brainiac.experimental.train.Calibration.withTemperature(z, 1);
    }

    /**
     * Prints the tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (UncertaintyFixture.Task t : UncertaintyFixture.Task.values()) {
            sets(t, 0.1, 1).forEach(r -> System.out.println(t + " " + r));
        }
        for (double imb : new double[] {10, 100}) {
            longTail(imb, 0.1, 600, 1).forEach(r -> System.out.println("long tail " + imb + " " + r));
        }
        System.out.println(Arrays.toString(riskControl(0.1, 600, 1)));
        conformalTraining(UncertaintyFixture.Task.DIGITS, 0.1, 200, 1).forEach(System.out::println);
    }
}
