package io.github.zzuegg.brainiac.experiments.uncertainty;

import io.github.zzuegg.brainiac.experimental.calibrate.EnsembleTemperature;
import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrator;
import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;
import io.github.zzuegg.brainiac.experimental.calibrate.SplineCalibration;
import io.github.zzuegg.brainiac.experimental.calibrate.StructuredMatrixScaling;
import io.github.zzuegg.brainiac.experimental.calibrate.VennAbers;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experiments.shift.ShiftHarness;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * Post-hoc recalibration maps against temperature scaling (wave 11 items 11.4.1–11.4.3), on the
 * cached overconfident {@link UncertaintyFixture} models, at several validation sizes, in
 * distribution and on the shifted test set:
 *
 * <ul>
 *   <li>{@link #recalibration}: every {@link Recalibrators.Method}, structured matrix scaling
 *       (full and low rank), ensemble temperature scaling, spline calibration and one-vs-rest
 *       Venn–Abers; metrics: debiased top-label ℓ2 ECE and classwise ECE
 *       ({@link CalibrationEstimators}), NLL and accuracy;</li>
 *   <li>{@link #vennAbers}: binary Venn–Abers (inductive and cross) against isotonic, Platt and
 *       temperature scaling on the tabular task: NLL, Brier, ECE and interval width, in
 *       distribution and shifted.</li>
 * </ul>
 */
public final class RecalibrationShootout {

    /**
     * One map at one validation size.
     *
     * @param method the map
     * @param validation the validation rows it was fitted on
     * @param ece debiased top-label ℓ2 ECE on the test set
     * @param classwiseEce classwise ECE on the test set
     * @param nll test NLL
     * @param accuracy test accuracy
     * @param shiftedEce debiased ECE on the shifted test set
     * @param shiftedNll NLL on the shifted test set
     */
    public record Row(String method, int validation, double ece, double classwiseEce, double nll, double accuracy,
            double shiftedEce, double shiftedNll) {
    }

    private RecalibrationShootout() {
    }

    /** The maps, by name, as functions of validation logits and labels. */
    static Map<String, BiFunction<double[][], int[], Recalibrator>> maps() {
        Map<String, BiFunction<double[][], int[], Recalibrator>> m = new LinkedHashMap<>();
        for (Recalibrators.Method method : Recalibrators.Method.values()) {
            m.put(method.name().toLowerCase(java.util.Locale.ROOT), (z, y) -> Recalibrators.fit(method, z, y));
        }
        m.put("sms", (z, y) -> StructuredMatrixScaling.fit(z, y, StructuredMatrixScaling.Kind.FULL, 0));
        m.put("sms low-rank", (z, y) -> StructuredMatrixScaling.fit(z, y, StructuredMatrixScaling.Kind.LOW_RANK, 2));
        m.put("ensemble ts", EnsembleTemperature::fit);
        m.put("spline", SplineCalibration::fit);
        m.put("venn-abers", (z, y) -> {
            double[][] cal = AffineRows.softmax(z);
            return t -> VennAbers.oneVsRest(cal, y, AffineRows.softmax(t))[0];
        });
        return m;
    }

    /**
     * Every map at every validation size.
     *
     * @param task the fixture task
     * @param validationSizes the validation sizes (at most {@link UncertaintyFixture#VALIDATION})
     * @param methods the map names to run (null for all)
     * @param seed the seed
     * @return one row per map and size
     */
    public static List<Row> recalibration(UncertaintyFixture.Task task, int[] validationSizes, List<String> methods,
            long seed) {
        UncertaintyFixture.Setup s = UncertaintyFixture.get(task, seed);
        List<Row> rows = new ArrayList<>();
        for (int n : validationSizes) {
            double[][] vz = Arrays.copyOf(s.validationLogits(), n);
            int[] vy = Arrays.copyOf(s.validation().y(), n);
            for (Map.Entry<String, BiFunction<double[][], int[], Recalibrator>> e : maps().entrySet()) {
                if (methods != null && !methods.contains(e.getKey())) {
                    continue;
                }
                Recalibrator r = e.getValue().apply(vz, vy);
                double[][] pt = r.apply(s.testLogits());
                double[][] ps = r.apply(s.shiftedLogits());
                rows.add(new Row(e.getKey(), n, ece(pt, s.test().y()),
                        CalibrationEstimators.classwise(pt, s.test().y(), 15),
                        Calibration.negativeLogLikelihood(pt, s.test().y()), Calibration.accuracy(pt, s.test().y()),
                        ece(ps, s.shifted().y()), Calibration.negativeLogLikelihood(ps, s.shifted().y())));
            }
        }
        return rows;
    }

    /**
     * Debiased top-label ℓ2 ECE over 15 equal-mass bins.
     *
     * @param probs probabilities
     * @param labels labels
     * @return the estimate
     */
    public static double ece(double[][] probs, int[] labels) {
        double[][] t = CalibrationEstimators.topLabel(probs, labels);
        return CalibrationEstimators.debiasedL2(t[0], t[1], 15);
    }

    /**
     * One binary map.
     *
     * @param method the map
     * @param validation the validation rows
     * @param nll test NLL
     * @param brier test Brier score
     * @param ece test ECE (debiased, top label)
     * @param width mean Venn–Abers interval width (NaN for other maps)
     * @param shiftedNll shifted NLL
     * @param shiftedWidth shifted interval width
     */
    public record BinaryRow(String method, int validation, double nll, double brier, double ece, double width,
            double shiftedNll, double shiftedWidth) {
    }

    /**
     * Binary Venn–Abers on the tabular task. Cross Venn–Abers retrains three models, each on the
     * training rows plus two thirds of the validation rows, and calibrates each on the remaining
     * third.
     *
     * @param validationSizes the validation sizes
     * @param seed the seed
     * @return rows for temperature, Platt, isotonic, IVAP and CVAP per size
     */
    public static List<BinaryRow> vennAbers(int[] validationSizes, long seed) {
        UncertaintyFixture.Setup s = UncertaintyFixture.get(UncertaintyFixture.Task.TABULAR, seed);
        int[] ty = s.test().y();
        int[] sy = s.shifted().y();
        List<BinaryRow> rows = new ArrayList<>();
        for (int n : validationSizes) {
            double[][] vz = Arrays.copyOf(s.validationLogits(), n);
            int[] vy = Arrays.copyOf(s.validation().y(), n);
            for (Recalibrators.Method m : List.of(Recalibrators.Method.TEMPERATURE, Recalibrators.Method.PLATT,
                    Recalibrators.Method.ISOTONIC)) {
                Recalibrator r = Recalibrators.fit(m, vz, vy);
                double[][] pt = r.apply(s.testLogits());
                double[][] ps = r.apply(s.shiftedLogits());
                rows.add(new BinaryRow(m.name().toLowerCase(java.util.Locale.ROOT), n,
                        Calibration.negativeLogLikelihood(pt, ty), Calibration.brierScore(pt, ty), ece(pt, ty),
                        Double.NaN, Calibration.negativeLogLikelihood(ps, sy), Double.NaN));
            }
            VennAbers va = VennAbers.fit(positive(AffineRows.softmax(vz)), vy);
            rows.add(binaryRow("ivap", n, va, positive(AffineRows.softmax(s.testLogits())),
                    positive(AffineRows.softmax(s.shiftedLogits())), ty, sy));
            // cross Venn-Abers: three fold models
            int folds = 3;
            VennAbers[] fv = new VennAbers[folds];
            double[][] tScores = new double[folds][];
            double[][] sScores = new double[folds][];
            for (int f = 0; f < folds; f++) {
                List<Integer> trainRows = new ArrayList<>();
                List<Integer> calRows = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    (i % folds == f ? calRows : trainRows).add(i);
                }
                ShiftSuites.Split val = s.validation().range(0, n);
                ShiftSuites.Split tr = s.train().concat(val.rows(trainRows.stream().mapToInt(Integer::intValue)
                        .toArray()));
                ShiftSuites.Split cal = val.rows(calRows.stream().mapToInt(Integer::intValue).toArray());
                FeatureClassifier m = ShiftHarness.mlp(tr.x().shape().dim(1), 2, seed + 101 + f);
                ShiftHarness.train(m, tr, 400, 3e-3f, seed + f);
                fv[f] = VennAbers.fit(positive(AffineRows.softmax(UncertaintyFixture.logits(m, cal.x()))), cal.y());
                tScores[f] = positive(AffineRows.softmax(UncertaintyFixture.logits(m, s.test().x())));
                sScores[f] = positive(AffineRows.softmax(UncertaintyFixture.logits(m, s.shifted().x())));
            }
            double[] pt = VennAbers.cross(fv, tScores);
            double[] ps = VennAbers.cross(fv, sScores);
            double[][] t2 = binaryProbs(pt);
            double[][] s2 = binaryProbs(ps);
            rows.add(new BinaryRow("cvap", n, Calibration.negativeLogLikelihood(t2, ty),
                    Calibration.brierScore(t2, ty), ece(t2, ty), meanWidth(fv, tScores),
                    Calibration.negativeLogLikelihood(s2, sy), meanWidth(fv, sScores)));
        }
        return rows;
    }

    private static double meanWidth(VennAbers[] folds, double[][] scores) {
        double w = 0;
        for (int f = 0; f < folds.length; f++) {
            for (double v : scores[f]) {
                double[] p = folds[f].interval(v);
                w += p[1] - p[0];
            }
        }
        return w / (folds.length * scores[0].length);
    }

    private static BinaryRow binaryRow(String name, int n, VennAbers va, double[] test, double[] shifted, int[] ty,
            int[] sy) {
        double[] pt = new double[test.length];
        double wt = 0;
        for (int i = 0; i < test.length; i++) {
            double[] iv = va.interval(test[i]);
            pt[i] = VennAbers.merge(iv[0], iv[1]);
            wt += iv[1] - iv[0];
        }
        double[] ps = new double[shifted.length];
        double ws = 0;
        for (int i = 0; i < shifted.length; i++) {
            double[] iv = va.interval(shifted[i]);
            ps[i] = VennAbers.merge(iv[0], iv[1]);
            ws += iv[1] - iv[0];
        }
        double[][] t2 = binaryProbs(pt);
        double[][] s2 = binaryProbs(ps);
        return new BinaryRow(name, n, Calibration.negativeLogLikelihood(t2, ty), Calibration.brierScore(t2, ty),
                ece(t2, ty), wt / test.length, Calibration.negativeLogLikelihood(s2, sy), ws / shifted.length);
    }

    private static double[] positive(double[][] p) {
        double[] out = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            out[i] = p[i][1];
        }
        return out;
    }

    private static double[][] binaryProbs(double[] p1) {
        double[][] out = new double[p1.length][];
        for (int i = 0; i < p1.length; i++) {
            out[i] = new double[] {1 - p1[i], p1[i]};
        }
        return out;
    }

    /** Softmax rows. */
    static final class AffineRows {

        private AffineRows() {
        }

        static double[][] softmax(double[][] z) {
            return Calibration.withTemperature(z, 1);
        }
    }

    /**
     * Prints the full tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (UncertaintyFixture.Task t : UncertaintyFixture.Task.values()) {
            for (Row r : recalibration(t, new int[] {100, 300, 1000, 2000}, null, 1)) {
                System.out.printf("%s %-14s n=%4d ece %.4f cw %.4f nll %.3f acc %.3f | shift ece %.4f nll %.3f%n", t,
                        r.method(), r.validation(), r.ece(), r.classwiseEce(), r.nll(), r.accuracy(), r.shiftedEce(),
                        r.shiftedNll());
            }
        }
        for (BinaryRow r : vennAbers(new int[] {100, 300, 1000, 2000}, 1)) {
            System.out.println(r);
        }
    }
}
