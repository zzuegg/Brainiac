package io.github.zzuegg.brainiac.experiments.evaluation;

import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.ToDoubleBiFunction;

/**
 * Bias and RMSE of calibration-error estimators where the true error is known (9.3.10), and
 * whether method rankings flip with the estimator or the bin count.
 * <ul>
 *   <li>{@link #synthetic}: a predictor whose confidences {@code c = 1 - u²/2} ({@code u}
 *       uniform) are right with probability {@code c^γ}; γ = 1 is perfectly calibrated, γ &gt; 1
 *       overconfident. The true ℓ1 and ℓ2 errors are integrals over {@code u}. Each estimator
 *       runs on {@code replications} fresh samples of size n: equal-width (the
 *       {@link Calibration#expectedCalibrationError} baseline's 15 bins), equal-mass, the
 *       monotone sweep and smooth ECE against the ℓ1 truth; the plug-in and debiased ℓ2
 *       estimators against the ℓ2 truth;</li>
 *   <li>{@link #rankings}: methods of {@link UncertaintyBenchmark} ranked by each estimator,
 *       by equal-width ECE at several bin counts, and by Brier and NLL.</li>
 * </ul>
 */
public final class CalibrationEstimators {

    private CalibrationEstimators() {
    }

    /**
     * One estimator's error at one sample size.
     *
     * @param estimator the estimator
     * @param n the sample size
     * @param truth the true calibration error it estimates
     * @param mean the mean estimate over replications
     * @param bias mean estimate minus truth
     * @param rmse root mean squared error against the truth
     */
    public record Error(String estimator, int n, double truth, double mean, double bias, double rmse) {
        /** @return the mean estimate over the truth (overstatement factor) */
        public double ratio() {
            return mean / truth;
        }

        @Override
        public String toString() {
            return String.format("%s n=%d: mean %.4f (truth %.4f), bias %+.4f, RMSE %.4f", estimator, n, mean, truth,
                    bias, rmse);
        }
    }

    private static double confidence(double u) {
        return 1 - 0.5 * u * u;
    }

    /**
     * The true ℓ1 and ℓ2 calibration error of the synthetic predictor.
     *
     * @param gamma the miscalibration exponent
     * @return {@code {l1, l2}}
     */
    public static double[] truth(double gamma) {
        int grid = 200_000;
        double l1 = 0;
        double l2 = 0;
        for (int i = 0; i < grid; i++) {
            double c = confidence((i + 0.5) / grid);
            double d = Math.pow(c, gamma) - c;
            l1 += Math.abs(d) / grid;
            l2 += d * d / grid;
        }
        return new double[] {l1, Math.sqrt(l2)};
    }

    /**
     * Draws a sample of the synthetic predictor.
     *
     * @param gamma the miscalibration exponent
     * @param n the sample size
     * @param r the random source
     * @return {@code {confidence, correct}}
     */
    public static double[][] sample(double gamma, int n, SplittableRandom r) {
        double[] c = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            c[i] = confidence(r.nextDouble());
            y[i] = r.nextDouble() < Math.pow(c[i], gamma) ? 1 : 0;
        }
        return new double[][] {c, y};
    }

    private static Map<String, ToDoubleBiFunction<double[], double[]>> l1Estimators() {
        Map<String, ToDoubleBiFunction<double[], double[]>> m = new LinkedHashMap<>();
        m.put("equal-width 15", (c, y) -> io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators
                .equalWidth(c, y, 15));
        m.put("equal-mass 15", (c, y) -> io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators
                .equalMass(c, y, 15));
        m.put("monotone sweep", io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators::monotoneSweep);
        m.put("smooth ECE", io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators::smooth);
        return m;
    }

    private static Map<String, ToDoubleBiFunction<double[], double[]>> l2Estimators() {
        Map<String, ToDoubleBiFunction<double[], double[]>> m = new LinkedHashMap<>();
        m.put("plug-in L2 15", (c, y) -> io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators
                .pluginL2(c, y, 15));
        m.put("debiased L2 15", (c, y) -> io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators
                .debiasedL2(c, y, 15));
        return m;
    }

    /**
     * Bias and RMSE of every estimator at every sample size.
     *
     * @param gamma the miscalibration exponent
     * @param sizes the sample sizes
     * @param replications fresh samples per size
     * @param seed the seed
     * @return one error per estimator and size
     */
    public static List<Error> synthetic(double gamma, int[] sizes, int replications, long seed) {
        double[] truth = truth(gamma);
        List<Error> out = new ArrayList<>();
        Map<String, ToDoubleBiFunction<double[], double[]>> l1 = l1Estimators();
        Map<String, ToDoubleBiFunction<double[], double[]>> l2 = l2Estimators();
        for (int n : sizes) {
            Map<String, double[]> est = new LinkedHashMap<>();
            SplittableRandom r = new SplittableRandom(seed + n);
            for (int k = 0; k < replications; k++) {
                double[][] s = sample(gamma, n, r);
                for (Map.Entry<String, ToDoubleBiFunction<double[], double[]>> e : l1.entrySet()) {
                    est.computeIfAbsent(e.getKey(), x -> new double[replications])[k] = e.getValue()
                            .applyAsDouble(s[0], s[1]);
                }
                for (Map.Entry<String, ToDoubleBiFunction<double[], double[]>> e : l2.entrySet()) {
                    est.computeIfAbsent(e.getKey(), x -> new double[replications])[k] = e.getValue()
                            .applyAsDouble(s[0], s[1]);
                }
            }
            for (Map.Entry<String, double[]> e : est.entrySet()) {
                double t = l1.containsKey(e.getKey()) ? truth[0] : truth[1];
                double mean = Arrays.stream(e.getValue()).average().orElse(0);
                double mse = Arrays.stream(e.getValue()).map(v -> (v - t) * (v - t)).average().orElse(0);
                out.add(new Error(e.getKey(), n, t, mean, mean - t, Math.sqrt(mse)));
            }
        }
        return out;
    }

    /**
     * Rankings of methods under each score.
     *
     * @param scores score name to the method ranking it gives, best (lowest) first
     * @param distinctEqualWidth distinct rankings by equal-width ECE over bin counts 5..50
     * @param flipsVsBaseline scores whose ranking differs from equal-width 15 bins
     */
    public record Rankings(Map<String, List<String>> scores, int distinctEqualWidth, List<String> flipsVsBaseline) {
    }

    /**
     * Ranks {@link UncertaintyBenchmark} methods by every estimator.
     *
     * @param methods the methods to train
     * @param seed the seed
     * @return the rankings
     */
    public static Rankings rankings(List<UncertaintyBenchmark.Method> methods, long seed) {
        Map<String, double[][]> data = new LinkedHashMap<>();
        Map<String, UncertaintyBenchmark.Predictions> preds = new LinkedHashMap<>();
        for (UncertaintyBenchmark.Method m : methods) {
            UncertaintyBenchmark.Predictions p = UncertaintyBenchmark.predictions(m, seed);
            preds.put(m.name(), p);
            data.put(m.name(), io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators.topLabel(p.probs(),
                    p.labels()));
        }
        Map<String, ToDoubleBiFunction<double[], double[]>> scores = new LinkedHashMap<>();
        for (int bins : new int[] {5, 10, 15, 20, 30, 50}) {
            scores.put("equal-width " + bins, (c, y) -> io.github.zzuegg.brainiac.experimental.eval
                    .CalibrationEstimators.equalWidth(c, y, bins));
        }
        scores.putAll(l1Estimators());
        scores.putAll(l2Estimators());
        scores.put("Brier", io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators::brier);
        Map<String, List<String>> ranked = new LinkedHashMap<>();
        for (Map.Entry<String, ToDoubleBiFunction<double[], double[]>> e : scores.entrySet()) {
            Map<String, Double> v = new LinkedHashMap<>();
            data.forEach((m, d) -> v.put(m, e.getValue().applyAsDouble(d[0], d[1])));
            ranked.put(e.getKey(), sorted(v));
        }
        Map<String, Double> nll = new LinkedHashMap<>();
        preds.forEach((m, p) -> nll.put(m, Calibration.negativeLogLikelihood(p.probs(), p.labels())));
        ranked.put("NLL", sorted(nll));
        Set<List<String>> distinct = new LinkedHashSet<>();
        for (int bins : new int[] {5, 10, 15, 20, 30, 50}) {
            distinct.add(ranked.get("equal-width " + bins));
        }
        List<String> flips = new ArrayList<>();
        List<String> base = ranked.get("equal-width 15");
        ranked.forEach((k, v) -> {
            if (!v.equals(base)) {
                flips.add(k);
            }
        });
        return new Rankings(ranked, distinct.size(), flips);
    }

    private static List<String> sorted(Map<String, Double> v) {
        List<String> k = new ArrayList<>(v.keySet());
        k.sort((a, b) -> Double.compare(v.get(a), v.get(b)));
        return k;
    }
}
