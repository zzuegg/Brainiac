package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.experimental.anomaly.BayesianOnlineChangePoint;
import io.github.zzuegg.brainiac.experimental.anomaly.Pelt;
import io.github.zzuegg.brainiac.experimental.data.AnomalySeries;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Change-point detection (8.2.15) on {@link AnomalySeries#changePoints}: piecewise segments
 * whose mean, noise level or seasonal amplitude changes. Methods:
 * <ul>
 *   <li>{@link Pelt} with the Gaussian mean-and-variance cost and a BIC-like penalty
 *       ({@code 3 log n}), offline and exact;</li>
 *   <li>{@link BayesianOnlineChangePoint} (Normal-Gamma model, hazard 1/100), online, reporting
 *       a change 8 steps after it once the short run lengths hold half the posterior mass;</li>
 *   <li>the thresholded innovations of a local-level Kalman filter
 *       ({@link StateSpaceFilter#kalman}), online: a change where the standardised one-step
 *       error exceeds 3.5 after a quiet stretch;</li>
 *   <li>Viterbi segmentation of a four-state Gaussian {@link HiddenMarkovModel} fitted by
 *       Baum-Welch: a change wherever the decoded state switches and stays for 10 steps;</li>
 *   <li>the single change point model of {@code probabilistic.ProbabilisticPrograms} (one
 *       change of the mean, exact posterior by enumeration), which can find one change at most.</li>
 * </ul>
 * Scores: F1 of detected against true change points within a 5-step tolerance, recall per
 * change kind, and the mean delay (report step minus true step for the online methods,
 * absolute location error for the offline ones).
 */
public final class ChangePoints {

    /**
     * One method's scores.
     *
     * @param f1 F1 within the tolerance
     * @param delay mean detection delay (steps) over matched change points
     * @param recallMean recall on mean changes
     * @param recallVariance recall on variance changes
     * @param recallSeason recall on seasonality changes
     */
    public record Score(double f1, double delay, double recallMean, double recallVariance, double recallSeason) {
        @Override
        public String toString() {
            return String.format("F1 %.2f delay %.1f recall mean/var/season %.2f/%.2f/%.2f", f1, delay, recallMean,
                    recallVariance, recallSeason);
        }
    }

    /** Detected change points and the steps at which they were reported. */
    private record Detection(int[] points, int[] reportedAt) {
    }

    private ChangePoints() {
    }

    /**
     * Runs every method on series from several seeds and averages the scores.
     *
     * @param length steps per series
     * @param seeds how many series
     * @param tolerance the matching tolerance in steps
     * @return method to scores
     */
    public static Map<String, Score> run(int length, int seeds, int tolerance) {
        Map<String, double[]> acc = new LinkedHashMap<>();
        for (int s = 0; s < seeds; s++) {
            AnomalySeries.Segmented series = AnomalySeries.changePoints(length, 20,
                    List.of(AnomalySeries.Change.MEAN, AnomalySeries.Change.VARIANCE, AnomalySeries.Change.SEASONALITY),
                    100 + s);
            double[] y = new double[length];
            for (int t = 0; t < length; t++) {
                y[t] = series.values()[t];
            }
            Map<String, Detection> found = new LinkedHashMap<>();
            int[] pelt = new Pelt(Pelt.Cost.MEAN_VARIANCE, 3 * Math.log(length), 8).segment(y);
            found.put("PELT", new Detection(pelt, pelt));
            BayesianOnlineChangePoint.Result b = new BayesianOnlineChangePoint(100, 0, 0.1, 1, 0.3).run(y, 8, 0.5);
            found.put("BOCPD", new Detection(b.changePoints(), b.detectedAt()));
            found.put("Kalman innovations", kalman(y));
            found.put("HMM Viterbi", hmm(y));
            int single = singleChange(y);
            found.put("single change model", new Detection(new int[] {single}, new int[] {single}));
            for (Map.Entry<String, Detection> e : found.entrySet()) {
                double[] sc = score(e.getValue(), series, tolerance);
                double[] a = acc.computeIfAbsent(e.getKey(), k -> new double[5]);
                for (int i = 0; i < 5; i++) {
                    a[i] += sc[i] / seeds;
                }
            }
        }
        Map<String, Score> out = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> e : acc.entrySet()) {
            double[] a = e.getValue();
            out.put(e.getKey(), new Score(a[0], a[1], a[2], a[3], a[4]));
        }
        return out;
    }

    /** {F1, mean delay, recall per kind}. */
    private static double[] score(Detection d, AnomalySeries.Segmented truth, int tolerance) {
        int[] cps = truth.changePoints();
        boolean[] used = new boolean[d.points().length];
        int matched = 0;
        double delay = 0;
        double[] hit = new double[3];
        double[] count = new double[3];
        for (int i = 0; i < cps.length; i++) {
            int kind = truth.changes().get(i).ordinal();
            count[kind]++;
            int best = -1;
            for (int j = 0; j < d.points().length; j++) {
                if (!used[j] && Math.abs(d.points()[j] - cps[i]) <= tolerance
                        && (best < 0 || Math.abs(d.points()[j] - cps[i]) < Math.abs(d.points()[best] - cps[i]))) {
                    best = j;
                }
            }
            if (best >= 0) {
                used[best] = true;
                matched++;
                hit[kind]++;
                delay += Math.abs(d.reportedAt()[best] - cps[i]);
            }
        }
        double precision = d.points().length == 0 ? 0 : matched / (double) d.points().length;
        double recall = matched / (double) cps.length;
        double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        return new double[] {f1, matched == 0 ? tolerance : delay / matched, hit[0] / Math.max(1, count[0]),
            hit[1] / Math.max(1, count[1]), hit[2] / Math.max(1, count[2])};
    }

    /** A local-level Kalman filter; a change where the standardised innovation exceeds 3.5 after 15 quiet steps. */
    private static Detection kalman(double[] y) {
        double dv = 0;
        for (int t = 1; t < y.length; t++) {
            dv += (y[t] - y[t - 1]) * (y[t] - y[t - 1]) / (y.length - 1);
        }
        double r = dv / 2;
        double q = 0.01 * r;
        StateSpaceFilter.LinearModel m = new StateSpaceFilter.LinearModel(new double[][] {{1}}, new double[][] {{q}},
                new double[][] {{1}}, new double[][] {{r}}, new double[] {y[0]}, new double[][] {{r}});
        double[][] ys = new double[y.length][1];
        for (int t = 0; t < y.length; t++) {
            ys[t][0] = y[t];
        }
        StateSpaceFilter.Estimate e = StateSpaceFilter.kalman(m, ys, false);
        List<Integer> cps = new ArrayList<>();
        int quiet = 0;
        for (int t = 1; t < y.length; t++) {
            double pred = e.means()[t - 1][0];
            double s = e.covariances()[t - 1][0][0] + q + r;
            double z = Math.abs(y[t] - pred) / Math.sqrt(s);
            if (z > 3.5 && quiet >= 15) {
                cps.add(t);
                quiet = 0;
            } else {
                quiet++;
            }
        }
        int[] p = cps.stream().mapToInt(Integer::intValue).toArray();
        return new Detection(p, p);
    }

    /** Viterbi segmentation of a four-state Gaussian HMM fitted by Baum-Welch. */
    private static Detection hmm(double[] y) {
        double[] sorted = y.clone();
        java.util.Arrays.sort(sorted);
        int k = 4;
        double[] means = new double[k];
        double[] vars = new double[k];
        double[] start = new double[k];
        double[][] trans = new double[k][k];
        for (int i = 0; i < k; i++) {
            means[i] = sorted[(int) ((i + 0.5) / k * sorted.length)];
            vars[i] = 0.5;
            start[i] = 1.0 / k;
            for (int j = 0; j < k; j++) {
                trans[i][j] = i == j ? 0.97 : 0.03 / (k - 1);
            }
        }
        HiddenMarkovModel model = HiddenMarkovModel.gaussian(start, trans, means, vars).baumWelch(List.of(y), 15);
        int[] states = model.viterbi(y);
        List<Integer> cps = new ArrayList<>();
        int current = states[0];
        for (int t = 1; t < states.length; t++) {
            if (states[t] != current) {
                boolean stays = true;
                for (int u = t; u < Math.min(states.length, t + 10); u++) {
                    stays &= states[u] == states[t];
                }
                if (stays) {
                    if (cps.isEmpty() || t - cps.get(cps.size() - 1) > 10) {
                        cps.add(t);
                    }
                    current = states[t];
                }
            }
        }
        int[] p = cps.stream().mapToInt(Integer::intValue).toArray();
        return new Detection(p, p);
    }

    /**
     * The MAP position of a single change of the mean, by exact enumeration (flat prior on the
     * position, segment means profiled out), the Gaussian analogue of the coin-bias model.
     */
    private static int singleChange(double[] y) {
        int n = y.length;
        double[] s1 = new double[n + 1];
        double[] s2 = new double[n + 1];
        for (int i = 0; i < n; i++) {
            s1[i + 1] = s1[i] + y[i];
            s2[i + 1] = s2[i] + y[i] * y[i];
        }
        int best = 1;
        double bestSse = Double.POSITIVE_INFINITY;
        for (int tau = 5; tau < n - 5; tau++) {
            double a = s2[tau] - s1[tau] * s1[tau] / tau;
            double b = (s2[n] - s2[tau]) - (s1[n] - s1[tau]) * (s1[n] - s1[tau]) / (n - tau);
            if (a + b < bestSse) {
                bestSse = a + b;
                best = tau;
            }
        }
        return best;
    }
}
