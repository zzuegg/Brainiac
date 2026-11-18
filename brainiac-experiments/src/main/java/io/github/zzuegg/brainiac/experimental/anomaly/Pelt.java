package io.github.zzuegg.brainiac.experimental.anomaly;

import java.util.ArrayList;
import java.util.List;

/**
 * PELT, pruned exact linear time segmentation (Killick, Fearnhead and Eckley, JASA 2012,
 * <a href="https://arxiv.org/abs/1101.1438">arXiv:1101.1438</a>): the optimal partitioning
 * recursion {@code F(t) = min_s F(s) + C(y_s+1..t) + beta} over the last change {@code s},
 * with candidate positions pruned once {@code F(s) + C(y_s+1..t) > F(t)}: such an {@code s}
 * can never be optimal later (for costs where adding a change never increases the cost), so
 * the search is exact yet near-linear when changes keep arriving. Segment costs come from
 * cumulative sums in {@code O(1)}: the Gaussian negative log-likelihood with a changing mean
 * (known variance), or with changing mean and variance.
 */
public final class Pelt {

    /** The segment cost. */
    public enum Cost {
        /** The squared error around the segment mean (mean changes, fixed variance). */
        MEAN,
        /** {@code n log(variance)} of the segment (mean and variance changes). */
        MEAN_VARIANCE
    }

    private final Cost cost;
    private final double penalty;
    private final int minSegment;

    /**
     * Creates the segmenter.
     *
     * @param cost the segment cost
     * @param penalty the cost of one change, for example {@code 2 log n} times the parameters per segment
     * @param minSegment the shortest segment
     */
    public Pelt(Cost cost, double penalty, int minSegment) {
        this.cost = cost;
        this.penalty = penalty;
        this.minSegment = minSegment;
    }

    /**
     * Segments a series.
     *
     * @param y the series
     * @return the change points (first step of each new segment), in order
     */
    public int[] segment(double[] y) {
        int n = y.length;
        double[] s1 = new double[n + 1];
        double[] s2 = new double[n + 1];
        for (int i = 0; i < n; i++) {
            s1[i + 1] = s1[i] + y[i];
            s2[i + 1] = s2[i] + y[i] * y[i];
        }
        double var = Math.max(1e-12, s2[n] / n - (s1[n] / n) * (s1[n] / n));
        double[] f = new double[n + 1];
        int[] last = new int[n + 1];
        f[0] = -penalty;
        List<Integer> candidates = new ArrayList<>();
        candidates.add(0);
        for (int t = minSegment; t <= n; t++) {
            double best = Double.POSITIVE_INFINITY;
            int arg = 0;
            double[] values = new double[candidates.size()];
            for (int c = 0; c < candidates.size(); c++) {
                int s = candidates.get(c);
                if (t - s < minSegment) {
                    values[c] = Double.NaN;
                    continue;
                }
                double v = f[s] + segmentCost(s1, s2, s, t, var) + penalty;
                values[c] = v;
                if (v < best) {
                    best = v;
                    arg = s;
                }
            }
            f[t] = best;
            last[t] = arg;
            List<Integer> kept = new ArrayList<>();
            for (int c = 0; c < candidates.size(); c++) {
                if (Double.isNaN(values[c]) || values[c] - penalty <= best) {
                    kept.add(candidates.get(c));
                }
            }
            if (t + minSegment <= n) {
                kept.add(t);
            }
            candidates = kept;
        }
        List<Integer> cps = new ArrayList<>();
        int t = n;
        while (t > 0) {
            int s = last[t];
            if (s > 0) {
                cps.add(0, s);
            }
            t = s;
        }
        return cps.stream().mapToInt(Integer::intValue).toArray();
    }

    private double segmentCost(double[] s1, double[] s2, int from, int to, double globalVar) {
        int m = to - from;
        double sum = s1[to] - s1[from];
        double sq = s2[to] - s2[from];
        double sse = Math.max(0, sq - sum * sum / m);
        return switch (cost) {
            case MEAN -> sse / globalVar;
            case MEAN_VARIANCE -> m * Math.log(Math.max(1e-6 * globalVar, sse / m));
        };
    }
}
