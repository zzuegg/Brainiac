package io.github.zzuegg.brainiac.experimental.security;

import java.util.Arrays;
import java.util.Objects;

/**
 * Byzantine-robust aggregation rules, drop-in replacements for the mean in a federated
 * server step ({@link Aggregator}). With {@code n} clients of which up to {@code f} are
 * malicious:
 *
 * <ul>
 *   <li>{@link Rule#KRUM}: the update with the smallest sum of squared distances to its
 *       {@code n - f - 2} nearest neighbours (Blanchard et al., NeurIPS 2017,
 *       <a href="https://arxiv.org/abs/1703.02757">arXiv:1703.02757</a>);
 *       {@link Rule#MULTI_KRUM} averages the {@code n - f} best-scored updates.</li>
 *   <li>{@link Rule#MEDIAN} and {@link Rule#TRIMMED_MEAN}: per coordinate, the median, or the
 *       mean after dropping the {@code f} largest and {@code f} smallest values (Yin et al.,
 *       ICML 2018, <a href="https://arxiv.org/abs/1803.01498">arXiv:1803.01498</a>).</li>
 *   <li>{@link Rule#GEOMETRIC_MEDIAN}: RFA's smoothed Weiszfeld iterations for the point
 *       minimising the sum of Euclidean distances (Pillutla et al., IEEE TSP 2022,
 *       <a href="https://arxiv.org/abs/1912.13445">arXiv:1912.13445</a>).</li>
 * </ul>
 */
public final class RobustAggregators {

    /** An aggregation rule. */
    public enum Rule {
        /** the plain mean (FedAvg) */
        MEAN,
        /** Krum */
        KRUM,
        /** Multi-Krum, averaging the {@code n - f} best */
        MULTI_KRUM,
        /** coordinate-wise median */
        MEDIAN,
        /** coordinate-wise trimmed mean */
        TRIMMED_MEAN,
        /** geometric median by smoothed Weiszfeld */
        GEOMETRIC_MEDIAN
    }

    private RobustAggregators() {
    }

    /**
     * A rule as an {@link Aggregator}.
     *
     * @param rule the rule
     * @param f the number of malicious clients the rule should tolerate
     * @return the aggregator
     */
    public static Aggregator of(Rule rule, int f) {
        Objects.requireNonNull(rule);
        return switch (rule) {
            case MEAN -> RobustAggregators::mean;
            case KRUM -> u -> krum(u, f, 1);
            case MULTI_KRUM -> u -> krum(u, f, Math.max(1, u.length - f));
            case MEDIAN -> RobustAggregators::median;
            case TRIMMED_MEAN -> u -> trimmedMean(u, f);
            case GEOMETRIC_MEDIAN -> u -> geometricMedian(u, 8, 1e-6);
        };
    }

    /**
     * The mean.
     *
     * @param u the updates
     * @return their mean
     */
    public static float[] mean(float[][] u) {
        float[] out = new float[u[0].length];
        for (float[] v : u) {
            for (int j = 0; j < out.length; j++) {
                out[j] += v[j] / u.length;
            }
        }
        return out;
    }

    /**
     * Squared Euclidean distance.
     *
     * @param a one vector
     * @param b another
     * @return {@code ||a - b||²}
     */
    public static double distance2(float[] a, float[] b) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            double d = a[j] - b[j];
            s += d * d;
        }
        return s;
    }

    /**
     * Krum's scores: the summed squared distance of each update to its {@code n - f - 2}
     * nearest other updates.
     *
     * @param u the updates
     * @param f tolerated malicious clients
     * @return one score per update, smaller is more central
     */
    public static double[] krumScores(float[][] u, int f) {
        int n = u.length;
        double[][] d = pairwise(u);
        int k = Math.max(1, n - f - 2);
        double[] scores = new double[n];
        for (int i = 0; i < n; i++) {
            double[] row = new double[n - 1];
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    row[c++] = d[i][j];
                }
            }
            Arrays.sort(row);
            for (int j = 0; j < Math.min(k, row.length); j++) {
                scores[i] += row[j];
            }
        }
        return scores;
    }

    /**
     * Multi-Krum: the mean of the {@code m} updates with the best Krum scores ({@code m = 1}
     * is Krum).
     *
     * @param u the updates
     * @param f tolerated malicious clients
     * @param m updates averaged
     * @return the aggregate
     */
    public static float[] krum(float[][] u, int f, int m) {
        double[] scores = krumScores(u, f);
        Integer[] order = new Integer[u.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        float[][] chosen = new float[Math.min(m, u.length)][];
        for (int i = 0; i < chosen.length; i++) {
            chosen[i] = u[order[i]];
        }
        return mean(chosen);
    }

    /**
     * The coordinate-wise median.
     *
     * @param u the updates
     * @return the median
     */
    public static float[] median(float[][] u) {
        return trimmedMean(u, (u.length - 1) / 2);
    }

    /**
     * The coordinate-wise trimmed mean: per coordinate, drop the {@code f} largest and
     * {@code f} smallest values and average the rest ({@code f = (n-1)/2} gives the median).
     *
     * @param u the updates
     * @param f values trimmed at each end
     * @return the aggregate
     */
    public static float[] trimmedMean(float[][] u, int f) {
        int n = u.length;
        int trim = Math.min(f, (n - 1) / 2);
        float[] out = new float[u[0].length];
        float[] col = new float[n];
        for (int j = 0; j < out.length; j++) {
            for (int i = 0; i < n; i++) {
                col[i] = u[i][j];
            }
            Arrays.sort(col);
            double s = 0;
            for (int i = trim; i < n - trim; i++) {
                s += col[i];
            }
            out[j] = (float) (s / (n - 2 * trim));
        }
        return out;
    }

    /**
     * The geometric median by smoothed Weiszfeld iterations, started at the coordinate-wise
     * median.
     *
     * @param u the updates
     * @param iterations Weiszfeld iterations, RFA uses 3 to 10
     * @param nu the smoothing, which bounds each weight by {@code 1/ν}
     * @return the approximate geometric median
     */
    public static float[] geometricMedian(float[][] u, int iterations, double nu) {
        float[] z = median(u);
        for (int t = 0; t < iterations; t++) {
            double[] acc = new double[z.length];
            double total = 0;
            for (float[] v : u) {
                double w = 1 / Math.max(nu, Math.sqrt(distance2(v, z)));
                total += w;
                for (int j = 0; j < z.length; j++) {
                    acc[j] += w * v[j];
                }
            }
            for (int j = 0; j < z.length; j++) {
                z[j] = (float) (acc[j] / total);
            }
        }
        return z;
    }

    static double[][] pairwise(float[][] u) {
        int n = u.length;
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                d[i][j] = distance2(u[i], u[j]);
                d[j][i] = d[i][j];
            }
        }
        return d;
    }
}
