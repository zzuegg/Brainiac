package io.github.zzuegg.brainiac.experimental.tsc;

import java.util.SplittableRandom;

/**
 * Elastic distances between univariate series, the measures behind the Elastic Ensemble (Lines and
 * Bagnall, DMKD 2015) and Proximity Forest (Lucas et al., DMKD 2019,
 * <a href="https://arxiv.org/abs/1808.10594">arXiv:1808.10594</a>):
 * <ul>
 *   <li>{@link #dtw}: dynamic time warping with squared point costs and a Sakoe-Chiba window;</li>
 *   <li>{@link #wdtw}: weighted DTW (Jeong et al. 2011), a logistic penalty on the phase
 *       difference {@code |i - j|} instead of a hard window;</li>
 *   <li>{@link #derivative}: Keogh and Pazzani's derivative estimate, for DDTW and WDDTW;</li>
 *   <li>{@link #lcss}: the longest common subsequence with a matching threshold {@code epsilon};</li>
 *   <li>{@link #erp}: edit distance with real penalty (Chen and Ng, VLDB 2004), a metric whose gaps
 *       cost the distance to a constant {@code g};</li>
 *   <li>{@link #msm}: move-split-merge (Stefan, Athitsos and Das, TKDE 2013), a metric with a cost
 *       {@code c} per split or merge;</li>
 *   <li>{@link #twe}: time warp edit distance (Marteau, TPAMI 2009) with stiffness {@code nu} and
 *       an edit penalty {@code lambda};</li>
 *   <li>{@link #lbKeogh}: the LB_Keogh lower bound of windowed DTW, for pruning 1-NN search.</li>
 * </ul>
 * {@link Measure} wraps them with the parameter ranges Proximity Forest draws from.
 */
public final class ElasticDistances {

    private ElasticDistances() {
    }

    /**
     * Squared Euclidean distance (lock-step).
     *
     * @param a one series
     * @param b another of the same length
     * @return {@code sum (a_t - b_t)^2}
     */
    public static double euclidean(double[] a, double[] b) {
        double s = 0;
        for (int t = 0; t < Math.min(a.length, b.length); t++) {
            double d = a[t] - b[t];
            s += d * d;
        }
        return s;
    }

    /**
     * DTW with squared point costs.
     *
     * @param a one series
     * @param b another
     * @param window the Sakoe-Chiba half-width in steps; negative for none
     * @return the cheapest alignment cost
     */
    public static double dtw(double[] a, double[] b, int window) {
        return weightedDtw(a, b, window, null);
    }

    /**
     * Weighted DTW: the point cost {@code (a_i - b_j)^2} is multiplied by
     * {@code w_max / (1 + exp(-g (|i - j| - m / 2)))} with {@code w_max = 1}.
     *
     * @param a one series
     * @param b another
     * @param g the penalty steepness, typically in {@code [0, 1]}
     * @return the weighted alignment cost
     */
    public static double wdtw(double[] a, double[] b, double g) {
        int m = Math.max(a.length, b.length);
        double[] weight = new double[m];
        for (int k = 0; k < m; k++) {
            weight[k] = 1 / (1 + Math.exp(-g * (k - m / 2.0)));
        }
        return weightedDtw(a, b, -1, weight);
    }

    private static double weightedDtw(double[] a, double[] b, int window, double[] weight) {
        int n = a.length;
        int m = b.length;
        int w = window < 0 ? Math.max(n, m) : Math.max(window, Math.abs(n - m));
        double[] prev = new double[m + 1];
        double[] cur = new double[m + 1];
        java.util.Arrays.fill(prev, Double.POSITIVE_INFINITY);
        prev[0] = 0;
        for (int i = 1; i <= n; i++) {
            java.util.Arrays.fill(cur, Double.POSITIVE_INFINITY);
            int lo = Math.max(1, i - w);
            int hi = Math.min(m, i + w);
            for (int j = lo; j <= hi; j++) {
                double d = a[i - 1] - b[j - 1];
                double cost = d * d * (weight == null ? 1 : weight[Math.abs(i - j)]);
                cur[j] = cost + Math.min(prev[j - 1], Math.min(prev[j], cur[j - 1]));
            }
            double[] t = prev;
            prev = cur;
            cur = t;
        }
        return prev[m];
    }

    /**
     * Keogh and Pazzani's derivative: {@code ((x_t - x_{t-1}) + (x_{t+1} - x_{t-1}) / 2) / 2}, with
     * the end values copied from their neighbours.
     *
     * @param x the series, at least 3 values
     * @return the derivative series of the same length
     */
    public static double[] derivative(double[] x) {
        int n = x.length;
        double[] d = new double[n];
        for (int t = 1; t < n - 1; t++) {
            d[t] = ((x[t] - x[t - 1]) + (x[t + 1] - x[t - 1]) / 2) / 2;
        }
        d[0] = d[1];
        d[n - 1] = d[n - 2];
        return d;
    }

    /**
     * The LCSS distance {@code 1 - LCSS / min(n, m)}: two values match when within
     * {@code epsilon} and within {@code window} steps of each other.
     *
     * @param a one series
     * @param b another
     * @param epsilon the matching threshold
     * @param window the matching window; negative for none
     * @return the distance in {@code [0, 1]}
     */
    public static double lcss(double[] a, double[] b, double epsilon, int window) {
        int n = a.length;
        int m = b.length;
        int w = window < 0 ? Math.max(n, m) : window;
        int[][] l = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (Math.abs(i - j) <= w && Math.abs(a[i - 1] - b[j - 1]) <= epsilon) {
                    l[i][j] = l[i - 1][j - 1] + 1;
                } else {
                    l[i][j] = Math.max(l[i - 1][j], l[i][j - 1]);
                }
            }
        }
        return 1 - l[n][m] / (double) Math.min(n, m);
    }

    /**
     * Edit distance with real penalty: a gap costs {@code |x - g|}, a match {@code |a_i - b_j|}
     * (absolute costs, so ERP is a metric).
     *
     * @param a one series
     * @param b another
     * @param g the gap value
     * @param window the band half-width; negative for none
     * @return the distance
     */
    public static double erp(double[] a, double[] b, double g, int window) {
        int n = a.length;
        int m = b.length;
        int w = window < 0 ? Math.max(n, m) : Math.max(window, Math.abs(n - m));
        double[][] d = new double[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            d[i][0] = d[i - 1][0] + Math.abs(a[i - 1] - g);
        }
        for (int j = 1; j <= m; j++) {
            d[0][j] = d[0][j - 1] + Math.abs(b[j - 1] - g);
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (Math.abs(i - j) > w) {
                    d[i][j] = Double.POSITIVE_INFINITY;
                    continue;
                }
                double match = d[i - 1][j - 1] + Math.abs(a[i - 1] - b[j - 1]);
                double gapA = d[i - 1][j] + Math.abs(a[i - 1] - g);
                double gapB = d[i][j - 1] + Math.abs(b[j - 1] - g);
                d[i][j] = Math.min(match, Math.min(gapA, gapB));
            }
        }
        return d[n][m];
    }

    /**
     * Move-split-merge.
     *
     * @param a one series
     * @param b another
     * @param c the cost of a split or merge
     * @return the distance
     */
    public static double msm(double[] a, double[] b, double c) {
        int n = a.length;
        int m = b.length;
        double[][] d = new double[n][m];
        d[0][0] = Math.abs(a[0] - b[0]);
        for (int i = 1; i < n; i++) {
            d[i][0] = d[i - 1][0] + msmCost(a[i], a[i - 1], b[0], c);
        }
        for (int j = 1; j < m; j++) {
            d[0][j] = d[0][j - 1] + msmCost(b[j], a[0], b[j - 1], c);
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                double move = d[i - 1][j - 1] + Math.abs(a[i] - b[j]);
                double split = d[i - 1][j] + msmCost(a[i], a[i - 1], b[j], c);
                double merge = d[i][j - 1] + msmCost(b[j], a[i], b[j - 1], c);
                d[i][j] = Math.min(move, Math.min(split, merge));
            }
        }
        return d[n - 1][m - 1];
    }

    /** The cost of a split or merge of {@code x} between {@code y} and {@code z}. */
    private static double msmCost(double x, double y, double z, double c) {
        if ((y <= x && x <= z) || (y >= x && x >= z)) {
            return c;
        }
        return c + Math.min(Math.abs(x - y), Math.abs(x - z));
    }

    /**
     * Time warp edit distance with time stamps {@code 1, 2, ...}.
     *
     * @param a one series
     * @param b another
     * @param nu the stiffness (penalty per step of time difference)
     * @param lambda the cost of a deletion
     * @return the distance
     */
    public static double twe(double[] a, double[] b, double nu, double lambda) {
        int n = a.length;
        int m = b.length;
        double[] x = new double[n + 1];
        double[] y = new double[m + 1];
        System.arraycopy(a, 0, x, 1, n);
        System.arraycopy(b, 0, y, 1, m);
        double[][] d = new double[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            d[i][0] = Double.POSITIVE_INFINITY;
        }
        for (int j = 1; j <= m; j++) {
            d[0][j] = Double.POSITIVE_INFINITY;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                double delA = d[i - 1][j] + Math.abs(x[i] - x[i - 1]) + nu + lambda;
                double delB = d[i][j - 1] + Math.abs(y[j] - y[j - 1]) + nu + lambda;
                double match = d[i - 1][j - 1] + Math.abs(x[i] - y[j]) + Math.abs(x[i - 1] - y[j - 1])
                        + 2 * nu * Math.abs(i - j);
                d[i][j] = Math.min(match, Math.min(delA, delB));
            }
        }
        return d[n][m];
    }

    /**
     * LB_Keogh: the squared distance from the query to the candidate's upper and lower envelope
     * over the window; never more than the windowed DTW distance.
     *
     * @param query the query
     * @param candidate the candidate, the same length
     * @param window the DTW window half-width
     * @return the lower bound
     */
    public static double lbKeogh(double[] query, double[] candidate, int window) {
        double s = 0;
        int n = query.length;
        for (int i = 0; i < n; i++) {
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (int j = Math.max(0, i - window); j <= Math.min(candidate.length - 1, i + window); j++) {
                lo = Math.min(lo, candidate[j]);
                hi = Math.max(hi, candidate[j]);
            }
            double q = query[i];
            if (q > hi) {
                s += (q - hi) * (q - hi);
            } else if (q < lo) {
                s += (q - lo) * (q - lo);
            }
        }
        return s;
    }

    /** An elastic measure with its parameter space. */
    public enum Measure {
        /** Lock-step squared Euclidean. */
        EUCLIDEAN,
        /** DTW with the full window. */
        DTW_FULL,
        /** DTW with a window (parameter: the window share of the length). */
        DTW,
        /** Derivative DTW with a window. */
        DDTW,
        /** Weighted DTW (parameter: g). */
        WDTW,
        /** Weighted derivative DTW. */
        WDDTW,
        /** LCSS (parameters: epsilon, window share). */
        LCSS,
        /** ERP (parameters: g, window share). */
        ERP,
        /** Move-split-merge (parameter: c). */
        MSM,
        /** Time warp edit (parameters: nu, lambda). */
        TWE;

        /**
         * Distance under parameters.
         *
         * @param a one series
         * @param b another
         * @param p parameters from {@link #random} or {@link #grid}
         * @return the distance
         */
        public double distance(double[] a, double[] b, double[] p) {
            int len = a.length;
            return switch (this) {
                case EUCLIDEAN -> euclidean(a, b);
                case DTW_FULL -> dtw(a, b, -1);
                case DTW -> dtw(a, b, (int) Math.round(p[0] * len));
                case DDTW -> dtw(derivative(a), derivative(b), (int) Math.round(p[0] * len));
                case WDTW -> wdtw(a, b, p[0]);
                case WDDTW -> wdtw(derivative(a), derivative(b), p[0]);
                case LCSS -> lcss(a, b, p[0], (int) Math.round(p[1] * len));
                case ERP -> erp(a, b, p[0], (int) Math.round(p[1] * len));
                case MSM -> msm(a, b, p[0]);
                case TWE -> twe(a, b, p[0], p[1]);
            };
        }

        /**
         * Proximity Forest's random parameters.
         *
         * @param sd the standard deviation of the training values (scales LCSS's epsilon and ERP's g)
         * @param r the random source
         * @return the parameters
         */
        public double[] random(double sd, SplittableRandom r) {
            return switch (this) {
                case EUCLIDEAN, DTW_FULL -> new double[0];
                case DTW, DDTW -> new double[] {0.25 * r.nextDouble()};
                case WDTW, WDDTW -> new double[] {r.nextDouble()};
                case LCSS -> new double[] {sd * (0.2 + 0.8 * r.nextDouble()), 0.25 * r.nextDouble()};
                case ERP -> new double[] {sd * (0.2 + 0.8 * r.nextDouble()), 0.25 * r.nextDouble()};
                case MSM -> new double[] {Math.pow(10, -2 + 4 * r.nextDouble())};
                case TWE -> new double[] {Math.pow(10, -5 + 5 * r.nextDouble()), 0.1 * r.nextInt(10) / 9.0};
            };
        }

        /**
         * A small grid of parameters for the Elastic Ensemble's cross-validation.
         *
         * @param sd the training values' standard deviation
         * @return candidate parameter vectors
         */
        public double[][] grid(double sd) {
            return switch (this) {
                case EUCLIDEAN, DTW_FULL -> new double[][] {{}};
                case DTW, DDTW -> new double[][] {{0.02}, {0.05}, {0.1}, {0.2}};
                case WDTW, WDDTW -> new double[][] {{0.01}, {0.1}, {1}};
                case LCSS -> new double[][] {{0.2 * sd, 0.05}, {0.5 * sd, 0.1}, {sd, 0.2}};
                case ERP -> new double[][] {{0.2 * sd, 0.1}, {0.5 * sd, 0.2}, {sd, 0.25}};
                case MSM -> new double[][] {{0.01}, {0.1}, {1}};
                case TWE -> new double[][] {{1e-4, 0}, {1e-3, 0.05}, {1e-2, 0.1}};
            };
        }
    }
}
