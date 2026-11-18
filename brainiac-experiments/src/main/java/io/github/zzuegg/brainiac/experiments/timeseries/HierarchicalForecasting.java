package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.timeseries.ExponentialSmoothing;
import io.github.zzuegg.brainiac.experimental.timeseries.Reconciliation;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Hierarchical forecast reconciliation (wave 11, item 11.3.15) on a 1 / 4 / 16 hierarchy.
 *
 * <p>Each of the 16 leaves is a {@link SeasonalSeries} pattern (period 12) plus noise made of a
 * common factor, a regional factor and an idiosyncratic part, so base-forecast errors are
 * correlated across the hierarchy; regions and the total are sums. Every node gets its own
 * additive Holt-Winters model, ETS(A,N,A) as in {@link ExponentialSmoothing}, run by its
 * error-correction recursions ({@link #recursion}) with the smoothing weights picked from a grid
 * by the one-step SSE on the first 120 steps (the L-BFGS maximum-likelihood fit of
 * {@code ExponentialSmoothing} costs about a second per node here); forecasts for horizons 1 to
 * 6 are made from rolling origins over the remaining steps, and the
 * one-step in-sample residuals feed the variance-based methods. With {@code outliers}, at each
 * origin and with probability 0.3 one random node's base forecasts are shifted by six residual
 * standard deviations. Scores: MASE (scaled by the in-sample seasonal naive error) for the total,
 * the regions, the leaves and all nodes, for the unreconciled base forecasts and every
 * {@link Reconciliation.Method}.
 */
public final class HierarchicalForecasting {

    private HierarchicalForecasting() {
    }

    /**
     * The ETS(A,N,A) error-correction recursions with fitted smoothing weights, states started
     * as in {@link ExponentialSmoothing} (level the first season's mean, seasonals the first
     * season's deviations): {@code l_t = l_t-1 + alpha e_t}, {@code s_t = s_t-m + gamma e_t}.
     *
     * @return {@code [t][h]}: the forecast of {@code y[t + h]} made from {@code y[0..t-1]}
     */
    static double[][] recursion(double[] y, double alpha, double gamma, int period, int horizon) {
        int n = y.length;
        double level = 0;
        for (int k = 0; k < period; k++) {
            level += y[k] / period;
        }
        double[] season = new double[n + horizon + period];
        for (int k = 0; k < period; k++) {
            season[k] = y[k] - level;
        }
        double[][] out = new double[n][horizon];
        for (int t = period; t < n; t++) {
            for (int h = 0; h < horizon; h++) {
                out[t][h] = level + season[t + h - period * (1 + h / period)];
            }
            double e = y[t] - (level + season[t - period]);
            level += alpha * e;
            season[t] = season[t - period] + gamma * e;
        }
        return out;
    }

    /**
     * Runs the comparison.
     *
     * @param outliers whether base forecasts get planted outliers
     * @param hierarchies simulated hierarchies
     * @param seed the seed
     * @return method to {@code {MASE total, MASE regions, MASE leaves, MASE all}}
     */
    public static Map<String, double[]> compare(boolean outliers, int hierarchies, long seed) {
        int[] branching = {4, 4};
        double[][] s = Reconciliation.hierarchy(branching);
        int[] level = Reconciliation.levels(branching);
        int nodes = s.length;
        int leaves = s[0].length;
        int length = 156;
        int train = 120;
        int horizon = 6;
        int period = 12;
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("base forecasts (unreconciled)", new double[4]);
        for (Reconciliation.Method m : Reconciliation.Method.values()) {
            out.put(m.name(), new double[4]);
        }
        int[] perLevel = new int[3];
        for (int lv : level) {
            perLevel[lv]++;
        }
        for (int hIdx = 0; hIdx < hierarchies; hIdx++) {
            SplittableRandom r = new SplittableRandom(seed * 1000 + hIdx);
            double[][] leaf = new double[leaves][length];
            double[] common = new double[length];
            double[][] region = new double[4][length];
            for (int t = 0; t < length; t++) {
                common[t] = r.nextGaussian();
                for (int g = 0; g < 4; g++) {
                    region[g][t] = r.nextGaussian();
                }
            }
            for (int j = 0; j < leaves; j++) {
                float[] pattern = SeasonalSeries.generate(length, period, 0.01 * r.nextGaussian(),
                        0.5 + r.nextDouble(), 0, seed * 7 + hIdx * 31 + j);
                double scale = 0.3 + 0.4 * r.nextDouble();
                for (int t = 0; t < length; t++) {
                    leaf[j][t] = 5 + pattern[t] + scale * (0.6 * common[t] + 0.6 * region[j / 4][t]
                            + 0.5 * r.nextGaussian());
                }
            }
            double[][] y = new double[nodes][length];
            for (int i = 0; i < nodes; i++) {
                for (int j = 0; j < leaves; j++) {
                    if (s[i][j] != 0) {
                        for (int t = 0; t < length; t++) {
                            y[i][t] += leaf[j][t];
                        }
                    }
                }
            }
            double[][][] paths = new double[nodes][][];
            double[] mase = new double[nodes];
            double[] sigma = new double[nodes];
            int start = 2 * period;
            double[][] resid = new double[train - start][nodes];
            for (int i = 0; i < nodes; i++) {
                double[] tr = java.util.Arrays.copyOf(y[i], train);
                // Holt-Winters weights by in-sample one-step SSE on a grid
                double bestSse = Double.POSITIVE_INFINITY;
                for (double alpha : new double[] {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}) {
                    for (double gamma : new double[] {0.01, 0.05, 0.1, 0.2, 0.3}) {
                        double[][] path = recursion(tr, alpha, gamma, period, 1);
                        double sse = 0;
                        for (int t = start; t < train; t++) {
                            sse += (tr[t] - path[t][0]) * (tr[t] - path[t][0]);
                        }
                        if (sse < bestSse) {
                            bestSse = sse;
                            paths[i] = recursion(y[i], alpha, gamma, period, horizon);
                        }
                    }
                }
                for (int t = period; t < train; t++) {
                    mase[i] += Math.abs(y[i][t] - y[i][t - period]) / (train - period);
                }
                for (int t = start; t < train; t++) {
                    double f = paths[i][t][0];
                    resid[t - start][i] = y[i][t] - f;
                    sigma[i] += resid[t - start][i] * resid[t - start][i] / (train - start);
                }
                sigma[i] = Math.sqrt(sigma[i]);
            }
            double[][] cov = Reconciliation.shrinkCovariance(resid);
            int origins = length - horizon - train + 1;
            for (int o = 0; o < origins; o++) {
                int origin = train + o;
                double[][] base = new double[horizon][nodes];
                for (int i = 0; i < nodes; i++) {
                    double[] f = paths[i][origin];
                    for (int h = 0; h < horizon; h++) {
                        base[h][i] = f[h];
                    }
                }
                if (outliers && r.nextDouble() < 0.3) {
                    int i = r.nextInt(nodes);
                    double shift = (r.nextBoolean() ? 6 : -6) * sigma[i];
                    for (int h = 0; h < horizon; h++) {
                        base[h][i] += shift;
                    }
                }
                for (Map.Entry<String, double[]> e : out.entrySet()) {
                    Reconciliation.Method m = e.getKey().startsWith("base") ? null
                            : Reconciliation.Method.valueOf(e.getKey());
                    for (int h = 0; h < horizon; h++) {
                        double[] rec = m == null ? base[h] : Reconciliation.reconcile(s, base[h], m, cov);
                        for (int i = 0; i < nodes; i++) {
                            double err = Math.abs(rec[i] - y[i][origin + h]) / mase[i];
                            double w = 1.0 / (hierarchies * origins * horizon);
                            e.getValue()[level[i]] += err * w / perLevel[level[i]];
                            e.getValue()[3] += err * w / nodes;
                        }
                    }
                }
            }
        }
        return out;
    }
}
