package io.github.zzuegg.brainiac.experimental.panel;

import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;

/**
 * Two-way fixed-effects regressions on a balanced panel, the reference every modern panel
 * estimator is compared with. In a balanced panel, unit and period dummies are projected out
 * exactly by two-way demeaning ({@code v_it - v_i. - v_.t + v_..}, Frisch-Waugh-Lovell), so the
 * regression of the demeaned outcome on the demeaned regressors gives the TWFE coefficients.
 *
 * <ul>
 *   <li>{@link #did}: {@code y_it = a_i + b_t + beta D_it + e}, the static TWFE DiD.</li>
 *   <li>{@link #eventStudy}: dynamic TWFE with relative-time dummies {@code 1[t - g_i = e]}
 *       for {@code e} in {@code [-leads, lags]}, endpoints binned and {@code e = -1} the
 *       reference.</li>
 * </ul>
 */
public final class TwoWayFixedEffects {

    private TwoWayFixedEffects() {
    }

    /**
     * Two-way demeaning.
     *
     * @param v values {@code [units][periods]}
     * @return the demeaned copy
     */
    public static double[][] demean(double[][] v) {
        int n = v.length;
        int t = v[0].length;
        double[] row = new double[n];
        double[] col = new double[t];
        double all = 0;
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < t; s++) {
                row[i] += v[i][s] / t;
                col[s] += v[i][s] / n;
                all += v[i][s] / (n * (double) t);
            }
        }
        double[][] o = new double[n][t];
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < t; s++) {
                o[i][s] = v[i][s] - row[i] - col[s] + all;
            }
        }
        return o;
    }

    /**
     * Regression on two-way-demeaned regressors.
     *
     * @param y outcomes {@code [units][periods]}
     * @param regressors {@code [k][units][periods]}
     * @return the {@code k} coefficients
     */
    public static double[] regress(double[][] y, double[][][] regressors) {
        int n = y.length;
        int t = y[0].length;
        int k = regressors.length;
        double[][] yd = demean(y);
        double[][][] xd = new double[k][][];
        for (int a = 0; a < k; a++) {
            xd[a] = demean(regressors[a]);
        }
        double[][] x = new double[n * t][k];
        double[] v = new double[n * t];
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < t; s++) {
                v[i * t + s] = yd[i][s];
                for (int a = 0; a < k; a++) {
                    x[i * t + s][a] = xd[a][i][s];
                }
            }
        }
        return LeastSquares.fit(x, v, false, 1e-10).beta();
    }

    /**
     * Static TWFE DiD.
     *
     * @param y outcomes
     * @param treated treatment indicators
     * @return the coefficient on {@code D}
     */
    public static double did(double[][] y, boolean[][] treated) {
        return regress(y, new double[][][] {indicator(treated)})[0];
    }

    /**
     * The 0/1 matrix of a boolean panel.
     *
     * @param b the indicators
     * @return 1 where true
     */
    public static double[][] indicator(boolean[][] b) {
        double[][] o = new double[b.length][b[0].length];
        for (int i = 0; i < b.length; i++) {
            for (int s = 0; s < b[0].length; s++) {
                o[i][s] = b[i][s] ? 1 : 0;
            }
        }
        return o;
    }

    /**
     * Dynamic TWFE event study.
     *
     * @param y outcomes
     * @param adoption first treated period per unit, {@code -1} if never
     * @param leads the earliest event time is {@code -leads} (binned below)
     * @param lags the latest event time (binned above)
     * @return coefficients for {@code e = -leads..lags}, with {@code e = -1} set to 0
     */
    public static double[] eventStudy(double[][] y, int[] adoption, int leads, int lags) {
        int n = y.length;
        int t = y[0].length;
        int span = leads + lags + 1;
        double[][][] reg = new double[span - 1][n][t];
        for (int i = 0; i < n; i++) {
            if (adoption[i] < 0) {
                continue;
            }
            for (int s = 0; s < t; s++) {
                int e = Math.max(-leads, Math.min(lags, s - adoption[i]));
                if (e == -1) {
                    continue;
                }
                int idx = e + leads;
                reg[idx > leads - 1 ? idx - 1 : idx][i][s] = 1;
            }
        }
        double[] b = regress(y, reg);
        double[] out = new double[span];
        for (int e = -leads; e <= lags; e++) {
            int idx = e + leads;
            out[idx] = e == -1 ? 0 : b[idx > leads - 1 ? idx - 1 : idx];
        }
        return out;
    }
}
