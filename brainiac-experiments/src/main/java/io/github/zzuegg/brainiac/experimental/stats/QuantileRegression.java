package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.ml.LinearProgram;

import java.util.Arrays;

/**
 * Linear quantile regression (Koenker and Bassett, Econometrica 1978): minimise the pinball loss
 * {@code sum rho_tau(y - b0 - x b)}, {@code rho_tau(u) = u (tau - 1{u < 0})}, which gives the
 * conditional {@code tau}-quantile and is robust to outliers in {@code y}.
 *
 * <ul>
 *   <li>{@link Solver#LP}: the exact solution through the dual linear program
 *       {@code max y'a s.t. X'a = 0, -(1 - tau) <= a <= tau}, solved by the repo's
 *       {@link LinearProgram} simplex; the coefficients are the constraint duals.</li>
 *   <li>{@link Solver#IRLS}: iteratively reweighted least squares with weights
 *       {@code tau / |r|} or {@code (1 - tau) / |r|}, a fast approximation.</li>
 *   <li>{@link Solver#LP_L1}: the l1-penalised version {@code + lambda sum |b_j|} (the same dual
 *       with {@code |X_j'a| <= lambda} for the slopes), sparse quantile regression.</li>
 * </ul>
 * {@link #rearrange} applies the monotone rearrangement of Chernozhukov, Fernandez-Val and
 * Galichon (Econometrica 2010, <a href="https://arxiv.org/abs/0704.3649">arXiv:0704.3649</a>):
 * sorting the predicted quantiles of each point removes every crossing and never increases the
 * pinball loss.
 */
public final class QuantileRegression {

    /** The solver. */
    public enum Solver {
        /** The exact LP. */
        LP,
        /** IRLS. */
        IRLS,
        /** The l1-penalised LP. */
        LP_L1
    }

    /**
     * A fit.
     *
     * @param tau the quantile level
     * @param intercept the intercept
     * @param beta the slopes
     */
    public record Fit(double tau, double intercept, double[] beta) {

        /**
         * Predicts the quantile.
         *
         * @param x a row
         * @return {@code b0 + x b}
         */
        public double predict(double[] x) {
            double s = intercept;
            for (int j = 0; j < beta.length; j++) {
                s += x[j] * beta[j];
            }
            return s;
        }
    }

    private QuantileRegression() {
    }

    /**
     * Fits one quantile.
     *
     * @param solver the solver
     * @param x the design {@code [n][p]}
     * @param y the response
     * @param tau the level in {@code (0, 1)}
     * @param lambda the l1 level for {@link Solver#LP_L1} (total, not per row), else ignored
     * @return the fit
     */
    public static Fit fit(Solver solver, double[][] x, double[] y, double tau, double lambda) {
        return switch (solver) {
            case LP -> lp(x, y, tau, 0);
            case LP_L1 -> lp(x, y, tau, lambda);
            case IRLS -> irls(x, y, tau);
        };
    }

    private static Fit lp(double[][] x, double[] y, double tau, double lambda) {
        int n = y.length;
        int p = x[0].length;
        int q = p + 1;
        double[] c = new double[2 * n];
        for (int i = 0; i < n; i++) {
            c[i] = y[i];
            c[n + i] = -y[i];
        }
        double[][] a = new double[2 * q + 2 * n][2 * n];
        double[] b = new double[2 * q + 2 * n];
        for (int j = 0; j < q; j++) {
            for (int i = 0; i < n; i++) {
                double z = j == 0 ? 1 : x[i][j - 1];
                a[j][i] = z;
                a[j][n + i] = -z;
                a[q + j][i] = -z;
                a[q + j][n + i] = z;
            }
            b[j] = b[q + j] = j == 0 ? 0 : lambda;
        }
        for (int i = 0; i < n; i++) {
            a[2 * q + i][i] = 1;
            b[2 * q + i] = tau;
            a[2 * q + n + i][n + i] = 1;
            b[2 * q + n + i] = 1 - tau;
        }
        LinearProgram.Solution s = LinearProgram.maximize(c, a, b);
        double[] coef = new double[q];
        for (int j = 0; j < q; j++) {
            coef[j] = s.dual()[j] - s.dual()[q + j];
        }
        Fit plus = new Fit(tau, coef[0], Arrays.copyOfRange(coef, 1, q));
        Fit minus = new Fit(tau, -coef[0], negate(Arrays.copyOfRange(coef, 1, q)));
        return pinball(plus, x, y) <= pinball(minus, x, y) ? plus : minus;
    }

    private static double[] negate(double[] v) {
        double[] o = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            o[i] = -v[i];
        }
        return o;
    }

    private static Fit irls(double[][] x, double[] y, double tau) {
        int n = y.length;
        double[] w = new double[n];
        Arrays.fill(w, 1);
        RobustRegression.Fit f = RobustRegression.weighted(x, y, w);
        for (int it = 0; it < 100; it++) {
            double[] r = f.residuals(x, y);
            for (int i = 0; i < n; i++) {
                w[i] = (r[i] >= 0 ? tau : 1 - tau) / Math.max(Math.abs(r[i]), 1e-6);
            }
            RobustRegression.Fit next = RobustRegression.weighted(x, y, w);
            double change = Math.abs(next.intercept() - f.intercept());
            for (int j = 0; j < next.beta().length; j++) {
                change = Math.max(change, Math.abs(next.beta()[j] - f.beta()[j]));
            }
            f = next;
            if (change < 1e-9) {
                break;
            }
        }
        return new Fit(tau, f.intercept(), f.beta());
    }

    /**
     * The mean pinball loss of a fit.
     *
     * @param f the fit
     * @param x rows
     * @param y targets
     * @return the loss
     */
    public static double pinball(Fit f, double[][] x, double[] y) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            s += pinball(y[i] - f.predict(x[i]), f.tau());
        }
        return s / y.length;
    }

    /**
     * The pinball loss of one residual.
     *
     * @param u the residual {@code y - q}
     * @param tau the level
     * @return {@code u (tau - 1{u < 0})}
     */
    public static double pinball(double u, double tau) {
        return u >= 0 ? tau * u : (tau - 1) * u;
    }

    /**
     * Predicts several quantiles per row.
     *
     * @param fits fits for increasing levels
     * @param x rows
     * @return {@code [n][levels]}
     */
    public static double[][] predict(Fit[] fits, double[][] x) {
        double[][] q = new double[x.length][fits.length];
        for (int i = 0; i < x.length; i++) {
            for (int k = 0; k < fits.length; k++) {
                q[i][k] = fits[k].predict(x[i]);
            }
        }
        return q;
    }

    /**
     * Monotone rearrangement: sorts each row's predicted quantiles.
     *
     * @param quantiles {@code [n][levels]} for increasing levels
     * @return a sorted copy
     */
    public static double[][] rearrange(double[][] quantiles) {
        double[][] o = new double[quantiles.length][];
        for (int i = 0; i < o.length; i++) {
            o[i] = quantiles[i].clone();
            Arrays.sort(o[i]);
        }
        return o;
    }

    /**
     * The share of rows whose predicted quantiles are not non-decreasing.
     *
     * @param quantiles {@code [n][levels]}
     * @return the crossing rate
     */
    public static double crossingRate(double[][] quantiles) {
        int c = 0;
        for (double[] row : quantiles) {
            for (int k = 1; k < row.length; k++) {
                if (row[k] < row[k - 1] - 1e-12) {
                    c++;
                    break;
                }
            }
        }
        return (double) c / quantiles.length;
    }
}
