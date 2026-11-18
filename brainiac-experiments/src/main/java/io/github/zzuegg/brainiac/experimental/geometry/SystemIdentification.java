package io.github.zzuegg.brainiac.experimental.geometry;

import io.github.zzuegg.brainiac.experimental.sciml.DynamicModeDecomposition;

/**
 * Linear single-input single-output system identification (Ljung, <i>System
 * Identification</i>, 1999):
 *
 * <ul>
 *   <li>{@link #arx ARX}: {@code A(q) y = B(q) u + e}, one linear least-squares solve; biased
 *       when the noise is not white in the equation error;</li>
 *   <li>{@link #armax ARMAX}: {@code A y = B u + C e}, by extended (pseudo-linear) least
 *       squares on the lagged residuals, with {@code C} kept stable and the round with the
 *       smallest residual kept;</li>
 *   <li>{@link #outputError output error}: {@code y = B/F u + e}, by Levenberg-Marquardt on
 *       the simulation error from an ARX start, with a finite-difference Jacobian;</li>
 *   <li>{@link #n4sid N4SID}: subspace identification (Van Overschee &amp; De Moor,
 *       Automatica 1994): the oblique projection of future outputs on past data along future
 *       inputs, its singular values to pick the order, the state sequence from the leading
 *       directions, and {@code (A, B, C, D)} by least squares on consecutive states.</li>
 * </ul>
 */
public final class SystemIdentification {

    private SystemIdentification() {
    }

    /**
     * A polynomial model {@code A(q) y = B(q) / F(q) u + C(q) e}, with monic {@code A}, {@code C}
     * and {@code F} stored without their leading one, and {@code B}'s first tap at delay
     * {@code nk}.
     *
     * @param a the {@code A} coefficients {@code a_1..a_na}
     * @param b the {@code B} coefficients {@code b_0..b_(nb-1)}
     * @param c the {@code C} coefficients
     * @param f the {@code F} coefficients
     * @param nk the input delay
     */
    public record Polynomial(double[] a, double[] b, double[] c, double[] f, int nk) {

        /** @return the denominator {@code A F} without its leading one */
        public double[] denominator() {
            double[] pa = monic(a);
            double[] pf = monic(f);
            double[] d = new double[pa.length + pf.length - 1];
            for (int i = 0; i < pa.length; i++) {
                for (int j = 0; j < pf.length; j++) {
                    d[i + j] += pa[i] * pf[j];
                }
            }
            double[] out = new double[d.length - 1];
            System.arraycopy(d, 1, out, 0, out.length);
            return out;
        }

        /**
         * Simulates the noise-free output {@code B / (A F) u} from zero initial conditions.
         *
         * @param u the input
         * @return the output
         */
        public double[] simulate(double[] u) {
            double[] d = denominator();
            double[] y = new double[u.length];
            for (int t = 0; t < u.length; t++) {
                double v = 0;
                for (int i = 0; i < d.length && t - 1 - i >= 0; i++) {
                    v -= d[i] * y[t - 1 - i];
                }
                for (int j = 0; j < b.length && t - nk - j >= 0; j++) {
                    v += b[j] * u[t - nk - j];
                }
                y[t] = v;
            }
            return y;
        }

        /**
         * The one-step-ahead predictor: ARX and ARMAX use measured outputs (and, for ARMAX,
         * past prediction errors); output-error models have no noise model, so their
         * predictor is the simulation.
         *
         * @param u the input
         * @param y the measured output
         * @return the predictions
         */
        public double[] predict(double[] u, double[] y) {
            if (f.length > 0) {
                return simulate(u);
            }
            double[] p = new double[u.length];
            double[] eps = new double[u.length];
            for (int t = 0; t < u.length; t++) {
                double v = 0;
                for (int i = 0; i < a.length && t - 1 - i >= 0; i++) {
                    v -= a[i] * y[t - 1 - i];
                }
                for (int j = 0; j < b.length && t - nk - j >= 0; j++) {
                    v += b[j] * u[t - nk - j];
                }
                for (int i = 0; i < c.length && t - 1 - i >= 0; i++) {
                    v += c[i] * eps[t - 1 - i];
                }
                p[t] = v;
                eps[t] = y[t] - v;
            }
            return p;
        }

        /** @return the poles {@code [n][2]} (real, imaginary) */
        public double[][] poles() {
            return roots(denominator());
        }
    }

    /**
     * A state-space model {@code x' = A x + B u}, {@code y = C x + D u}.
     *
     * @param a {@code A}
     * @param b {@code B}
     * @param c {@code C}
     * @param d {@code D}
     * @param singularValues the singular values used to choose the order (N4SID), or empty
     */
    public record StateSpace(double[][] a, double[] b, double[] c, double d, double[] singularValues) {

        /**
         * Simulates from a zero state.
         *
         * @param u the input
         * @return the output
         */
        public double[] simulate(double[] u) {
            int n = b.length;
            double[] x = new double[n];
            double[] y = new double[u.length];
            for (int t = 0; t < u.length; t++) {
                y[t] = GeometryLinalg.dot(c, x) + d * u[t];
                double[] nx = GeometryLinalg.multiply(a, x);
                for (int i = 0; i < n; i++) {
                    nx[i] += b[i] * u[t];
                }
                x = nx;
            }
            return y;
        }

        /** @return the poles {@code [n][2]} */
        public double[][] poles() {
            return DynamicModeDecomposition.eigenvalues(a);
        }
    }

    private static double[] monic(double[] p) {
        double[] m = new double[p.length + 1];
        m[0] = 1;
        System.arraycopy(p, 0, m, 1, p.length);
        return m;
    }

    /**
     * The roots of {@code z^n + d_1 z^(n-1) + ... + d_n}, from the companion matrix.
     *
     * @param d the coefficients without the leading one
     * @return the roots {@code [n][2]}
     */
    public static double[][] roots(double[] d) {
        int n = d.length;
        if (n == 0) {
            return new double[0][2];
        }
        double[][] m = new double[n][n];
        for (int j = 0; j < n; j++) {
            m[0][j] = -d[j];
        }
        for (int i = 1; i < n; i++) {
            m[i][i - 1] = 1;
        }
        return DynamicModeDecomposition.eigenvalues(m);
    }

    private static double[] lstsq(double[][] phi, double[] y, double ridge) {
        int p = phi[0].length;
        double[][] g = new double[p][p];
        double[] r = new double[p];
        for (int t = 0; t < phi.length; t++) {
            for (int i = 0; i < p; i++) {
                r[i] += phi[t][i] * y[t];
                for (int j = 0; j < p; j++) {
                    g[i][j] += phi[t][i] * phi[t][j];
                }
            }
        }
        for (int i = 0; i < p; i++) {
            g[i][i] += ridge;
        }
        return GeometryLinalg.solve(g, r);
    }

    private static double[] regressors(double[] u, double[] y, double[] eps, int t, int na, int nb, int nc, int nk) {
        double[] row = new double[na + nb + nc];
        for (int i = 0; i < na; i++) {
            row[i] = -y[t - 1 - i];
        }
        for (int j = 0; j < nb; j++) {
            row[na + j] = u[t - nk - j];
        }
        for (int i = 0; i < nc; i++) {
            row[na + nb + i] = eps[t - 1 - i];
        }
        return row;
    }

    /**
     * Shrinks {@code C(q)} toward one ({@code c_i <- rho^i c_i}) until its roots lie inside
     * radius 0.98, so that the residual recursion of extended least squares stays stable.
     */
    private static void stabilise(double[] theta, int offset, int nc) {
        double[] c = new double[nc];
        System.arraycopy(theta, offset, c, 0, nc);
        double radius = 0;
        for (double[] z : roots(c)) {
            radius = Math.max(radius, Math.hypot(z[0], z[1]));
        }
        if (radius > 0.98) {
            double rho = 0.98 / radius;
            double f = 1;
            for (int i = 0; i < nc; i++) {
                f *= rho;
                theta[offset + i] = c[i] * f;
            }
        }
    }

    /**
     * Fits an ARX model by least squares.
     *
     * @param u the input
     * @param y the output
     * @param na the output lags
     * @param nb the input taps
     * @param nk the input delay
     * @return the model
     */
    public static Polynomial arx(double[] u, double[] y, int na, int nb, int nk) {
        return armax(u, y, na, nb, 0, nk, 0);
    }

    /**
     * Fits an ARMAX model by extended least squares; with {@code nc = 0} it is ARX.
     *
     * @param u the input
     * @param y the output
     * @param na the output lags
     * @param nb the input taps
     * @param nc the noise lags
     * @param nk the input delay
     * @param iterations the residual-refit rounds
     * @return the model
     */
    public static Polynomial armax(double[] u, double[] y, int na, int nb, int nc, int nk, int iterations) {
        int start = Math.max(Math.max(na, nc), nk + nb - 1);
        int n = y.length - start;
        double[] eps = new double[y.length];
        double[] theta = null;
        double[] best = null;
        double bestRss = Double.POSITIVE_INFINITY;
        for (int round = 0; round <= (nc == 0 ? 0 : iterations); round++) {
            double[][] phi = new double[n][];
            double[] target = new double[n];
            for (int t = start; t < y.length; t++) {
                phi[t - start] = regressors(u, y, eps, t, na, nb, round == 0 ? 0 : nc, nk);
                target[t - start] = y[t];
            }
            theta = lstsq(phi, target, 1e-9);
            if (round > 0) {
                stabilise(theta, na + nb, nc);
            }
            double rss = 0;
            for (int t = start; t < y.length; t++) {
                eps[t] = y[t] - GeometryLinalg.dot(regressors(u, y, eps, t, na, nb, round == 0 ? 0 : nc, nk), theta);
                rss += eps[t] * eps[t];
            }
            if (round > 0 && rss < bestRss) {
                bestRss = rss;
                best = theta.clone();
            }
        }
        if (best != null) {
            theta = best;
        }
        double[] a = new double[na];
        double[] b = new double[nb];
        double[] c = new double[theta.length - na - nb];
        System.arraycopy(theta, 0, a, 0, na);
        System.arraycopy(theta, na, b, 0, nb);
        System.arraycopy(theta, na + nb, c, 0, c.length);
        return new Polynomial(a, b, c, new double[0], nk);
    }

    private static double cost(double[] u, double[] y, double[] theta, int nb, int nf, int nk) {
        double[] b = new double[nb];
        double[] f = new double[nf];
        System.arraycopy(theta, 0, b, 0, nb);
        System.arraycopy(theta, nb, f, 0, nf);
        double[] s = new Polynomial(new double[0], b, new double[0], f, nk).simulate(u);
        double c = 0;
        for (int t = 0; t < y.length; t++) {
            double r = y[t] - s[t];
            c += r * r;
        }
        return Double.isFinite(c) ? c : Double.POSITIVE_INFINITY;
    }

    private static double[] residual(double[] u, double[] y, double[] theta, int nb, int nf, int nk) {
        double[] b = new double[nb];
        double[] f = new double[nf];
        System.arraycopy(theta, 0, b, 0, nb);
        System.arraycopy(theta, nb, f, 0, nf);
        double[] s = new Polynomial(new double[0], b, new double[0], f, nk).simulate(u);
        double[] r = new double[y.length];
        for (int t = 0; t < y.length; t++) {
            r[t] = y[t] - s[t];
        }
        return r;
    }

    /**
     * Fits an output-error model by Levenberg-Marquardt on the simulation error, started from
     * ARX with {@code na = nf}.
     *
     * @param u the input
     * @param y the output
     * @param nb the input taps
     * @param nf the denominator order
     * @param nk the input delay
     * @param iterations the iterations
     * @return the model
     */
    public static Polynomial outputError(double[] u, double[] y, int nb, int nf, int nk, int iterations) {
        Polynomial init = arx(u, y, nf, nb, nk);
        int p = nb + nf;
        double[] theta = new double[p];
        System.arraycopy(init.b(), 0, theta, 0, nb);
        System.arraycopy(init.a(), 0, theta, nb, nf);
        double best = cost(u, y, theta, nb, nf, nk);
        double mu = 1e-3;
        for (int it = 0; it < iterations; it++) {
            double[] r = residual(u, y, theta, nb, nf, nk);
            double[][] jac = new double[p][];
            for (int k = 0; k < p; k++) {
                double h = 1e-6 * Math.max(1, Math.abs(theta[k]));
                double[] th = theta.clone();
                th[k] += h;
                double[] rk = residual(u, y, th, nb, nf, nk);
                jac[k] = new double[y.length];
                for (int t = 0; t < y.length; t++) {
                    jac[k][t] = -(rk[t] - r[t]) / h;
                }
            }
            double[][] g = new double[p][p];
            double[] grad = new double[p];
            for (int i = 0; i < p; i++) {
                grad[i] = GeometryLinalg.dot(jac[i], r);
                for (int j = 0; j < p; j++) {
                    g[i][j] = GeometryLinalg.dot(jac[i], jac[j]);
                }
            }
            boolean improved = false;
            for (int tries = 0; tries < 8 && !improved; tries++) {
                double[][] gm = new double[p][];
                for (int i = 0; i < p; i++) {
                    gm[i] = g[i].clone();
                    gm[i][i] += mu * (g[i][i] + 1e-12);
                }
                double[] step = GeometryLinalg.solve(gm, grad);
                if (step == null) {
                    mu *= 10;
                    continue;
                }
                double[] cand = theta.clone();
                for (int i = 0; i < p; i++) {
                    cand[i] += step[i];
                }
                double c = cost(u, y, cand, nb, nf, nk);
                if (c < best) {
                    theta = cand;
                    best = c;
                    mu = Math.max(mu / 3, 1e-9);
                    improved = true;
                } else {
                    mu *= 10;
                }
            }
            if (!improved) {
                break;
            }
        }
        double[] b = new double[nb];
        double[] f = new double[nf];
        System.arraycopy(theta, 0, b, 0, nb);
        System.arraycopy(theta, nb, f, 0, nf);
        return new Polynomial(new double[0], b, new double[0], f, nk);
    }

    /**
     * Subspace identification.
     *
     * @param u the input
     * @param y the output
     * @param horizon the block rows {@code i} of the past and future Hankel matrices
     * @param order the model order, or 0 to pick the largest gap among the first
     *     {@code maxOrder} singular values
     * @param maxOrder the largest order considered when {@code order == 0}
     * @return the model
     */
    public static StateSpace n4sid(double[] u, double[] y, int horizon, int order, int maxOrder) {
        int i = horizon;
        int cols = y.length - 2 * i + 1;
        double[][] z = new double[3 * i][cols];
        double[][] yf = new double[i][cols];
        for (int r = 0; r < i; r++) {
            for (int j = 0; j < cols; j++) {
                z[r][j] = u[j + r];
                z[i + r][j] = y[j + r];
                z[2 * i + r][j] = u[j + i + r];
                yf[r][j] = y[j + i + r];
            }
        }
        double[][] zt = GeometryLinalg.transpose(z);
        double[][] gram = GeometryLinalg.multiply(z, zt);
        double trace = 0;
        for (int k = 0; k < gram.length; k++) {
            trace += gram[k][k];
        }
        for (int k = 0; k < gram.length; k++) {
            gram[k][k] += 1e-10 * trace / gram.length;
        }
        double[][] cross = GeometryLinalg.multiply(yf, zt);
        double[][] l = GeometryLinalg.transpose(GeometryLinalg.solve(gram, GeometryLinalg.transpose(cross)));
        double[][] o = new double[i][cols];
        for (int r = 0; r < i; r++) {
            for (int j = 0; j < cols; j++) {
                double v = 0;
                for (int k = 0; k < 2 * i; k++) {
                    v += l[r][k] * z[k][j];
                }
                o[r][j] = v;
            }
        }
        double[][][] eig = GeometryLinalg.eigh(GeometryLinalg.multiply(o, GeometryLinalg.transpose(o)));
        double[] sv = new double[i];
        for (int k = 0; k < i; k++) {
            sv[k] = Math.sqrt(Math.max(0, eig[0][0][k]));
        }
        int n = order;
        if (n <= 0) {
            double bestGap = -1;
            for (int k = 1; k <= Math.min(maxOrder, i - 1); k++) {
                double gap = sv[k - 1] / Math.max(sv[k], 1e-300);
                if (gap > bestGap) {
                    bestGap = gap;
                    n = k;
                }
            }
        }
        double[][] x = new double[n][cols];
        for (int k = 0; k < n; k++) {
            double s = 1 / Math.sqrt(Math.max(sv[k], 1e-300));
            for (int j = 0; j < cols; j++) {
                double v = 0;
                for (int r = 0; r < i; r++) {
                    v += eig[1][r][k] * o[r][j];
                }
                x[k][j] = s * v;
            }
        }
        double[][] phi = new double[cols - 1][n + 1];
        for (int j = 0; j < cols - 1; j++) {
            for (int k = 0; k < n; k++) {
                phi[j][k] = x[k][j];
            }
            phi[j][n] = u[i + j];
        }
        double[][] a = new double[n][];
        double[] b = new double[n];
        for (int k = 0; k < n; k++) {
            double[] target = new double[cols - 1];
            for (int j = 0; j < cols - 1; j++) {
                target[j] = x[k][j + 1];
            }
            double[] th = lstsq(phi, target, 1e-10);
            a[k] = new double[n];
            System.arraycopy(th, 0, a[k], 0, n);
            b[k] = th[n];
        }
        double[] target = new double[cols - 1];
        for (int j = 0; j < cols - 1; j++) {
            target[j] = y[i + j];
        }
        double[] th = lstsq(phi, target, 1e-10);
        double[] c = new double[n];
        System.arraycopy(th, 0, c, 0, n);
        return new StateSpace(a, b, c, th[n], sv);
    }

    /**
     * The normalised fit {@code 100 (1 - |y - yhat| / |y - mean(y)|)}, in percent.
     *
     * @param y the measured output
     * @param yhat the model output
     * @return the fit
     */
    public static double fit(double[] y, double[] yhat) {
        double mean = 0;
        for (double v : y) {
            mean += v;
        }
        mean /= y.length;
        double num = 0;
        double den = 0;
        for (int t = 0; t < y.length; t++) {
            num += (y[t] - yhat[t]) * (y[t] - yhat[t]);
            den += (y[t] - mean) * (y[t] - mean);
        }
        double f = 100 * (1 - Math.sqrt(num / den));
        return Double.isFinite(f) ? f : Double.NEGATIVE_INFINITY;
    }

    /**
     * The mean distance from each true pole to the nearest estimated one.
     *
     * @param truth the true poles {@code [n][2]}
     * @param estimate the estimated poles
     * @return the distance
     */
    public static double poleError(double[][] truth, double[][] estimate) {
        double sum = 0;
        for (double[] p : truth) {
            double best = Double.POSITIVE_INFINITY;
            for (double[] q : estimate) {
                best = Math.min(best, Math.hypot(p[0] - q[0], p[1] - q[1]));
            }
            sum += best;
        }
        return sum / truth.length;
    }
}
