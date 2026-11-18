package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.function.Function;

/**
 * Dynamic mode decomposition and its extended form: linear Koopman approximations fitted to
 * snapshot pairs.
 *
 * <ul>
 *   <li>{@link #dmd} (Schmid, 2010): with snapshot matrices {@code X = [x_0 .. x_{m-1}]},
 *       {@code Y = [x_1 .. x_m]} and the rank-{@code r} SVD {@code X = U S V^T}, the reduced
 *       operator is {@code A~ = U^T Y V S^-1}; its eigenvalues give the frequencies and growth
 *       rates of the modes, and {@code x_{k+1} = U A~ U^T x_k} forecasts.
 *   <li>{@link #edmd} (Williams, Kevrekidis and Rowley, 2015): the same on a dictionary of
 *       nonlinear observables {@code psi(x)} (which must contain the state itself so it can be
 *       read back), {@code K = argmin |Psi(X) K - Psi(Y)|}.
 * </ul>
 *
 * <p>The SVD is the experimental {@link ThinSvd}; the eigenvalues of the small non-symmetric
 * reduced operator come from a shifted Hessenberg QR iteration.
 *
 * <p>Papers: <a href="https://doi.org/10.1017/S0022112010001217">Schmid, JFM 2010</a>,
 * <a href="https://arxiv.org/abs/1408.4408">arXiv:1408.4408</a>.
 */
public final class DynamicModeDecomposition {

    private final double[][] operator;
    private final double[][] basis;
    private final Function<double[], double[]> dictionary;
    private final int stateSize;

    private DynamicModeDecomposition(double[][] operator, double[][] basis, Function<double[], double[]> dictionary,
            int stateSize) {
        this.operator = operator;
        this.basis = basis;
        this.dictionary = dictionary;
        this.stateSize = stateSize;
    }

    /**
     * Exact DMD on a trajectory.
     *
     * @param snapshots states {@code [m + 1][n]}, uniformly spaced in time
     * @param rank the SVD truncation, at most {@code min(n, m)}; singular values below
     *     {@code 1e-6} of the largest are always dropped
     * @return the model
     */
    public static DynamicModeDecomposition dmd(double[][] snapshots, int rank) {
        int m = snapshots.length - 1;
        int n = snapshots[0].length;
        double[][] x = new double[n][m];
        double[][] y = new double[n][m];
        for (int t = 0; t < m; t++) {
            for (int i = 0; i < n; i++) {
                x[i][t] = snapshots[t][i];
                y[i][t] = snapshots[t + 1][i];
            }
        }
        ThinSvd svd = ThinSvd.of(x);
        int r = Math.min(rank, svd.rank());
        while (r > 1 && svd.s()[r - 1] < 1e-6 * svd.s()[0]) {
            r--;   // ThinSvd goes through the Gram matrix: smaller values are round-off
        }
        double[][] u = new double[n][r];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < r; k++) {
                u[i][k] = svd.u()[i][k];
            }
        }
        // A~ = U^T Y V S^-1
        double[][] yv = new double[n][r];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < r; k++) {
                double s = 0;
                for (int t = 0; t < m; t++) {
                    s += y[i][t] * svd.v()[t][k];
                }
                yv[i][k] = s / svd.s()[k];
            }
        }
        double[][] a = new double[r][r];
        for (int p = 0; p < r; p++) {
            for (int k = 0; k < r; k++) {
                double s = 0;
                for (int i = 0; i < n; i++) {
                    s += u[i][p] * yv[i][k];
                }
                a[p][k] = s;
            }
        }
        return new DynamicModeDecomposition(a, u, null, n);
    }

    /**
     * Extended DMD on a dictionary of observables.
     *
     * @param snapshots states {@code [m + 1][n]}
     * @param dictionary observables {@code psi(x)} whose first {@code n} entries are the state
     * @param ridge least-squares regularisation
     * @return the model
     */
    public static DynamicModeDecomposition edmd(double[][] snapshots, Function<double[], double[]> dictionary,
            double ridge) {
        int m = snapshots.length - 1;
        double[][] px = new double[m][];
        double[][] py = new double[m][];
        for (int t = 0; t < m; t++) {
            px[t] = dictionary.apply(snapshots[t]);
            py[t] = dictionary.apply(snapshots[t + 1]);
        }
        double[][] k = Sindy.lstsq(px, py, ridge);   // psi(y)^T ~ psi(x)^T K
        int p = k.length;
        double[][] kt = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                kt[i][j] = k[j][i];
            }
        }
        return new DynamicModeDecomposition(kt, null, dictionary, snapshots[0].length);
    }

    /** @return the (reduced) linear operator */
    public double[][] operator() {
        double[][] c = new double[operator.length][];
        for (int i = 0; i < c.length; i++) {
            c[i] = operator[i].clone();
        }
        return c;
    }

    /**
     * Forecasts by iterating the linear model.
     *
     * @param x0 the initial state
     * @param steps the steps
     * @return states {@code [steps + 1][n]}
     */
    public double[][] forecast(double[] x0, int steps) {
        double[][] out = new double[steps + 1][];
        out[0] = x0.clone();
        double[] z = basis != null ? project(x0) : dictionary.apply(x0);
        for (int t = 1; t <= steps; t++) {
            z = multiply(operator, z);
            out[t] = basis != null ? lift(z) : java.util.Arrays.copyOf(z, stateSize);
        }
        return out;
    }

    private double[] project(double[] x) {
        double[] z = new double[basis[0].length];
        for (int k = 0; k < z.length; k++) {
            for (int i = 0; i < x.length; i++) {
                z[k] += basis[i][k] * x[i];
            }
        }
        return z;
    }

    private double[] lift(double[] z) {
        double[] x = new double[basis.length];
        for (int i = 0; i < x.length; i++) {
            for (int k = 0; k < z.length; k++) {
                x[i] += basis[i][k] * z[k];
            }
        }
        return x;
    }

    private static double[] multiply(double[][] a, double[] z) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double s = 0;
            for (int j = 0; j < z.length; j++) {
                s += a[i][j] * z[j];
            }
            out[i] = s;
        }
        return out;
    }

    /** @return the eigenvalues of the operator, {@code [k][2]} (real, imaginary) */
    public double[][] eigenvalues() {
        return eigenvalues(operator);
    }

    /**
     * The eigenvalues of a real square matrix: reduction to Hessenberg form, then the shifted
     * QR iteration (the classic {@code hqr} algorithm).
     *
     * @param matrix the matrix
     * @return eigenvalues {@code [n][2]} (real, imaginary)
     */
    public static double[][] eigenvalues(double[][] matrix) {
        int n = matrix.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) {
            a[i] = matrix[i].clone();
        }
        hessenberg(a);
        double[] wr = new double[n];
        double[] wi = new double[n];
        hqr(a, wr, wi);
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            out[i] = new double[] {wr[i], wi[i]};
        }
        return out;
    }

    private static void hessenberg(double[][] a) {
        int n = a.length;
        for (int m = 1; m < n - 1; m++) {
            double x = 0;
            int i = m;
            for (int j = m; j < n; j++) {
                if (Math.abs(a[j][m - 1]) > Math.abs(x)) {
                    x = a[j][m - 1];
                    i = j;
                }
            }
            if (i != m) {
                for (int j = m - 1; j < n; j++) {
                    double t = a[i][j];
                    a[i][j] = a[m][j];
                    a[m][j] = t;
                }
                for (int j = 0; j < n; j++) {
                    double t = a[j][i];
                    a[j][i] = a[j][m];
                    a[j][m] = t;
                }
            }
            if (x != 0) {
                for (i = m + 1; i < n; i++) {
                    double y = a[i][m - 1];
                    if (y != 0) {
                        y /= x;
                        a[i][m - 1] = y;
                        for (int j = m; j < n; j++) {
                            a[i][j] -= y * a[m][j];
                        }
                        for (int j = 0; j < n; j++) {
                            a[j][m] += y * a[j][i];
                        }
                    }
                }
            }
        }
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                a[i][j] = 0;
            }
        }
    }

    private static void hqr(double[][] a, double[] wr, double[] wi) {
        int n = a.length;
        double anorm = 0;
        for (int i = 0; i < n; i++) {
            for (int j = Math.max(i - 1, 0); j < n; j++) {
                anorm += Math.abs(a[i][j]);
            }
        }
        int nn = n - 1;
        double t = 0;
        double p = 0;
        double q = 0;
        double r = 0;
        double s;
        double w;
        double x;
        double y;
        double z = 0;
        while (nn >= 0) {
            int its = 0;
            int l;
            do {
                for (l = nn; l >= 1; l--) {
                    s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l]);
                    if (s == 0) {
                        s = anorm;
                    }
                    if (Math.abs(a[l][l - 1]) + s == s) {
                        a[l][l - 1] = 0;
                        break;
                    }
                }
                x = a[nn][nn];
                if (l == nn) {
                    wr[nn] = x + t;
                    wi[nn--] = 0;
                } else {
                    y = a[nn - 1][nn - 1];
                    w = a[nn][nn - 1] * a[nn - 1][nn];
                    if (l == nn - 1) {
                        p = 0.5 * (y - x);
                        q = p * p + w;
                        z = Math.sqrt(Math.abs(q));
                        x += t;
                        if (q >= 0) {
                            z = p + Math.copySign(z, p);
                            wr[nn - 1] = x + z;
                            wr[nn] = z != 0 ? x - w / z : x + z;
                            wi[nn - 1] = 0;
                            wi[nn] = 0;
                        } else {
                            wr[nn - 1] = x + p;
                            wr[nn] = x + p;
                            wi[nn - 1] = -z;
                            wi[nn] = z;
                        }
                        nn -= 2;
                    } else {
                        if (its == 60) {
                            throw new IllegalStateException("eigenvalue iteration did not converge");
                        }
                        if (its == 10 || its == 20) {
                            t += x;
                            for (int i = 0; i <= nn; i++) {
                                a[i][i] -= x;
                            }
                            s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2]);
                            x = 0.75 * s;
                            y = x;
                            w = -0.4375 * s * s;
                        }
                        ++its;
                        int m;
                        for (m = nn - 2; m >= l; m--) {
                            z = a[m][m];
                            r = x - z;
                            s = y - z;
                            p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
                            q = a[m + 1][m + 1] - z - r - s;
                            r = a[m + 2][m + 1];
                            s = Math.abs(p) + Math.abs(q) + Math.abs(r);
                            p /= s;
                            q /= s;
                            r /= s;
                            if (m == l) {
                                break;
                            }
                            double u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r));
                            double v = Math.abs(p) * (Math.abs(a[m - 1][m - 1]) + Math.abs(z)
                                    + Math.abs(a[m + 1][m + 1]));
                            if (u + v == v) {
                                break;
                            }
                        }
                        for (int i = m; i < nn - 1; i++) {
                            a[i + 2][i] = 0;
                            if (i != m) {
                                a[i + 2][i - 1] = 0;
                            }
                        }
                        for (int k = m; k < nn; k++) {
                            if (k != m) {
                                p = a[k][k - 1];
                                q = a[k + 1][k - 1];
                                r = 0;
                                if (k + 1 != nn) {
                                    r = a[k + 2][k - 1];
                                }
                                x = Math.abs(p) + Math.abs(q) + Math.abs(r);
                                if (x != 0) {
                                    p /= x;
                                    q /= x;
                                    r /= x;
                                }
                            }
                            s = Math.copySign(Math.sqrt(p * p + q * q + r * r), p);
                            if (s != 0) {
                                if (k == m) {
                                    if (l != m) {
                                        a[k][k - 1] = -a[k][k - 1];
                                    }
                                } else {
                                    a[k][k - 1] = -s * x;
                                }
                                p += s;
                                x = p / s;
                                y = q / s;
                                z = r / s;
                                q /= p;
                                r /= p;
                                for (int j = k; j <= nn; j++) {
                                    p = a[k][j] + q * a[k + 1][j];
                                    if (k + 1 != nn) {
                                        p += r * a[k + 2][j];
                                        a[k + 2][j] -= p * z;
                                    }
                                    a[k + 1][j] -= p * y;
                                    a[k][j] -= p * x;
                                }
                                int mmin = nn < k + 3 ? nn : k + 3;
                                for (int i = l; i <= mmin; i++) {
                                    p = x * a[i][k] + y * a[i][k + 1];
                                    if (k + 1 != nn) {
                                        p += z * a[i][k + 2];
                                        a[i][k + 2] -= p * r;
                                    }
                                    a[i][k + 1] -= p * q;
                                    a[i][k] -= p;
                                }
                            }
                        }
                    }
                }
            } while (l < nn - 1);
        }
    }
}
