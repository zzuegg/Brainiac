package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Matrix decompositions with gradients: the symmetric eigen decomposition and the thin singular
 * value decomposition of {@link Linalg}, differentiated by the standard formulas (as PyTorch's
 * {@code linalg.eigh} and {@code linalg.svd}). The decomposition and its gradient are computed on
 * the host in double precision and the results are put on the input's device.
 *
 * <p>Eigen and singular vectors are defined up to sign, so a loss should not depend on their signs
 * (for example, use projections {@code u u^T} or squared entries). Gradients become unbounded when
 * two eigenvalues (or two singular values) coincide; such pairs are skipped.
 */
public final class Decompositions {

    private static final double TIE = 1e-12;

    private Decompositions() {
    }

    /**
     * The eigen decomposition of symmetric matrices with gradients; only the symmetric part of
     * {@code a} is used, and its gradient is symmetric.
     *
     * @param a symmetric matrices {@code [..., n, n]}
     * @return {values {@code [..., n]} ascending, vectors {@code [..., n, n]} one per column}
     */
    public static Var[] eigh(Var a) {
        Tensor[] d = Linalg.eigh(a.value());
        Tensor values = d[0];
        Tensor vectors = d[1];
        int n = a.shape().dim(a.shape().rank() - 1);
        int batch = Math.toIntExact(a.value().size() / ((long) n * n));
        double[] lam = values.toDoubleArray();
        double[] vec = vectors.toDoubleArray();
        Var vals = Var.custom(in -> values, (in, out, g) -> new Tensor[] {
            eighBack(g.toDoubleArray(), null, lam, vec, n, batch, a.value())}, a);
        Var vecs = Var.custom(in -> vectors, (in, out, g) -> new Tensor[] {
            eighBack(null, g.toDoubleArray(), lam, vec, n, batch, a.value())}, a);
        return new Var[] {vals, vecs};
    }

    // A-bar = V (diag(gl) + F o (V^T gV)) V^T, symmetrized; F_ij = 1 / (l_j - l_i)
    private static Tensor eighBack(double[] gl, double[] gv, double[] lam, double[] vec, int n, int batch,
            Tensor like) {
        double[] out = new double[batch * n * n];
        double[][] inner = new double[n][n];
        for (int b = 0; b < batch; b++) {
            int base = b * n * n;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double x = 0;
                    if (gl != null) {
                        x = i == j ? gl[b * n + i] : 0;
                    } else if (i != j) {
                        double gap = lam[b * n + j] - lam[b * n + i];
                        if (Math.abs(gap) > TIE) {
                            double vtg = 0;
                            for (int k = 0; k < n; k++) {
                                vtg += vec[base + k * n + i] * gv[base + k * n + j];
                            }
                            x = vtg / gap;
                        }
                    }
                    inner[i][j] = x;
                }
            }
            double[][] abar = sandwich(vec, base, n, n, inner, vec, base, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    out[base + i * n + j] = 0.5 * (abar[i][j] + abar[j][i]);
                }
            }
        }
        return Tensor.of(like.shape(), out, like.dtype(), like.device());
    }

    /**
     * The thin singular value decomposition with gradients, as {@code torch.linalg.svd(a,
     * full_matrices=False)}.
     *
     * @param a matrices {@code [..., m, n]}
     * @return {u {@code [..., m, k]}, s {@code [..., k]}, vt {@code [..., k, n]}} with
     *     {@code k = min(m, n)}
     */
    public static Var[] svd(Var a) {
        Tensor[] d = Linalg.svd(a.value());
        int r = a.shape().rank();
        int m = a.shape().dim(r - 2);
        int n = a.shape().dim(r - 1);
        int k = Math.min(m, n);
        int batch = Math.toIntExact(a.value().size() / ((long) m * n));
        double[] u = d[0].toDoubleArray();
        double[] s = d[1].toDoubleArray();
        double[] vt = d[2].toDoubleArray();
        Svd f = new Svd(u, s, vt, m, n, k, batch, a.value());
        Var uv = Var.custom(in -> d[0], (in, out, g) -> new Tensor[] {f.back(g.toDoubleArray(), null, null)}, a);
        Var sv = Var.custom(in -> d[1], (in, out, g) -> new Tensor[] {f.back(null, g.toDoubleArray(), null)}, a);
        Var vtv = Var.custom(in -> d[2], (in, out, g) -> new Tensor[] {f.back(null, null, g.toDoubleArray())}, a);
        return new Var[] {uv, sv, vtv};
    }

    private record Svd(double[] u, double[] s, double[] vt, int m, int n, int k, int batch, Tensor like) {

        // Townsend (2016): A-bar = U [(F o (U^T gU - gU^T U)) S + diag(gS) + S (F o (V^T gV - gV^T V))] V^T
        //   + (I - U U^T) gU S^-1 V^T + U S^-1 gV^T (I - V V^T),  F_ij = 1 / (s_j^2 - s_i^2)
        Tensor back(double[] gu, double[] gs, double[] gvt) {
            double[] out = new double[batch * m * n];
            for (int b = 0; b < batch; b++) {
                int ub = b * m * k;
                int vb = b * k * n;
                double[] sv = new double[k];
                System.arraycopy(s, b * k, sv, 0, k);
                // V as n x k from vt (k x n)
                double[][] uu = new double[m][k];
                double[][] vv = new double[n][k];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < k; j++) {
                        uu[i][j] = u[ub + i * k + j];
                    }
                }
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < n; j++) {
                        vv[j][i] = vt[vb + i * n + j];
                    }
                }
                double[][] inner = new double[k][k];
                double[][] extra = new double[m][n];
                if (gs != null) {
                    for (int i = 0; i < k; i++) {
                        inner[i][i] = gs[b * k + i];
                    }
                }
                if (gu != null) {
                    double[][] g = new double[m][k];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < k; j++) {
                            g[i][j] = gu[ub + i * k + j];
                        }
                    }
                    double[][] utg = transposeTimes(uu, g);
                    for (int i = 0; i < k; i++) {
                        for (int j = 0; j < k; j++) {
                            if (i != j) {
                                inner[i][j] += f(sv, i, j) * (utg[i][j] - utg[j][i]) * sv[j];
                            }
                        }
                    }
                    if (m > k) {
                        // (I - U U^T) gU S^-1 V^T
                        double[][] p = minus(g, times(uu, utg));
                        for (int i = 0; i < m; i++) {
                            for (int j = 0; j < k; j++) {
                                p[i][j] = sv[j] > TIE ? p[i][j] / sv[j] : 0;
                            }
                        }
                        add(extra, timesTranspose(p, vv));
                    }
                }
                if (gvt != null) {
                    double[][] g = new double[n][k];   // gradient of V
                    for (int i = 0; i < k; i++) {
                        for (int j = 0; j < n; j++) {
                            g[j][i] = gvt[vb + i * n + j];
                        }
                    }
                    double[][] vtg = transposeTimes(vv, g);
                    for (int i = 0; i < k; i++) {
                        for (int j = 0; j < k; j++) {
                            if (i != j) {
                                inner[i][j] += sv[i] * f(sv, i, j) * (vtg[i][j] - vtg[j][i]);
                            }
                        }
                    }
                    if (n > k) {
                        // U S^-1 gV^T (I - V V^T) = U S^-1 ((I - V V^T) gV)^T
                        double[][] p = minus(g, times(vv, vtg));
                        double[][] us = new double[m][k];
                        for (int i = 0; i < m; i++) {
                            for (int j = 0; j < k; j++) {
                                us[i][j] = sv[j] > TIE ? uu[i][j] / sv[j] : 0;
                            }
                        }
                        add(extra, timesTranspose(us, p));
                    }
                }
                double[][] main = timesTranspose(times(uu, inner), vv);
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        out[b * m * n + i * n + j] = main[i][j] + extra[i][j];
                    }
                }
            }
            return Tensor.of(like.shape(), out, like.dtype(), like.device());
        }

        private static double f(double[] s, int i, int j) {
            double gap = s[j] * s[j] - s[i] * s[i];
            return Math.abs(gap) > TIE ? 1 / gap : 0;
        }
    }

    // X Y Z^T with X and Z read from flat row-major arrays of p x q
    private static double[][] sandwich(double[] x, int xb, int p, int q, double[][] y, double[] z, int zb, int r) {
        double[][] xm = new double[p][q];
        double[][] zm = new double[r][q];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                xm[i][j] = x[xb + i * q + j];
            }
        }
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < q; j++) {
                zm[i][j] = z[zb + i * q + j];
            }
        }
        return timesTranspose(times(xm, y), zm);
    }

    private static double[][] times(double[][] a, double[][] b) {
        int p = a.length;
        int q = b.length;
        int r = q == 0 ? 0 : b[0].length;
        double[][] c = new double[p][r];
        for (int i = 0; i < p; i++) {
            for (int l = 0; l < q; l++) {
                double x = a[i][l];
                for (int j = 0; j < r; j++) {
                    c[i][j] += x * b[l][j];
                }
            }
        }
        return c;
    }

    // A^T B
    private static double[][] transposeTimes(double[][] a, double[][] b) {
        int p = a[0].length;
        int r = b[0].length;
        double[][] c = new double[p][r];
        for (int l = 0; l < a.length; l++) {
            for (int i = 0; i < p; i++) {
                double x = a[l][i];
                for (int j = 0; j < r; j++) {
                    c[i][j] += x * b[l][j];
                }
            }
        }
        return c;
    }

    // A B^T
    private static double[][] timesTranspose(double[][] a, double[][] b) {
        double[][] c = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                double x = 0;
                for (int l = 0; l < a[i].length; l++) {
                    x += a[i][l] * b[j][l];
                }
                c[i][j] = x;
            }
        }
        return c;
    }

    private static double[][] minus(double[][] a, double[][] b) {
        double[][] c = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i].clone();
            for (int j = 0; j < c[i].length; j++) {
                c[i][j] -= b[i][j];
            }
        }
        return c;
    }

    private static void add(double[][] into, double[][] b) {
        for (int i = 0; i < into.length; i++) {
            for (int j = 0; j < into[i].length; j++) {
                into[i][j] += b[i][j];
            }
        }
    }
}
