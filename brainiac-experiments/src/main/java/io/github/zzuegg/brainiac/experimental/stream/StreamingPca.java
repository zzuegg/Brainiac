package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.experimental.train.HebbianRule;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Streaming principal subspace estimation, one point at a time with {@code O(dk)} memory.
 *
 * <ul>
 *   <li>{@link Kind#OJA_K}: Oja's rule {@code w ← w + η y (x - y w)} per component, through
 *       {@link HebbianRule#oja} (the exact Oja term), followed by a QR (Gram–Schmidt)
 *       orthonormalisation so the {@code k} columns span the top subspace instead of all
 *       converging to the first component (Oja, J. Math. Biol. 1982).</li>
 *   <li>{@link Kind#CCIPCA}: candid covariance-free incremental PCA (Weng, Zhang and Hwang,
 *       TPAMI 2003): {@code v_i ← ((n-1-l)/n) v_i + ((1+l)/n) u (u·v_i)/||v_i||} with amnesic
 *       parameter {@code l} and deflation {@code u ← u - (u·v̂_i) v̂_i}; no step size to tune.</li>
 *   <li>{@link Kind#INCREMENTAL_SVD}: Brand's incremental thin SVD (ECCV 2002; with Ross et al.'s
 *       forgetting factor, IJCV 2008): every block {@code B} of points is split into its
 *       projection {@code L = UᵀB} and residual {@code H = B - UL}, the small matrix
 *       {@code [[f S, L], [0, R]]} (with {@code QR = H}) is decomposed, and the top {@code k}
 *       singular vectors are rotated back; with {@code f = 1} it is exact up to truncation.</li>
 * </ul>
 *
 * <p>All three track a running mean and work on centred points.
 */
public final class StreamingPca {

    /** The update rule. */
    public enum Kind {
        /** Oja's rule with QR orthonormalisation */
        OJA_K,
        /** candid covariance-free incremental PCA */
        CCIPCA,
        /** Brand's incremental SVD */
        INCREMENTAL_SVD
    }

    private final Kind kind;
    private final int d;
    private final int k;
    private final double rate;
    private final double amnesic;
    private final double forget;
    private final int block;
    private final double[] mean;
    private long n;
    private double[][] w;
    private double[] s;
    private final List<double[]> pending = new ArrayList<>();
    private final HebbianRule oja;

    private StreamingPca(Kind kind, int d, int k, double rate, double amnesic, double forget, int block, long seed) {
        this.kind = kind;
        this.d = d;
        this.k = k;
        this.rate = rate;
        this.amnesic = amnesic;
        this.forget = forget;
        this.block = block;
        this.mean = new double[d];
        this.oja = HebbianRule.oja(rate);
        SplittableRandom r = new SplittableRandom(seed);
        this.w = new double[d][k];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < k; j++) {
                w[i][j] = r.nextGaussian();
            }
        }
        if (kind == Kind.OJA_K) {
            orthonormalise(w);
        } else if (kind == Kind.CCIPCA) {
            this.w = new double[d][k];
        }
        this.s = new double[k];
    }

    /**
     * Oja's rule with QR.
     *
     * @param d the dimension
     * @param k the components
     * @param rate the step size {@code η}
     * @param seed the seed of the random start
     * @return the estimator
     */
    public static StreamingPca oja(int d, int k, double rate, long seed) {
        return new StreamingPca(Kind.OJA_K, d, k, rate, 0, 1, 1, seed);
    }

    /**
     * CCIPCA.
     *
     * @param d the dimension
     * @param k the components
     * @param amnesic the amnesic parameter {@code l}, 2 in the paper (0 for plain averaging)
     * @param seed the seed of the random start
     * @return the estimator
     */
    public static StreamingPca ccipca(int d, int k, double amnesic, long seed) {
        return new StreamingPca(Kind.CCIPCA, d, k, 0, amnesic, 1, 1, seed);
    }

    /**
     * Brand's incremental SVD.
     *
     * @param d the dimension
     * @param k the components
     * @param forget the forgetting factor {@code f} in {@code (0, 1]} (1 forgets nothing)
     * @param block the points per update
     * @return the estimator
     */
    public static StreamingPca incrementalSvd(int d, int k, double forget, int block) {
        return new StreamingPca(Kind.INCREMENTAL_SVD, d, k, 0, 0, forget, block, 0);
    }

    /**
     * Absorbs one point.
     *
     * @param x the point
     */
    public void add(double[] x) {
        n++;
        double[] c = new double[d];
        for (int i = 0; i < d; i++) {
            mean[i] += (x[i] - mean[i]) / n;
            c[i] = x[i] - mean[i];
        }
        switch (kind) {
            case OJA_K -> {
                double[] y = new double[k];
                for (int j = 0; j < k; j++) {
                    for (int i = 0; i < d; i++) {
                        y[j] += w[i][j] * c[i];
                    }
                }
                oja.update(w, c, y);
                orthonormalise(w);
            }
            case CCIPCA -> ccipca(c);
            case INCREMENTAL_SVD -> {
                pending.add(c);
                if (pending.size() >= block) {
                    flush();
                }
            }
            default -> throw new IllegalStateException();
        }
    }

    private void ccipca(double[] x) {
        double[] u = x.clone();
        for (int j = 0; j < k; j++) {
            double norm = 0;
            for (int i = 0; i < d; i++) {
                norm += w[i][j] * w[i][j];
            }
            norm = Math.sqrt(norm);
            if (norm < 1e-12) {
                // the first non-zero residual starts the component
                for (int i = 0; i < d; i++) {
                    w[i][j] = u[i];
                }
            } else {
                double dot = 0;
                for (int i = 0; i < d; i++) {
                    dot += u[i] * w[i][j];
                }
                double keep = Math.max(0, (n - 1 - amnesic) / n);
                double add = (1 + amnesic) / n;
                for (int i = 0; i < d; i++) {
                    w[i][j] = keep * w[i][j] + add * u[i] * dot / Math.max(norm, 1e-12);
                }
            }
            norm = 0;
            for (int i = 0; i < d; i++) {
                norm += w[i][j] * w[i][j];
            }
            norm = Math.sqrt(norm);
            double dot = 0;
            for (int i = 0; i < d; i++) {
                dot += u[i] * w[i][j] / Math.max(norm, 1e-12);
            }
            for (int i = 0; i < d; i++) {
                u[i] -= dot * w[i][j] / Math.max(norm, 1e-12);
            }
        }
    }

    /** Applies the pending block to the thin SVD. */
    public void flush() {
        if (pending.isEmpty() || kind != Kind.INCREMENTAL_SVD) {
            return;
        }
        int b = pending.size();
        int r = (int) Math.min(k, Math.max(0, n - b));
        // L = U^T B (r x b), H = B - U L (d x b)
        double[][] l = new double[r][b];
        double[][] h = new double[d][b];
        for (int c = 0; c < b; c++) {
            double[] col = pending.get(c);
            for (int j = 0; j < r; j++) {
                double dot = 0;
                for (int i = 0; i < d; i++) {
                    dot += w[i][j] * col[i];
                }
                l[j][c] = dot;
            }
            for (int i = 0; i < d; i++) {
                double v = col[i];
                for (int j = 0; j < r; j++) {
                    v -= w[i][j] * l[j][c];
                }
                h[i][c] = v;
            }
        }
        // QR of H by modified Gram-Schmidt: J (d x b), R (b x b)
        double[][] jm = new double[d][b];
        double[][] rm = new double[b][b];
        for (int c = 0; c < b; c++) {
            double[] v = new double[d];
            for (int i = 0; i < d; i++) {
                v[i] = h[i][c];
            }
            for (int p = 0; p < c; p++) {
                double dot = 0;
                for (int i = 0; i < d; i++) {
                    dot += jm[i][p] * v[i];
                }
                rm[p][c] = dot;
                for (int i = 0; i < d; i++) {
                    v[i] -= dot * jm[i][p];
                }
            }
            double norm = 0;
            for (double a : v) {
                norm += a * a;
            }
            norm = Math.sqrt(norm);
            rm[c][c] = norm;
            for (int i = 0; i < d; i++) {
                jm[i][c] = norm > 1e-12 ? v[i] / norm : 0;
            }
        }
        // K = [[f S, L], [0, R]] ((r + b) x (r + b))
        int m = r + b;
        double[][] kk = new double[m][m];
        for (int j = 0; j < r; j++) {
            kk[j][j] = forget * s[j];
            for (int c = 0; c < b; c++) {
                kk[j][r + c] = l[j][c];
            }
        }
        for (int p = 0; p < b; p++) {
            for (int c = 0; c < b; c++) {
                kk[r + p][r + c] = rm[p][c];
            }
        }
        ThinSvd svd = ThinSvd.of(kk);
        int keep = Math.min(k, svd.s().length);
        double[][] next = new double[d][k];
        double[] ns = new double[k];
        for (int j = 0; j < keep; j++) {
            ns[j] = svd.s()[j];
            for (int i = 0; i < d; i++) {
                double v = 0;
                for (int a = 0; a < r; a++) {
                    v += w[i][a] * svd.u()[a][j];
                }
                for (int c = 0; c < b; c++) {
                    v += jm[i][c] * svd.u()[r + c][j];
                }
                next[i][j] = v;
            }
        }
        w = next;
        s = ns;
        pending.clear();
    }

    /** @return the current orthonormal basis {@code [d][k]} */
    public double[][] basis() {
        if (kind == Kind.INCREMENTAL_SVD) {
            flush();
        }
        double[][] q = new double[d][];
        for (int i = 0; i < d; i++) {
            q[i] = w[i].clone();
        }
        orthonormalise(q);
        return q;
    }

    /** @return the points seen */
    public long count() {
        return n;
    }

    /**
     * Orthonormalises the columns of a matrix in place (modified Gram–Schmidt).
     *
     * @param m {@code [d][k]}
     */
    public static void orthonormalise(double[][] m) {
        int d = m.length;
        int k = m[0].length;
        for (int j = 0; j < k; j++) {
            for (int p = 0; p < j; p++) {
                double dot = 0;
                for (int i = 0; i < d; i++) {
                    dot += m[i][p] * m[i][j];
                }
                for (int i = 0; i < d; i++) {
                    m[i][j] -= dot * m[i][p];
                }
            }
            double norm = 0;
            for (int i = 0; i < d; i++) {
                norm += m[i][j] * m[i][j];
            }
            norm = Math.sqrt(norm);
            for (int i = 0; i < d; i++) {
                m[i][j] = norm > 1e-12 ? m[i][j] / norm : (i == j ? 1 : 0);
            }
        }
    }

    /**
     * The sine of the largest principal angle between two subspaces with orthonormal bases:
     * {@code sqrt(1 - σ_min(AᵀB)^2)}.
     *
     * @param a {@code [d][k]}
     * @param b {@code [d][k]}
     * @return {@code sin Θ_max} in {@code [0, 1]}
     */
    public static double sinTheta(double[][] a, double[][] b) {
        int k = a[0].length;
        double[][] m = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double s = 0;
                for (int r = 0; r < a.length; r++) {
                    s += a[r][i] * b[r][j];
                }
                m[i][j] = s;
            }
        }
        double[] sv = ThinSvd.of(m).s();
        double min = sv.length < k ? 0 : sv[k - 1];
        return Math.sqrt(Math.max(0, 1 - Math.min(1, min * min)));
    }
}
