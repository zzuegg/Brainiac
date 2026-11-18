package io.github.zzuegg.brainiac.experimental.sketch;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.SplittableRandom;

/**
 * Frequent Directions, the matrix analogue of Misra-Gries: a sketch {@code B} of {@code l}
 * rows of a stream of rows {@code A} such that {@code 0 <= A^T A - B^T B} and
 * {@code ||A^T A - B^T B||_2 <= ||A - A_k||_F^2 / (l - k)}. Rows go into a buffer of
 * {@code 2l}; when it is full, its SVD is taken, every squared singular value is reduced by
 * the {@code l}-th one and the bottom half of the rows becomes free.
 *
 * <p>{@link #learned} is the learning-augmented variant (after Aamand, Chen, Nguyen, Silwal
 * and Vakilian, ICLR 2025): {@code k} rows of memory are reserved for predicted top
 * directions {@code U} (orthonormal, {@code k x d}), where the sketch keeps
 * {@code C = sum_t a_t (U a_t)^T} exactly; that gives every covariance term but the one of the
 * residuals {@code Q a_t} ({@code Q = I - U^T U}), which a plain sketch of {@code l - k} rows
 * summarises. With good predictions the residuals carry little mass and the error falls;
 * with bad ones the plain bound on the residual stream still holds.
 *
 * <p>Paper: Liberty, "Simple and Deterministic Matrix Sketching" (KDD 2013); Ghashami,
 * Liberty, Phillips and Woodruff, "Frequent Directions: Simple and Deterministic Matrix
 * Sketching" (SICOMP 2016).
 */
public final class FrequentDirections {

    private final int ell;
    private final int d;
    private final double[][] buffer;
    private int filled;

    /**
     * Creates an empty sketch.
     *
     * @param ell sketch rows
     * @param d row width
     */
    public FrequentDirections(int ell, int d) {
        this.ell = ell;
        this.d = d;
        this.buffer = new double[2 * ell][d];
    }

    /**
     * Adds a row.
     *
     * @param row the row
     */
    public void add(double[] row) {
        if (filled == buffer.length) {
            shrink();
        }
        System.arraycopy(row, 0, buffer[filled++], 0, d);
    }

    private void shrink() {
        ThinSvd svd = ThinSvd.of(buffer);
        int k = svd.s().length;
        double delta = ell - 1 < k ? svd.s()[ell - 1] * svd.s()[ell - 1] : 0;
        for (double[] r : buffer) {
            java.util.Arrays.fill(r, 0);
        }
        int rows = 0;
        for (int i = 0; i < Math.min(ell, k); i++) {
            double s2 = svd.s()[i] * svd.s()[i] - delta;
            if (s2 <= 1e-12) {
                continue;
            }
            double s = Math.sqrt(s2);
            for (int j = 0; j < d; j++) {
                buffer[rows][j] = s * svd.v()[j][i];
            }
            rows++;
        }
        filled = rows;
    }

    /** @return the sketch {@code B} ({@code l x d}, after a final shrink) */
    public double[][] sketch() {
        if (filled > ell) {
            shrink();
        }
        double[][] b = new double[ell][];
        for (int i = 0; i < ell; i++) {
            b[i] = buffer[i].clone();
        }
        return b;
    }

    /** @return {@code B^T B} */
    public double[][] covariance() {
        return gram(sketch());
    }

    /** @return the memory in rows */
    public int rows() {
        return ell;
    }

    /**
     * The learning-augmented sketch.
     *
     * @param ell rows of memory in all
     * @param predicted the predicted top directions {@code [k][d]}, orthonormal, {@code k < ell}
     * @return the sketch
     */
    public static Learned learned(int ell, double[][] predicted) {
        return new Learned(ell, predicted);
    }

    /** Frequent Directions with reserved rows for predicted directions. */
    public static final class Learned {
        private final double[][] u;
        private final double[][] c;
        private final FrequentDirections residual;
        private final int ell;

        Learned(int ell, double[][] predicted) {
            this.ell = ell;
            this.u = predicted;
            int d = predicted[0].length;
            this.c = new double[d][predicted.length];
            this.residual = new FrequentDirections(ell - predicted.length, d);
        }

        /**
         * Adds a row.
         *
         * @param row the row
         */
        public void add(double[] row) {
            int k = u.length;
            int d = row.length;
            double[] proj = new double[k];
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < d; j++) {
                    proj[i] += u[i][j] * row[j];
                }
            }
            double[] q = row.clone();
            for (int j = 0; j < d; j++) {
                for (int i = 0; i < k; i++) {
                    c[j][i] += row[j] * proj[i];
                    q[j] -= proj[i] * u[i][j];
                }
            }
            residual.add(q);
        }

        /** @return the covariance estimate {@code C U + (Q C U)^T + B^T B} */
        public double[][] covariance() {
            int d = c.length;
            int k = u.length;
            double[][] cu = new double[d][d];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    for (int i = 0; i < k; i++) {
                        cu[a][b] += c[a][i] * u[i][b];
                    }
                }
            }
            double[][] qc = new double[d][k];
            for (int a = 0; a < d; a++) {
                for (int i = 0; i < k; i++) {
                    double s = c[a][i];
                    for (int j = 0; j < k; j++) {
                        double dot = 0;
                        for (int b = 0; b < d; b++) {
                            dot += u[j][a] * u[j][b] * c[b][i];
                        }
                        s -= dot;
                    }
                    qc[a][i] = s;
                }
            }
            double[][] out = residual.covariance();
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    double cross = 0;
                    for (int i = 0; i < k; i++) {
                        cross += qc[b][i] * u[i][a];
                    }
                    out[a][b] += cu[a][b] + cross;
                }
            }
            return out;
        }

        /** @return the memory in rows */
        public int rows() {
            return ell;
        }
    }

    /**
     * {@code M^T M}.
     *
     * @param m a matrix {@code [n][d]}
     * @return its Gram matrix
     */
    public static double[][] gram(double[][] m) {
        int d = m[0].length;
        double[][] g = new double[d][d];
        for (double[] r : m) {
            for (int a = 0; a < d; a++) {
                if (r[a] == 0) {
                    continue;
                }
                for (int b = 0; b < d; b++) {
                    g[a][b] += r[a] * r[b];
                }
            }
        }
        return g;
    }

    /**
     * The relative covariance error {@code ||A^T A - E||_2 / ||A||_F^2}, the spectral norm by
     * power iteration on the symmetric difference.
     *
     * @param gramA {@code A^T A}
     * @param estimate the estimate {@code E}
     * @return the relative error
     */
    public static double covarianceError(double[][] gramA, double[][] estimate) {
        int d = gramA.length;
        double[][] diff = new double[d][d];
        double fro = 0;
        for (int a = 0; a < d; a++) {
            fro += gramA[a][a];
            for (int b = 0; b < d; b++) {
                diff[a][b] = gramA[a][b] - estimate[a][b];
            }
        }
        return spectralNorm(diff) / fro;
    }

    private static double spectralNorm(double[][] sym) {
        int d = sym.length;
        double[] x = new double[d];
        SplittableRandom r = new SplittableRandom(7);
        for (int i = 0; i < d; i++) {
            x[i] = r.nextGaussian();
        }
        double lambda = 0;
        for (int it = 0; it < 200; it++) {
            double[] y = new double[d];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    y[a] += sym[a][b] * x[b];
                }
            }
            double n = 0;
            for (double v : y) {
                n += v * v;
            }
            n = Math.sqrt(n);
            if (n < 1e-300) {
                return 0;
            }
            for (int a = 0; a < d; a++) {
                x[a] = y[a] / n;
            }
            lambda = n;
        }
        return lambda;
    }

    /**
     * Uniform row sampling: {@code l} rows drawn by reservoir sampling, scaled by
     * {@code sqrt(n / l)} so that {@code B^T B} is unbiased.
     *
     * @param rows the stream
     * @param ell rows kept
     * @param seed the seed
     * @return {@code B^T B}
     */
    public static double[][] rowSampling(double[][] rows, int ell, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] keep = new double[ell][];
        for (int t = 0; t < rows.length; t++) {
            if (t < ell) {
                keep[t] = rows[t];
            } else {
                int j = r.nextInt(t + 1);
                if (j < ell) {
                    keep[j] = rows[t];
                }
            }
        }
        double scale = rows.length / (double) Math.min(ell, rows.length);
        double[][] g = gram(java.util.Arrays.stream(keep).filter(java.util.Objects::nonNull).toArray(double[][]::new));
        for (double[] gr : g) {
            for (int b = 0; b < gr.length; b++) {
                gr[b] *= scale;
            }
        }
        return g;
    }

    /**
     * Oja's rule, a streaming estimate of the top principal direction:
     * {@code w += eta y (x - y w)} with {@code y = w . x}; returns the rank-one covariance
     * estimate {@code lambda w w^T} with {@code lambda = sum (w . x)^2} over a second pass.
     *
     * @param rows the stream
     * @param eta the learning rate
     * @param seed the seed
     * @return the rank-one covariance estimate
     */
    public static double[][] oja(double[][] rows, double eta, long seed) {
        int d = rows[0].length;
        double[] w = new double[d];
        SplittableRandom r = new SplittableRandom(seed);
        double n0 = 0;
        for (int j = 0; j < d; j++) {
            w[j] = r.nextGaussian();
            n0 += w[j] * w[j];
        }
        for (int j = 0; j < d; j++) {
            w[j] /= Math.sqrt(n0);
        }
        for (int t = 0; t < rows.length; t++) {
            double[] x = rows[t];
            double y = 0;
            for (int j = 0; j < d; j++) {
                y += w[j] * x[j];
            }
            double rate = eta / (1 + t * eta * 0.01);
            for (int j = 0; j < d; j++) {
                w[j] += rate * y * (x[j] - y * w[j]);
            }
        }
        double norm = 0;
        for (double v : w) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        double lambda = 0;
        for (double[] x : rows) {
            double y = 0;
            for (int j = 0; j < d; j++) {
                y += w[j] / norm * x[j];
            }
            lambda += y * y;
        }
        double[][] g = new double[d][d];
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                g[a][b] = lambda * w[a] * w[b] / (norm * norm);
            }
        }
        return g;
    }

    /**
     * The best rank-{@code k} covariance {@code V_k S_k^2 V_k^T} from the full SVD (the
     * offline oracle).
     *
     * @param rows the matrix
     * @param k the rank
     * @return the covariance estimate
     */
    public static double[][] svdOracle(double[][] rows, int k) {
        ThinSvd svd = ThinSvd.of(rows);
        int d = rows[0].length;
        double[][] g = new double[d][d];
        for (int i = 0; i < Math.min(k, svd.s().length); i++) {
            double s2 = svd.s()[i] * svd.s()[i];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    g[a][b] += s2 * svd.v()[a][i] * svd.v()[b][i];
                }
            }
        }
        return g;
    }

    /**
     * The top {@code k} right singular vectors of a matrix, as rows (for predictions).
     *
     * @param rows the matrix
     * @param k how many
     * @return {@code [k][d]}, orthonormal
     */
    public static double[][] topDirections(double[][] rows, int k) {
        ThinSvd svd = ThinSvd.of(rows);
        int d = rows[0].length;
        double[][] u = new double[k][d];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                u[i][j] = svd.v()[j][i];
            }
        }
        return u;
    }
}
