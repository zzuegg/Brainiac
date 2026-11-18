package io.github.zzuegg.brainiac.experimental.signal;

import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;

import java.util.SplittableRandom;

/**
 * Compressed-sensing recovery of a sparse {@code x} from {@code y = A x + noise} with
 * {@code m < n} measurements.
 *
 * <ul>
 *   <li>{@link Method#OMP}: orthogonal matching pursuit, greedily adding the column most
 *       correlated with the residual and refitting by least squares (Pati et al. 1993).</li>
 *   <li>{@link Method#ISTA} and {@link Method#FISTA}: proximal gradient on the lasso
 *       {@code 1/2 ||y - Ax||^2 + lambda ||x||_1} with step {@code 1 / ||A||_2^2}, FISTA with
 *       Nesterov momentum (Daubechies et al. 2004; Beck and Teboulle 2009).</li>
 *   <li>{@link Method#AMP}: approximate message passing with soft thresholding at
 *       {@code alpha ||z|| / sqrt(m)} and the Onsager correction
 *       {@code z_t = y - A x_t + z_(t-1) ||x_t||_0 / m}, which keeps the effective noise Gaussian
 *       (Donoho, Maleki and Montanari, PNAS 2009,
 *       <a href="https://arxiv.org/abs/0907.3574">arXiv:0907.3574</a>).</li>
 *   <li>{@link Method#MIN_NORM_LS}: the minimum-norm least-squares solution, which never
 *       recovers a sparse vector.</li>
 * </ul>
 *
 * <p>{@link #phaseTransition} sweeps the Donoho-Tanner diagram over sampling ratio
 * {@code m/n} and sparsity ratio {@code s/m}.
 */
public final class SparseRecovery {

    /** A recovery method. */
    public enum Method {
        /** orthogonal matching pursuit. */
        OMP,
        /** iterative soft thresholding. */
        ISTA,
        /** fast (accelerated) ISTA. */
        FISTA,
        /** approximate message passing. */
        AMP,
        /** minimum-norm least squares. */
        MIN_NORM_LS
    }

    private SparseRecovery() {
    }

    /**
     * A Gaussian measurement matrix with entries of variance {@code 1/m} (unit-norm columns in
     * expectation).
     *
     * @param m rows
     * @param n columns
     * @param rnd the random source
     * @return the matrix
     */
    public static double[][] gaussianMatrix(int m, int n, SplittableRandom rnd) {
        double[][] a = new double[m][n];
        double s = 1 / Math.sqrt(m);
        for (double[] row : a) {
            for (int j = 0; j < n; j++) {
                row[j] = s * ChannelCodes.gaussian(rnd);
            }
        }
        return a;
    }

    /**
     * A random {@code s}-sparse vector with Gaussian non-zeros.
     *
     * @param n length
     * @param s non-zeros
     * @param rnd the random source
     * @return the vector
     */
    public static double[] sparseVector(int n, int s, SplittableRandom rnd) {
        double[] x = new double[n];
        int placed = 0;
        while (placed < s) {
            int i = rnd.nextInt(n);
            if (x[i] == 0) {
                x[i] = ChannelCodes.gaussian(rnd);
                placed++;
            }
        }
        return x;
    }

    /**
     * The largest squared singular value of {@code A} by power iteration (the Lipschitz constant
     * of the least-squares gradient).
     *
     * @param a the matrix
     * @return {@code ||A||_2^2}
     */
    public static double lipschitz(double[][] a) {
        int n = a[0].length;
        double[] v = new double[n];
        java.util.Arrays.fill(v, 1 / Math.sqrt(n));
        double l = 1;
        for (int it = 0; it < 50; it++) {
            double[] w = DenseMatrices.multiplyTransposed(a, DenseMatrices.multiply(a, v));
            l = DenseMatrices.norm(w);
            for (int j = 0; j < n; j++) {
                v[j] = w[j] / l;
            }
        }
        return l;
    }

    /**
     * Soft thresholding.
     *
     * @param u the input
     * @param t the threshold
     * @return {@code sign(u) max(|u| - t, 0)}
     */
    public static double soft(double u, double t) {
        return u > t ? u - t : u < -t ? u + t : 0;
    }

    /**
     * Recovers {@code x}.
     *
     * @param method the method
     * @param a the measurement matrix {@code [m][n]}
     * @param y the measurements
     * @param sparsity OMP's number of atoms (ignored by the others)
     * @param lambda the lasso penalty (ISTA/FISTA) or AMP's threshold multiplier {@code alpha}
     * @param iterations iterations (ISTA/FISTA/AMP)
     * @param truth the true vector for the NMSE trace, or {@code null}
     * @param trace receives the NMSE after each iteration (length at least {@code iterations}),
     *     or {@code null}
     * @return the estimate
     */
    public static double[] solve(Method method, double[][] a, double[] y, int sparsity, double lambda, int iterations,
            double[] truth, double[] trace) {
        return switch (method) {
            case OMP -> omp(a, y, sparsity, 1e-9, truth, trace);
            case ISTA -> ista(a, y, lambda, iterations, false, truth, trace);
            case FISTA -> ista(a, y, lambda, iterations, true, truth, trace);
            case AMP -> amp(a, y, lambda, iterations, truth, trace);
            case MIN_NORM_LS -> minNorm(a, y);
        };
    }

    /**
     * Orthogonal matching pursuit.
     *
     * @param a the matrix
     * @param y the measurements
     * @param maxAtoms the largest support
     * @param tolerance stop when the residual norm falls below this
     * @param truth the true vector for the trace, or {@code null}
     * @param trace NMSE after each added atom, or {@code null}
     * @return the estimate
     */
    public static double[] omp(double[][] a, double[] y, int maxAtoms, double tolerance, double[] truth,
            double[] trace) {
        int m = a.length;
        int n = a[0].length;
        double[] colNorm = new double[n];
        for (double[] row : a) {
            for (int j = 0; j < n; j++) {
                colNorm[j] += row[j] * row[j];
            }
        }
        int[] support = new int[maxAtoms];
        boolean[] used = new boolean[n];
        double[] residual = y.clone();
        double[] x = new double[n];
        int k = 0;
        while (k < Math.min(maxAtoms, m) && DenseMatrices.norm(residual) > tolerance) {
            double[] corr = DenseMatrices.multiplyTransposed(a, residual);
            int best = -1;
            double bestV = -1;
            for (int j = 0; j < n; j++) {
                double v = Math.abs(corr[j]) / Math.sqrt(colNorm[j] + 1e-30);
                if (!used[j] && v > bestV) {
                    bestV = v;
                    best = j;
                }
            }
            support[k++] = best;
            used[best] = true;
            double[] coef = DenseMatrices.leastSquares(a, y, support, k);
            java.util.Arrays.fill(x, 0);
            for (int i = 0; i < k; i++) {
                x[support[i]] = coef[i];
            }
            double[] ax = DenseMatrices.multiply(a, x);
            for (int i = 0; i < m; i++) {
                residual[i] = y[i] - ax[i];
            }
            if (trace != null && k - 1 < trace.length) {
                trace[k - 1] = nmse(x, truth);
            }
        }
        if (trace != null) {
            for (int i = Math.max(k, 0); i < trace.length; i++) {
                trace[i] = nmse(x, truth);
            }
        }
        return x;
    }

    private static double[] ista(double[][] a, double[] y, double lambda, int iterations, boolean fast,
            double[] truth, double[] trace) {
        int n = a[0].length;
        double l = lipschitz(a);
        double[] x = new double[n];
        double[] z = new double[n];
        double t = 1;
        for (int it = 0; it < iterations; it++) {
            double[] ax = DenseMatrices.multiply(a, z);
            for (int i = 0; i < ax.length; i++) {
                ax[i] -= y[i];
            }
            double[] g = DenseMatrices.multiplyTransposed(a, ax);
            double[] next = new double[n];
            for (int j = 0; j < n; j++) {
                next[j] = soft(z[j] - g[j] / l, lambda / l);
            }
            if (fast) {
                double tNext = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
                for (int j = 0; j < n; j++) {
                    z[j] = next[j] + (t - 1) / tNext * (next[j] - x[j]);
                }
                t = tNext;
            } else {
                z = next;
            }
            x = next;
            if (trace != null && it < trace.length) {
                trace[it] = nmse(x, truth);
            }
        }
        return x;
    }

    private static double[] amp(double[][] a, double[] y, double alpha, int iterations, double[] truth,
            double[] trace) {
        int m = a.length;
        int n = a[0].length;
        double[] x = new double[n];
        double[] z = y.clone();
        for (int it = 0; it < iterations; it++) {
            double tau = alpha * DenseMatrices.norm(z) / Math.sqrt(m);
            double[] pseudo = DenseMatrices.multiplyTransposed(a, z);
            int active = 0;
            for (int j = 0; j < n; j++) {
                x[j] = soft(x[j] + pseudo[j], tau);
                active += x[j] != 0 ? 1 : 0;
            }
            double[] ax = DenseMatrices.multiply(a, x);
            double onsager = active / (double) m;
            for (int i = 0; i < m; i++) {
                z[i] = y[i] - ax[i] + onsager * z[i];
            }
            if (trace != null && it < trace.length) {
                trace[it] = nmse(x, truth);
            }
        }
        return x;
    }

    /**
     * The minimum-norm least-squares solution {@code A^T (A A^T)^-1 y}.
     *
     * @param a the matrix (full row rank)
     * @param y the measurements
     * @return the estimate
     */
    public static double[] minNorm(double[][] a, double[] y) {
        double[][] aat = DenseMatrices.gram(DenseMatrices.transpose(a));
        return DenseMatrices.multiplyTransposed(a, DenseMatrices.solveSpd(aat, y, 1e-10));
    }

    /**
     * The normalised mean squared error {@code ||x - truth||^2 / ||truth||^2}.
     *
     * @param x the estimate
     * @param truth the truth
     * @return the NMSE, or NaN without a truth
     */
    public static double nmse(double[] x, double[] truth) {
        if (truth == null) {
            return Double.NaN;
        }
        double e = 0;
        double t = 0;
        for (int i = 0; i < x.length; i++) {
            e += (x[i] - truth[i]) * (x[i] - truth[i]);
            t += truth[i] * truth[i];
        }
        return e / Math.max(t, 1e-30);
    }

    /**
     * The empirical Donoho-Tanner phase diagram: the probability that a method recovers the
     * vector to NMSE below {@code 1e-3}, noiseless.
     *
     * @param method the method (OMP gets the true sparsity; ISTA/FISTA use
     *     {@code lambda = 1e-3 max|A^T y|}; AMP uses {@code alpha = 1.5})
     * @param n the signal length
     * @param deltas sampling ratios {@code m/n}
     * @param rhos sparsity ratios {@code s/m}
     * @param trials trials per cell
     * @param iterations iterations of the iterative methods
     * @param seed the seed
     * @return {@code [delta][rho]} success probabilities
     */
    public static double[][] phaseTransition(Method method, int n, double[] deltas, double[] rhos, int trials,
            int iterations, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] out = new double[deltas.length][rhos.length];
        for (int d = 0; d < deltas.length; d++) {
            int m = (int) Math.round(deltas[d] * n);
            for (int r = 0; r < rhos.length; r++) {
                int s = Math.max(1, (int) Math.round(rhos[r] * m));
                for (int t = 0; t < trials; t++) {
                    double[][] a = gaussianMatrix(m, n, rnd);
                    double[] x = sparseVector(n, s, rnd);
                    double[] y = DenseMatrices.multiply(a, x);
                    double lambda = method == Method.AMP ? 1.5 : 1e-3 * maxAbs(DenseMatrices.multiplyTransposed(a, y));
                    double[] est = solve(method, a, y, s, lambda, iterations, null, null);
                    out[d][r] += nmse(est, x) < 1e-3 ? 1.0 / trials : 0;
                }
            }
        }
        return out;
    }

    /**
     * The largest absolute entry.
     *
     * @param v a vector
     * @return {@code max |v_i|}
     */
    public static double maxAbs(double[] v) {
        double m = 0;
        for (double x : v) {
            m = Math.max(m, Math.abs(x));
        }
        return m;
    }
}
