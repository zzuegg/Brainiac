package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * Kernel ridge regression at scale: the Nyström approximation and FALKON.
 *
 * <p>Exact kernel ridge regression solves {@code (K + n lambda I) a = y}, {@code O(n^3)} time
 * and {@code O(n^2)} memory. Nyström (Williams and Seeger 2001,
 * <a href="https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines">
 * NeurIPS</a>) restricts the solution to {@code m << n} landmark points, which turns it into
 * {@code (K_nm^T K_nm + n lambda K_mm) a = K_nm^T y}: {@code O(n m^2)} time.
 * FALKON (Rudi et al. 2017, <a href="https://arxiv.org/abs/1705.10958">arXiv:1705.10958</a>)
 * solves that same system by conjugate gradient, never forming {@code K_nm^T K_nm}: each
 * iteration costs two passes over {@code K_nm} ({@code O(n m)}), and the preconditioner
 * {@code P = n/m K_mm^2 + n lambda K_mm}, factored from the landmarks alone, makes a handful of
 * iterations enough, so it reaches the optimal statistical rate in {@code O(n m t)} time.
 */
public final class NystromKernelRidge {

    private final KernelFunction kernel;
    private final double[][] centres;
    private final double[] alpha;
    private final double yMean;

    private NystromKernelRidge(KernelFunction kernel, double[][] centres, double[] alpha, double yMean) {
        this.kernel = kernel;
        this.centres = centres;
        this.alpha = alpha;
        this.yMean = yMean;
    }

    /**
     * Predicts at new points.
     *
     * @param xs the points {@code [k][d]}
     * @return the predictions
     */
    public double[] predict(double[][] xs) {
        double[] out = new double[xs.length];
        for (int j = 0; j < xs.length; j++) {
            double s = yMean;
            for (int i = 0; i < centres.length; i++) {
                s += kernel.value(centres[i], xs[j]) * alpha[i];
            }
            out[j] = s;
        }
        return out;
    }

    /**
     * The number of centres (landmarks, or all training points for exact KRR).
     *
     * @return the centre count
     */
    public int centres() {
        return centres.length;
    }

    private static double mean(double[] y) {
        double m = 0;
        for (double v : y) {
            m += v / y.length;
        }
        return m;
    }

    /**
     * Exact kernel ridge regression.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets
     * @param kernel the kernel
     * @param lambda the ridge per point
     * @return the model
     */
    public static NystromKernelRidge exact(double[][] x, double[] y, KernelFunction kernel, double lambda) {
        double m = mean(y);
        double[] yc = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            yc[i] = y[i] - m;
        }
        double[][] k = kernel.gram(x);
        for (int i = 0; i < k.length; i++) {
            k[i][i] += x.length * lambda;
        }
        return new NystromKernelRidge(kernel, x, MlMatrices.choleskySolve(MlMatrices.cholesky(k), yc), m);
    }

    /**
     * Uniformly sampled landmarks.
     *
     * @param x the data
     * @param m the number of landmarks
     * @param seed the seed
     * @return {@code m} distinct rows of {@code x}
     */
    public static double[][] landmarks(double[][] x, int m, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] idx = new int[x.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        double[][] c = new double[m][];
        for (int i = 0; i < m; i++) {
            int j = i + r.nextInt(x.length - i);
            int t = idx[i];
            idx[i] = idx[j];
            idx[j] = t;
            c[i] = x[idx[i]];
        }
        return c;
    }

    /**
     * Nyström KRR solved directly.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets
     * @param kernel the kernel
     * @param m the number of landmarks
     * @param lambda the ridge per point
     * @param seed the landmark seed
     * @return the model
     */
    public static NystromKernelRidge nystrom(double[][] x, double[] y, KernelFunction kernel, int m, double lambda,
            long seed) {
        int n = x.length;
        double[][] c = landmarks(x, m, seed);
        double ym = mean(y);
        double[][] knm = kernel.matrix(x, c);
        double[][] h = kernel.gram(c);
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                h[a][b] *= n * lambda;
            }
            h[a][a] += 1e-8 * n;
        }
        double[] rhs = new double[m];
        for (int i = 0; i < n; i++) {
            double[] row = knm[i];
            double yi = y[i] - ym;
            for (int a = 0; a < m; a++) {
                rhs[a] += row[a] * yi;
                double ra = row[a];
                double[] ha = h[a];
                for (int b = 0; b <= a; b++) {
                    ha[b] += ra * row[b];
                }
            }
        }
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < a; b++) {
                h[b][a] = h[a][b];
            }
        }
        return new NystromKernelRidge(kernel, c, MlMatrices.choleskySolve(MlMatrices.cholesky(h), rhs), ym);
    }

    /**
     * FALKON: Nyström KRR by preconditioned conjugate gradient.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets
     * @param kernel the kernel
     * @param m the number of landmarks
     * @param lambda the ridge per point
     * @param iterations CG iterations, for example 10
     * @param seed the landmark seed
     * @return the model
     */
    public static NystromKernelRidge falkon(double[][] x, double[] y, KernelFunction kernel, int m, double lambda,
            int iterations, long seed) {
        int n = x.length;
        double[][] c = landmarks(x, m, seed);
        double ym = mean(y);
        double[][] kmm = kernel.gram(c);
        for (int a = 0; a < m; a++) {
            kmm[a][a] += 1e-8 * m;
        }
        double[][] lk = MlMatrices.cholesky(kmm);
        double[][] inner = new double[m][m];
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                inner[a][b] = (double) n / m * kmm[a][b];
            }
            inner[a][a] += n * lambda;
        }
        double[][] li = MlMatrices.cholesky(inner);
        float[][] knm = new float[n][m];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < m; a++) {
                knm[i][a] = (float) kernel.value(x[i], c[a]);
            }
        }
        double[] b = new double[m];
        for (int i = 0; i < n; i++) {
            double yi = y[i] - ym;
            for (int a = 0; a < m; a++) {
                b[a] += knm[i][a] * yi;
            }
        }
        double[] alpha = new double[m];
        double[] r = b.clone();
        double[] z = precondition(lk, li, r);
        double[] p = z.clone();
        double rz = MlMatrices.dot(r, z);
        for (int it = 0; it < iterations && rz > 1e-30; it++) {
            double[] hp = apply(knm, kmm, n * lambda, p);
            double step = rz / MlMatrices.dot(p, hp);
            for (int a = 0; a < m; a++) {
                alpha[a] += step * p[a];
                r[a] -= step * hp[a];
            }
            z = precondition(lk, li, r);
            double next = MlMatrices.dot(r, z);
            for (int a = 0; a < m; a++) {
                p[a] = z[a] + next / rz * p[a];
            }
            rz = next;
        }
        return new NystromKernelRidge(kernel, c, alpha, ym);
    }

    /** {@code (K_nm^T K_nm + reg K_mm) v} in two passes over {@code K_nm}. */
    private static double[] apply(float[][] knm, double[][] kmm, double reg, double[] v) {
        int m = v.length;
        double[] out = new double[m];
        for (float[] row : knm) {
            double s = 0;
            for (int a = 0; a < m; a++) {
                s += row[a] * v[a];
            }
            for (int a = 0; a < m; a++) {
                out[a] += row[a] * s;
            }
        }
        for (int a = 0; a < m; a++) {
            out[a] += reg * MlMatrices.dot(kmm[a], v);
        }
        return out;
    }

    /** {@code P^-1 v} for {@code P = K_mm (n/m K_mm + n lambda I)}. */
    private static double[] precondition(double[][] lk, double[][] li, double[] v) {
        return MlMatrices.choleskySolve(li, MlMatrices.choleskySolve(lk, v));
    }
}
