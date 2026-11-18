package io.github.zzuegg.brainiac.experimental.signal;

import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Dictionary learning for sparse coding, and patch denoising with the learned dictionary.
 *
 * <ul>
 *   <li>{@link Method#K_SVD}: alternate OMP sparse coding of all signals with a rank-1 update of
 *       each atom (and its coefficients) on the residual of the signals that use it (Aharon,
 *       Elad and Bruckstein, IEEE TSP 2006).</li>
 *   <li>{@link Method#ONLINE_DL}: Mairal et al.'s online dictionary learning: OMP-code a
 *       mini-batch, accumulate the sufficient statistics {@code A += a a^T}, {@code B += x a^T}
 *       (with forgetting), and update the atoms by one pass of block coordinate descent
 *       (ICML 2009, JMLR 2010, <a href="https://arxiv.org/abs/0908.0050">arXiv:0908.0050</a>).</li>
 * </ul>
 *
 * <p>{@link #denoise} codes every overlapping {@code p x p} patch of an image (mean removed) by
 * OMP with an error target {@code 1.15 sigma p}, and averages the overlapping reconstructions.
 * {@link #overcompleteDct} gives the fixed baseline dictionary.
 */
public final class DictionaryLearning {

    /** A dictionary learning algorithm. */
    public enum Method {
        /** K-SVD. */
        K_SVD,
        /** online dictionary learning. */
        ONLINE_DL
    }

    private DictionaryLearning() {
    }

    /**
     * Learns a dictionary.
     *
     * @param method the algorithm
     * @param signals training signals {@code [count][dim]}
     * @param atoms the dictionary size
     * @param sparsity OMP atoms per signal
     * @param iterations K-SVD iterations, or online passes over the data
     * @param seed the seed
     * @return the dictionary {@code [dim][atoms]} with unit-norm columns
     */
    public static double[][] learn(Method method, double[][] signals, int atoms, int sparsity, int iterations,
            long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int dim = signals[0].length;
        double[][] d = new double[dim][atoms];
        // initialise from random training signals
        for (int k = 0; k < atoms; k++) {
            double[] s = signals[rnd.nextInt(signals.length)];
            for (int i = 0; i < dim; i++) {
                d[i][k] = s[i] + 0.01 * ChannelCodes.gaussian(rnd);
            }
        }
        normalise(d);
        if (method == Method.K_SVD) {
            for (int it = 0; it < iterations; it++) {
                kSvdStep(d, signals, sparsity, rnd);
            }
        } else {
            online(d, signals, sparsity, iterations, rnd);
        }
        return d;
    }

    private static void normalise(double[][] d) {
        int atoms = d[0].length;
        for (int k = 0; k < atoms; k++) {
            double norm = 0;
            for (double[] row : d) {
                norm += row[k] * row[k];
            }
            norm = Math.sqrt(Math.max(norm, 1e-20));
            for (double[] row : d) {
                row[k] /= norm;
            }
        }
    }

    /**
     * Sparse codes of signals by OMP.
     *
     * @param d the dictionary
     * @param signals the signals
     * @param sparsity atoms per signal
     * @param tolerance residual-norm stop
     * @return the codes {@code [count][atoms]}
     */
    public static double[][] code(double[][] d, double[][] signals, int sparsity, double tolerance) {
        double[][] out = new double[signals.length][];
        for (int i = 0; i < signals.length; i++) {
            out[i] = SparseRecovery.omp(d, signals[i], sparsity, tolerance, null, null);
        }
        return out;
    }

    private static void kSvdStep(double[][] d, double[][] signals, int sparsity, SplittableRandom rnd) {
        int dim = d.length;
        int atoms = d[0].length;
        double[][] codes = code(d, signals, sparsity, 1e-6);
        for (int k = 0; k < atoms; k++) {
            List<Integer> users = new ArrayList<>();
            for (int i = 0; i < signals.length; i++) {
                if (codes[i][k] != 0) {
                    users.add(i);
                }
            }
            if (users.isEmpty()) {
                // replace an unused atom by a random signal
                double[] s = signals[rnd.nextInt(signals.length)];
                double norm = DenseMatrices.norm(s) + 1e-12;
                for (int r = 0; r < dim; r++) {
                    d[r][k] = s[r] / norm;
                }
                continue;
            }
            // residual without atom k on its users: E [dim][users]
            double[][] e = new double[dim][users.size()];
            for (int u = 0; u < users.size(); u++) {
                double[] x = codes[users.get(u)];
                double[] s = signals[users.get(u)];
                for (int r = 0; r < dim; r++) {
                    double v = s[r];
                    for (int j = 0; j < atoms; j++) {
                        if (x[j] != 0 && j != k) {
                            v -= d[r][j] * x[j];
                        }
                    }
                    e[r][u] = v;
                }
            }
            // rank-1 approximation by power iteration, started from the current atom
            double[] atom = new double[dim];
            for (int r = 0; r < dim; r++) {
                atom[r] = d[r][k];
            }
            double[] coef = new double[users.size()];
            for (int it = 0; it < 10; it++) {
                for (int u = 0; u < coef.length; u++) {
                    double c = 0;
                    for (int r = 0; r < dim; r++) {
                        c += e[r][u] * atom[r];
                    }
                    coef[u] = c;
                }
                double[] next = new double[dim];
                for (int r = 0; r < dim; r++) {
                    double v = 0;
                    for (int u = 0; u < coef.length; u++) {
                        v += e[r][u] * coef[u];
                    }
                    next[r] = v;
                }
                double norm = DenseMatrices.norm(next) + 1e-20;
                for (int r = 0; r < dim; r++) {
                    atom[r] = next[r] / norm;
                }
            }
            for (int u = 0; u < coef.length; u++) {
                double c = 0;
                for (int r = 0; r < dim; r++) {
                    c += e[r][u] * atom[r];
                }
                codes[users.get(u)][k] = c;
            }
            for (int r = 0; r < dim; r++) {
                d[r][k] = atom[r];
            }
        }
    }

    private static void online(double[][] d, double[][] signals, int sparsity, int passes, SplittableRandom rnd) {
        int dim = d.length;
        int atoms = d[0].length;
        double[][] aStat = new double[atoms][atoms];
        double[][] bStat = new double[dim][atoms];
        int batch = 64;
        int steps = passes * Math.max(1, signals.length / batch);
        for (int t = 1; t <= steps; t++) {
            double beta = Math.pow(1 - 1.0 / t, 1);
            for (double[] row : aStat) {
                for (int j = 0; j < atoms; j++) {
                    row[j] *= beta;
                }
            }
            for (double[] row : bStat) {
                for (int j = 0; j < atoms; j++) {
                    row[j] *= beta;
                }
            }
            for (int b = 0; b < batch; b++) {
                double[] s = signals[rnd.nextInt(signals.length)];
                double[] a = SparseRecovery.omp(d, s, sparsity, 1e-6, null, null);
                for (int i = 0; i < atoms; i++) {
                    if (a[i] == 0) {
                        continue;
                    }
                    for (int j = 0; j < atoms; j++) {
                        aStat[i][j] += a[i] * a[j] / batch;
                    }
                    for (int r = 0; r < dim; r++) {
                        bStat[r][i] += s[r] * a[i] / batch;
                    }
                }
            }
            // block coordinate descent on the atoms
            for (int j = 0; j < atoms; j++) {
                if (aStat[j][j] < 1e-10) {
                    double[] s = signals[rnd.nextInt(signals.length)];
                    double norm = DenseMatrices.norm(s) + 1e-12;
                    for (int r = 0; r < dim; r++) {
                        d[r][j] = s[r] / norm;
                    }
                    continue;
                }
                double[] u = new double[dim];
                for (int r = 0; r < dim; r++) {
                    double da = 0;
                    for (int i = 0; i < atoms; i++) {
                        da += d[r][i] * aStat[i][j];
                    }
                    u[r] = (bStat[r][j] - da) / aStat[j][j] + d[r][j];
                }
                double norm = Math.max(1, DenseMatrices.norm(u));
                for (int r = 0; r < dim; r++) {
                    d[r][j] = u[r] / norm;
                }
            }
        }
        normalise(d);
    }

    /**
     * The overcomplete 2-D DCT dictionary: the Kronecker product of a 1-D DCT with
     * {@code ceil(sqrt(atoms))} frequencies over {@code side} samples (mean-free atoms except
     * the DC atom), unit-norm columns.
     *
     * @param side the patch side
     * @param perAxis frequencies per axis
     * @return {@code [side^2][perAxis^2]}
     */
    public static double[][] overcompleteDct(int side, int perAxis) {
        double[][] one = new double[side][perAxis];
        for (int k = 0; k < perAxis; k++) {
            double mean = 0;
            for (int i = 0; i < side; i++) {
                one[i][k] = Math.cos(i * k * Math.PI / perAxis);
                mean += one[i][k] / side;
            }
            if (k > 0) {
                for (int i = 0; i < side; i++) {
                    one[i][k] -= mean;
                }
            }
        }
        double[][] d = new double[side * side][perAxis * perAxis];
        for (int a = 0; a < perAxis; a++) {
            for (int b = 0; b < perAxis; b++) {
                for (int i = 0; i < side; i++) {
                    for (int j = 0; j < side; j++) {
                        d[i * side + j][a * perAxis + b] = one[i][a] * one[j][b];
                    }
                }
            }
        }
        normalise(d);
        return d;
    }

    /**
     * Extracts patches (mean removed).
     *
     * @param image {@code [h][w]}
     * @param side the patch side
     * @param stride the stride
     * @return the patches
     */
    public static double[][] patches(double[][] image, int side, int stride) {
        List<double[]> out = new ArrayList<>();
        for (int r = 0; r + side <= image.length; r += stride) {
            for (int c = 0; c + side <= image[0].length; c += stride) {
                double[] p = new double[side * side];
                double mean = 0;
                for (int i = 0; i < side; i++) {
                    for (int j = 0; j < side; j++) {
                        p[i * side + j] = image[r + i][c + j];
                        mean += p[i * side + j] / (side * side);
                    }
                }
                for (int i = 0; i < p.length; i++) {
                    p[i] -= mean;
                }
                out.add(p);
            }
        }
        return out.toArray(new double[0][]);
    }

    /**
     * Denoises an image by sparse coding all overlapping patches with an error target and
     * averaging the reconstructions.
     *
     * @param noisy the noisy image
     * @param d the dictionary {@code [side^2][atoms]}
     * @param side the patch side
     * @param sigma the noise standard deviation
     * @param maxAtoms the largest number of atoms per patch
     * @return the denoised image
     */
    public static double[][] denoise(double[][] noisy, double[][] d, int side, double sigma, int maxAtoms) {
        int h = noisy.length;
        int w = noisy[0].length;
        double[][] sum = new double[h][w];
        double[][] count = new double[h][w];
        double target = 1.15 * sigma * side;
        for (int r = 0; r + side <= h; r++) {
            for (int c = 0; c + side <= w; c++) {
                double[] p = new double[side * side];
                double mean = 0;
                for (int i = 0; i < side; i++) {
                    for (int j = 0; j < side; j++) {
                        p[i * side + j] = noisy[r + i][c + j];
                        mean += p[i * side + j] / (side * side);
                    }
                }
                for (int i = 0; i < p.length; i++) {
                    p[i] -= mean;
                }
                double[] x = SparseRecovery.omp(d, p, maxAtoms, target, null, null);
                double[] rec = DenseMatrices.multiply(d, x);
                for (int i = 0; i < side; i++) {
                    for (int j = 0; j < side; j++) {
                        sum[r + i][c + j] += rec[i * side + j] + mean;
                        count[r + i][c + j] += 1;
                    }
                }
            }
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                sum[i][j] = count[i][j] > 0 ? sum[i][j] / count[i][j] : noisy[i][j];
            }
        }
        return sum;
    }

    /**
     * The share of planted atoms recovered: a planted atom counts when some learned atom has
     * absolute inner product above {@code threshold} with it.
     *
     * @param planted the true dictionary
     * @param learned the learned dictionary
     * @param threshold such as 0.99
     * @return the recovered share
     */
    public static double recovered(double[][] planted, double[][] learned, double threshold) {
        int dim = planted.length;
        int found = 0;
        for (int k = 0; k < planted[0].length; k++) {
            double best = 0;
            for (int j = 0; j < learned[0].length; j++) {
                double dot = 0;
                for (int r = 0; r < dim; r++) {
                    dot += planted[r][k] * learned[r][j];
                }
                best = Math.max(best, Math.abs(dot));
            }
            found += best > threshold ? 1 : 0;
        }
        return found / (double) planted[0].length;
    }

    /**
     * The peak signal-to-noise ratio for images in {@code [0, 1]}.
     *
     * @param a an image
     * @param b another
     * @return {@code 10 log10(1 / MSE)}
     */
    public static double psnr(double[][] a, double[][] b) {
        double mse = 0;
        int n = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                mse += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
                n++;
            }
        }
        return 10 * Math.log10(1 / Math.max(mse / n, 1e-12));
    }
}
