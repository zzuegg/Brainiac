package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * One-shot pruning of a weight matrix {@code W [in, out]} (the core {@link Dense} layout)
 * from a little calibration data {@code X [samples, in]}, without retraining:
 * <ul>
 * <li>{@link #magnitude}: drop the smallest {@code |w|} of each output unit (the classic
 *     baseline);</li>
 * <li>{@link #wanda}: Wanda (Sun et al. 2023, <a href="https://arxiv.org/abs/2306.11695">arXiv:2306.11695</a>),
 *     score {@code |w_ij| * ||X_i||}, the weight times the norm of its input feature, per
 *     output unit; no weight update, as fast as magnitude pruning but aware of the large
 *     activation outliers of language models;</li>
 * <li>{@link #sparseGpt}: SparseGPT (Frantar and Alistarh 2023,
 *     <a href="https://arxiv.org/abs/2301.00774">arXiv:2301.00774</a>), optimal brain
 *     surgeon applied column by column: with the inverse Hessian {@code (X^T X + damp I)^-1}
 *     it removes the weights of least saliency {@code w^2 / [H^-1]_jj^2} in blocks and updates
 *     the remaining weights of each unit to compensate the error;</li>
 * <li>{@link #semiStructured} and the {@code n:m} form of {@link #sparseGpt}: 2:4 sparsity
 *     (NVIDIA Ampere, Mishra et al. 2021, <a href="https://arxiv.org/abs/2104.08378">arXiv:2104.08378</a>),
 *     {@code n} kept of every {@code m} consecutive inputs, which sparse tensor cores run
 *     at twice the speed.</li>
 * </ul>
 * Each returns the pruned weight; {@link #replaceWeight} builds a {@code Dense} from it.
 */
public final class WeightPruning {

    private WeightPruning() {
    }

    /**
     * @param weight {@code [in, out]}
     * @param sparsity the share of weights to remove per output unit
     * @return the pruned weight
     */
    public static Tensor magnitude(Tensor weight, double sparsity) {
        double[][] w = ThinSvd.toMatrix(weight);
        return toTensor(prunePerOutput(w, abs(w), sparsity));
    }

    /**
     * @param weight {@code [in, out]}
     * @param inputs calibration inputs {@code [samples, in]}
     * @param sparsity the share of weights to remove per output unit
     * @return the pruned weight
     */
    public static Tensor wanda(Tensor weight, Tensor inputs, double sparsity) {
        double[][] w = ThinSvd.toMatrix(weight);
        return toTensor(prunePerOutput(w, wandaScores(w, ThinSvd.toMatrix(inputs)), sparsity));
    }

    /**
     * {@code n:m} sparsity: of every {@code m} consecutive inputs of each output unit keep the
     * {@code n} with the highest score (Wanda's if inputs are given, else magnitude).
     *
     * @param weight {@code [in, out]}; {@code in} divisible by {@code m}
     * @param inputs calibration inputs, or null
     * @param n kept per group
     * @param m group size
     * @return the pruned weight
     */
    public static Tensor semiStructured(Tensor weight, Tensor inputs, int n, int m) {
        double[][] w = ThinSvd.toMatrix(weight);
        double[][] score = inputs == null ? abs(w) : wandaScores(w, ThinSvd.toMatrix(inputs));
        int in = w.length;
        int out = w[0].length;
        if (in % m != 0) {
            throw new IllegalArgumentException("inputs " + in + " not divisible by " + m);
        }
        for (int j = 0; j < out; j++) {
            for (int g = 0; g < in; g += m) {
                Integer[] idx = new Integer[m];
                for (int k = 0; k < m; k++) {
                    idx[k] = g + k;
                }
                int col = j;
                Arrays.sort(idx, (a, b) -> Double.compare(score[a][col], score[b][col]));
                for (int k = 0; k < m - n; k++) {
                    w[idx[k]][j] = 0;
                }
            }
        }
        return toTensor(w);
    }

    /**
     * SparseGPT.
     *
     * @param weight {@code [in, out]}
     * @param inputs calibration inputs {@code [samples, in]}
     * @param sparsity the share to remove (ignored when {@code m > 0})
     * @param n for {@code n:m} sparsity, kept per group; 0 for unstructured
     * @param m for {@code n:m} sparsity, the group size; 0 for unstructured
     * @param blockSize the columns whose mask is chosen together (adaptive mask selection), for example 32
     * @param damp the dampening as a share of the mean Hessian diagonal, for example 0.01
     * @return the pruned and updated weight
     */
    public static Tensor sparseGpt(Tensor weight, Tensor inputs, double sparsity, int n, int m, int blockSize,
            double damp) {
        double[][] wt = transpose(ThinSvd.toMatrix(weight)); // [out][in]
        double[][] x = ThinSvd.toMatrix(inputs);
        int rows = wt.length;
        int cols = wt[0].length;
        double[][] h = new double[cols][cols];
        for (double[] sample : x) {
            for (int i = 0; i < cols; i++) {
                if (sample[i] == 0) {
                    continue;
                }
                for (int j = 0; j < cols; j++) {
                    h[i][j] += 2 * sample[i] * sample[j];
                }
            }
        }
        double mean = 0;
        for (int i = 0; i < cols; i++) {
            mean += h[i][i] / cols;
        }
        for (int i = 0; i < cols; i++) {
            if (h[i][i] == 0) {
                h[i][i] = 1;
            }
            h[i][i] += damp * Math.max(mean, 1e-8);
        }
        double[][] u = QuantMath.choleskyUpper(QuantMath.inverse(h));
        boolean[][] mask = new boolean[rows][cols];
        for (int b0 = 0; b0 < cols; b0 += blockSize) {
            int b1 = Math.min(cols, b0 + blockSize);
            if (m == 0) {
                int count = b1 - b0;
                int drop = (int) Math.round(sparsity * count);
                for (int r = 0; r < rows; r++) {
                    Integer[] idx = new Integer[count];
                    for (int k = 0; k < count; k++) {
                        idx[k] = b0 + k;
                    }
                    double[] row = wt[r];
                    Arrays.sort(idx, (a, b) -> Double.compare(saliency(row, u, a), saliency(row, u, b)));
                    for (int k = 0; k < drop; k++) {
                        mask[r][idx[k]] = true;
                    }
                }
            }
            for (int j = b0; j < b1; j++) {
                if (m > 0 && j % m == 0) {
                    for (int r = 0; r < rows; r++) {
                        Integer[] idx = new Integer[m];
                        for (int k = 0; k < m; k++) {
                            idx[k] = j + k;
                        }
                        double[] row = wt[r];
                        Arrays.sort(idx, (a, b) -> Double.compare(saliency(row, u, a), saliency(row, u, b)));
                        for (int k = 0; k < m - n; k++) {
                            mask[r][idx[k]] = true;
                        }
                    }
                }
                double d = u[j][j];
                for (int r = 0; r < rows; r++) {
                    double w = wt[r][j];
                    double q = mask[r][j] ? 0 : w;
                    double err = (w - q) / d;
                    for (int k = j + 1; k < cols; k++) {
                        wt[r][k] -= err * u[j][k];
                    }
                    wt[r][j] = q;
                }
            }
        }
        return toTensor(transpose(wt));
    }

    /**
     * @param base a layer
     * @param weight its new weight {@code [in, out]}
     * @return a {@code Dense} with that weight and the base's bias and activation
     */
    public static Dense replaceWeight(Dense base, Tensor weight) {
        return new Dense(weight.shape().dim(0), weight.shape().dim(1), base.activation(), weight,
                base.bias().value());
    }

    /**
     * @param weight a weight
     * @return the share of exact zeros
     */
    public static double sparsity(Tensor weight) {
        float[] v = weight.toArray();
        int z = 0;
        for (float f : v) {
            z += f == 0 ? 1 : 0;
        }
        return (double) z / v.length;
    }

    /**
     * @param weight {@code [in, out]}
     * @param n the most non-zeros per group
     * @param m the group size along the input axis
     * @return whether every group of {@code m} inputs of every unit has at most {@code n} non-zeros
     */
    public static boolean isNofM(Tensor weight, int n, int m) {
        double[][] w = ThinSvd.toMatrix(weight);
        for (int j = 0; j < w[0].length; j++) {
            for (int g = 0; g < w.length; g += m) {
                int nz = 0;
                for (int k = g; k < Math.min(w.length, g + m); k++) {
                    nz += w[k][j] != 0 ? 1 : 0;
                }
                if (nz > n) {
                    return false;
                }
            }
        }
        return true;
    }

    // ------------------------------------------------------------------ helpers

    private static double saliency(double[] row, double[][] u, int j) {
        return row[j] * row[j] / (u[j][j] * u[j][j]);
    }

    private static double[][] abs(double[][] w) {
        double[][] s = new double[w.length][w[0].length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                s[i][j] = Math.abs(w[i][j]);
            }
        }
        return s;
    }

    private static double[][] wandaScores(double[][] w, double[][] x) {
        int in = w.length;
        double[] norm = new double[in];
        for (double[] sample : x) {
            for (int i = 0; i < in; i++) {
                norm[i] += sample[i] * sample[i];
            }
        }
        double[][] s = abs(w);
        for (int i = 0; i < in; i++) {
            double n = Math.sqrt(norm[i]);
            for (int j = 0; j < w[0].length; j++) {
                s[i][j] *= n;
            }
        }
        return s;
    }

    private static double[][] prunePerOutput(double[][] w, double[][] score, double sparsity) {
        int in = w.length;
        int drop = (int) Math.round(sparsity * in);
        for (int j = 0; j < w[0].length; j++) {
            Integer[] idx = new Integer[in];
            for (int i = 0; i < in; i++) {
                idx[i] = i;
            }
            int col = j;
            Arrays.sort(idx, (a, b) -> Double.compare(score[a][col], score[b][col]));
            for (int k = 0; k < drop; k++) {
                w[idx[k]][j] = 0;
            }
        }
        return w;
    }

    private static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    private static Tensor toTensor(double[][] w) {
        int r = w.length;
        int c = w[0].length;
        float[] f = new float[r * c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                f[i * c + j] = (float) w[i][j];
            }
        }
        return Tensor.of(Shape.of(r, c), f);
    }
}
