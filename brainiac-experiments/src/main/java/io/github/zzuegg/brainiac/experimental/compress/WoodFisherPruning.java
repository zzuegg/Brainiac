package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.function.IntFunction;

/**
 * WoodFisher (Singh and Alistarh 2020, <a href="https://arxiv.org/abs/2004.14340">arXiv:2004.14340</a>):
 * optimal brain surgeon with an empirical-Fisher inverse. Optimal Brain Surgeon removes weight
 * {@code q} at saliency {@code w_q^2 / (2 [H^-1]_qq)} and moves the remaining weights by
 * {@code -w_q H^-1 e_q / [H^-1]_qq} to compensate. WoodFisher approximates {@code H} block-diagonally
 * by the empirical Fisher {@code F = lambda I + (1/N) sum_n g_n g_n^T} of per-sample gradients and
 * builds its inverse directly by Woodbury rank-one updates,
 * {@code F^-1 <- F^-1 - (F^-1 g)(F^-1 g)^T / (N + g^T F^-1 g)}, which makes OBS feasible for whole
 * networks. SparseGPT's layer-wise variant uses the input Hessian {@code X^T X} instead, which ignores
 * the loss.
 *
 * <p>Within each block the weights chosen for removal {@code Q} are pruned jointly, with the
 * exact multi-weight OBS update {@code dw = -F^-1[:, Q] (F^-1[Q, Q])^-1 w_Q}. {@link #obd} is the
 * diagonal-Fisher ablation (optimal brain damage: saliency {@code F_qq w_q^2 / 2}, no update).
 */
public final class WoodFisherPruning {

    private final int blockSize;
    private final double dampening;

    /**
     * @param blockSize the size of the diagonal blocks of the flattened weights, for example 64
     * @param dampening {@code lambda} relative to the mean Fisher diagonal, for example 1e-3
     */
    public WoodFisherPruning(int blockSize, double dampening) {
        this.blockSize = blockSize;
        this.dampening = dampening;
    }

    /**
     * Per-sample gradients of some weights.
     *
     * @param sampleLoss the loss of sample {@code n}
     * @param samples how many samples
     * @param weights the weights
     * @return {@code [samples][size of all weights]}, flattened in order
     */
    public static double[][] perSampleGradients(IntFunction<Var> sampleLoss, int samples, List<Var> weights) {
        double[][] g = new double[samples][];
        for (int n = 0; n < samples; n++) {
            g[n] = CurvatureProducts.gradient(sampleLoss.apply(n), weights);
        }
        return g;
    }

    /**
     * Prunes every weight matrix to the same sparsity, with the OBS update.
     *
     * @param weights the matrices
     * @param grads per-sample gradients of exactly these weights, see {@link #perSampleGradients}
     * @param sparsity the share to remove per matrix (counting already-zero weights)
     * @param update whether to apply the compensating update (false gives WoodFisher saliency only)
     */
    public void prune(List<Var> weights, double[][] grads, double sparsity, boolean update) {
        int offset = 0;
        for (Var w : weights) {
            int size = (int) w.shape().size();
            double[] v = w.value().toDoubleArray();
            int blocks = Math.ceilDiv(size, blockSize);
            double[][][] inv = new double[blocks][][];
            double[] sal = new double[size];
            for (int b = 0; b < blocks; b++) {
                int b0 = b * blockSize;
                int len = Math.min(blockSize, size - b0);
                inv[b] = inverseFisher(grads, offset + b0, len);
                for (int i = 0; i < len; i++) {
                    double wi = v[b0 + i];
                    sal[b0 + i] = wi == 0 ? -1 : wi * wi / (2 * inv[b][i][i]);
                }
            }
            boolean[] remove = lowest(sal, (int) Math.round(sparsity * size));
            for (int b = 0; b < blocks; b++) {
                int b0 = b * blockSize;
                int len = Math.min(blockSize, size - b0);
                int[] q = selected(remove, b0, len);
                if (q.length == 0) {
                    continue;
                }
                if (update) {
                    obsUpdate(v, b0, len, inv[b], q);
                }
                for (int i : q) {
                    v[b0 + i] = 0;
                }
            }
            w.assign(Tensor.of(w.shape(), toFloat(v)));
            offset += size;
        }
    }

    /**
     * Optimal brain damage with the diagonal empirical Fisher: removes the lowest
     * {@code F_qq w_q^2} per matrix, without a weight update.
     *
     * @param weights the matrices
     * @param grads per-sample gradients of these weights
     * @param sparsity the share to remove per matrix
     */
    public static void obd(List<Var> weights, double[][] grads, double sparsity) {
        int offset = 0;
        for (Var w : weights) {
            int size = (int) w.shape().size();
            double[] v = w.value().toDoubleArray();
            double[] sal = new double[size];
            for (int i = 0; i < size; i++) {
                double f = 0;
                for (double[] g : grads) {
                    f += g[offset + i] * g[offset + i];
                }
                sal[i] = v[i] == 0 ? -1 : f / grads.length * v[i] * v[i];
            }
            boolean[] remove = lowest(sal, (int) Math.round(sparsity * size));
            for (int i = 0; i < size; i++) {
                if (remove[i]) {
                    v[i] = 0;
                }
            }
            w.assign(Tensor.of(w.shape(), toFloat(v)));
            offset += size;
        }
    }

    /** The inverse of one diagonal block of the dampened empirical Fisher, by Woodbury updates. */
    double[][] inverseFisher(double[][] grads, int from, int len) {
        int n = grads.length;
        double diag = 0;
        for (double[] g : grads) {
            for (int i = 0; i < len; i++) {
                diag += g[from + i] * g[from + i];
            }
        }
        double lambda = Math.max(dampening * diag / (n * (double) len), 1e-12);
        double[][] inv = new double[len][len];
        for (int i = 0; i < len; i++) {
            inv[i][i] = 1 / lambda;
        }
        double[] u = new double[len];
        for (double[] g : grads) {
            double denom = n;
            for (int i = 0; i < len; i++) {
                double s = 0;
                for (int j = 0; j < len; j++) {
                    s += inv[i][j] * g[from + j];
                }
                u[i] = s;
                denom += g[from + i] * s;
            }
            for (int i = 0; i < len; i++) {
                double ui = u[i] / denom;
                for (int j = 0; j < len; j++) {
                    inv[i][j] -= ui * u[j];
                }
            }
        }
        return inv;
    }

    private static void obsUpdate(double[] v, int b0, int len, double[][] inv, int[] q) {
        int k = q.length;
        double[][] a = new double[k][k];
        double[] wq = new double[k];
        for (int i = 0; i < k; i++) {
            wq[i] = v[b0 + q[i]];
            for (int j = 0; j < k; j++) {
                a[i][j] = inv[q[i]][q[j]];
            }
            a[i][i] += 1e-12;
        }
        double[] x = CurvatureProducts.solveSpd(a, wq);
        for (int r = 0; r < len; r++) {
            double d = 0;
            for (int i = 0; i < k; i++) {
                d += inv[r][q[i]] * x[i];
            }
            v[b0 + r] -= d;
        }
    }

    private static int[] selected(boolean[] remove, int b0, int len) {
        int c = 0;
        for (int i = 0; i < len; i++) {
            c += remove[b0 + i] ? 1 : 0;
        }
        int[] q = new int[c];
        c = 0;
        for (int i = 0; i < len; i++) {
            if (remove[b0 + i]) {
                q[c++] = i;
            }
        }
        return q;
    }

    static boolean[] lowest(double[] score, int count) {
        Integer[] idx = new Integer[score.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(score[a], score[b]));
        boolean[] r = new boolean[score.length];
        for (int k = 0; k < Math.min(count, score.length); k++) {
            r[idx[k]] = true;
        }
        return r;
    }

    static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }
}
