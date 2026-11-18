package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Differentiable sorting and ranking on autograd {@link Var}s.
 *
 * <p>Sorting is piecewise constant in its input, so the gradient of anything computed from a
 * permutation is zero almost everywhere. The relaxations here replace the permutation matrix
 * {@code P} (row {@code i} the one-hot of the {@code i}-th largest score) by a smooth
 * row-stochastic matrix, or the sorted values and ranks by a smooth projection:
 * <ul>
 *   <li>{@link Method#NEURAL_SORT} (Grover et al. 2019, <a href="https://arxiv.org/abs/1903.08850">
 *       arXiv:1903.08850</a>): {@code P[i] = softmax(((n + 1 - 2i) s - A_s 1) / tau)} with
 *       {@code A_s = |s_j - s_k|}, unimodal and row-stochastic;</li>
 *   <li>{@link Method#SOFT_SORT} (Prillo and Eisenschlos 2020, <a href="https://arxiv.org/abs/2006.16038">
 *       arXiv:2006.16038</a>): {@code P[i] = softmax(-|sort(s)_i - s| / tau)};</li>
 *   <li>{@link Method#SINKHORN}: the same scores normalised to a doubly-stochastic matrix by
 *       log-domain Sinkhorn iterations (Mena et al. 2018; Cuturi et al. 2019), the optimal-transport
 *       baseline (the experimental {@code EntropicTransport} solves the same problem on doubles but
 *       is not differentiable with respect to the cost);</li>
 *   <li>{@link Method#ODD_EVEN_NETWORK} and {@link Method#BITONIC_NETWORK} (Petersen et al. 2021
 *       and 2022, <a href="https://arxiv.org/abs/2203.09630">arXiv:2203.09630</a>): a sorting network
 *       whose compare-and-swap gates mix the two wires by {@code alpha = f(beta (x_j - x_i))}, the
 *       product of the layers' mixing matrices giving {@code P}; {@code f} is the logistic or, for
 *       the monotonic variant, the Cauchy CDF {@code 1/2 + atan(x) / pi}, whose heavy tails keep the
 *       gradients of the right sign;</li>
 *   <li>{@link #fastSoftSort} and {@link #fastSoftRank} (Blondel et al. 2020,
 *       <a href="https://arxiv.org/abs/2002.08871">arXiv:2002.08871</a>): projections onto the
 *       permutahedron, solved exactly by pool-adjacent-violators in {@code O(n log n)} with a
 *       block-averaging Jacobian, so no {@code n x n} matrix is formed.</li>
 * </ul>
 * {@link #softTopK} turns a relaxed permutation into a soft top-{@code k} membership vector.
 * All inputs are score batches {@code [batch, n]}; permutation outputs are {@code [batch, n, n]}
 * for a descending sort.
 */
public final class SoftSorting {

    /** A relaxed permutation-matrix method. */
    public enum Method {
        /** NeuralSort. */
        NEURAL_SORT,
        /** SoftSort. */
        SOFT_SORT,
        /** Sinkhorn-normalised SoftSort scores. */
        SINKHORN,
        /** Odd-even transposition network with logistic gates. */
        ODD_EVEN_NETWORK,
        /** Bitonic network with logistic gates. */
        BITONIC_NETWORK,
        /** Odd-even transposition network with monotonic (Cauchy) gates. */
        MONOTONIC_ODD_EVEN,
        /** Bitonic network with monotonic (Cauchy) gates. */
        MONOTONIC_BITONIC
    }

    private SoftSorting() {
    }

    /**
     * A relaxed descending-sort permutation matrix.
     *
     * @param scores the scores {@code [batch, n]}
     * @param method the relaxation
     * @param temperature {@code tau} for the softmax methods, or {@code 1 / beta} (the gate
     *     steepness) for the networks
     * @return {@code [batch, n, n]}, row {@code i} the soft one-hot of the {@code i}-th largest score
     */
    public static Var permutation(Var scores, Method method, double temperature) {
        Objects.requireNonNull(method, "method");
        int batch = scores.shape().dim(0);
        int n = scores.shape().dim(1);
        float tau = (float) temperature;
        return switch (method) {
            case NEURAL_SORT -> {
                Var col = scores.reshape(batch, n, 1);
                Var row = scores.reshape(batch, 1, n);
                Var bsum = col.sub(row).abs().sum(2, false).reshape(batch, 1, n);
                float[] c = new float[n];
                for (int i = 0; i < n; i++) {
                    c[i] = n + 1 - 2 * (i + 1);
                }
                Var logits = row.mul(Tensor.of(Shape.of(1, n, 1), c)).sub(bsum);
                yield logits.div(tau).softmax(2);
            }
            case SOFT_SORT -> softSortLogits(scores, batch, n, tau).softmax(2);
            case SINKHORN -> {
                Var log = softSortLogits(scores, batch, n, tau);
                for (int it = 0; it < 30; it++) {
                    log = log.sub(log.logsumexp(2, true));
                    log = log.sub(log.logsumexp(1, true));
                }
                yield log.exp();
            }
            case ODD_EVEN_NETWORK, MONOTONIC_ODD_EVEN -> network(scores, oddEvenLayers(n), n, tau,
                    method == Method.MONOTONIC_ODD_EVEN);
            case BITONIC_NETWORK, MONOTONIC_BITONIC -> {
                int p = Integer.highestOneBit(n);
                if (p < n) {
                    p <<= 1;
                }
                Var padded = scores;
                if (p > n) {
                    float[] pad = new float[batch * (p - n)];
                    Arrays.fill(pad, -1e4f);
                    padded = Var.concat(1, scores, Var.of(Tensor.of(Shape.of(batch, p - n), pad)));
                }
                Var full = network(padded, bitonicLayers(p), p, tau, method == Method.MONOTONIC_BITONIC);
                yield full.slice(1, 0, n).slice(2, 0, n);
            }
        };
    }

    private static Var softSortLogits(Var scores, int batch, int n, float tau) {
        Var sorted = scores.sort(1, true).reshape(batch, n, 1);
        return sorted.sub(scores.reshape(batch, 1, n)).abs().neg().div(tau);
    }

    /**
     * The comparator layers of odd-even transposition sort: {@code n} layers of disjoint
     * neighbouring pairs.
     *
     * @param n the number of wires
     * @return per layer, the pairs {@code (i, j)} with {@code i < j}
     */
    public static List<int[][]> oddEvenLayers(int n) {
        List<int[][]> layers = new ArrayList<>();
        for (int l = 0; l < n; l++) {
            List<int[]> pairs = new ArrayList<>();
            for (int i = l % 2; i + 1 < n; i += 2) {
                pairs.add(new int[] {i, i + 1});
            }
            layers.add(pairs.toArray(new int[0][]));
        }
        return layers;
    }

    /**
     * The comparator layers of Batcher's bitonic sorter for a power-of-two size, every pair
     * written as (wire that receives the larger value, wire that receives the smaller).
     *
     * @param n the number of wires, a power of two
     * @return per layer, the directed pairs
     */
    public static List<int[][]> bitonicLayers(int n) {
        List<int[][]> layers = new ArrayList<>();
        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                List<int[]> pairs = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    int l = i ^ j;
                    if (l > i) {
                        boolean descending = (i & k) == 0;
                        pairs.add(descending ? new int[] {i, l} : new int[] {l, i});
                    }
                }
                layers.add(pairs.toArray(new int[0][]));
            }
        }
        return layers;
    }

    /**
     * Runs a relaxed sorting network. Each gate {@code (i, j)} puts the larger value on wire
     * {@code i}: {@code alpha = f((x_j - x_i) / tau)} is the soft "swap" probability,
     * {@code x_i' = (1 - alpha) x_i + alpha x_j}, {@code x_j' = alpha x_i + (1 - alpha) x_j}.
     */
    private static Var network(Var scores, List<int[][]> layers, int n, float tau, boolean monotonic) {
        int batch = scores.shape().dim(0);
        Var x = scores;
        Var perm = null;
        for (int[][] layer : layers) {
            Var[] rows = new Var[n];
            boolean[] touched = new boolean[n];
            for (int[] pair : layer) {
                int i = pair[0];
                int j = pair[1];
                touched[i] = true;
                touched[j] = true;
                Var diff = x.slice(1, j, j + 1).sub(x.slice(1, i, i + 1)).div(tau);
                Var alpha = monotonic ? diff.atan().mul((float) (1 / Math.PI)).add(0.5f) : diff.sigmoid();
                Var keep = alpha.neg().add(1f);
                rows[i] = keep.mul(oneHot(n, i)).add(alpha.mul(oneHot(n, j)));
                rows[j] = alpha.mul(oneHot(n, i)).add(keep.mul(oneHot(n, j)));
            }
            for (int i = 0; i < n; i++) {
                if (!touched[i]) {
                    rows[i] = Var.of(Tensor.zeros(Shape.of(batch, 1))).add(oneHot(n, i));
                }
            }
            Var[] shaped = new Var[n];
            for (int i = 0; i < n; i++) {
                shaped[i] = rows[i].reshape(batch, 1, n);
            }
            Var mix = Var.concat(1, shaped);
            x = mix.matmul(x.reshape(batch, n, 1)).reshape(batch, n);
            perm = perm == null ? mix : mix.matmul(perm);
        }
        return perm;
    }

    private static Tensor oneHot(int n, int i) {
        float[] v = new float[n];
        v[i] = 1;
        return Tensor.of(Shape.of(1, n), v);
    }

    /**
     * Applies a relaxed permutation to values: {@code P v}.
     *
     * @param permutation {@code [batch, n, n]}
     * @param values {@code [batch, n]}
     * @return the softly sorted values {@code [batch, n]}
     */
    public static Var apply(Var permutation, Var values) {
        int batch = values.shape().dim(0);
        int n = values.shape().dim(1);
        return permutation.matmul(values.reshape(batch, n, 1)).reshape(batch, n);
    }

    /**
     * Soft top-{@code k} membership: the column sums of the first {@code k} rows of a relaxed
     * permutation, in {@code [0, 1]} and summing to {@code k} for a doubly stochastic matrix.
     *
     * @param permutation {@code [batch, n, n]}
     * @param k how many to select
     * @return {@code [batch, n]}
     */
    public static Var softTopK(Var permutation, int k) {
        return permutation.slice(1, 0, k).sum(1, false);
    }

    /**
     * The fast soft sort: {@code argmin_{mu in Perm(theta)} |mu - rho / eps|^2} with
     * {@code rho = (n, ..., 1)}, a descending soft sort of {@code theta} that tends to the hard sort
     * as {@code eps -> 0}.
     *
     * @param theta {@code [batch, n]}
     * @param epsilon the regularisation strength
     * @return the softly sorted values, descending {@code [batch, n]}
     */
    public static Var fastSoftSort(Var theta, double epsilon) {
        int batch = theta.shape().dim(0);
        int n = theta.shape().dim(1);
        return project(Var.of(rho(batch, n, 1 / epsilon)), theta);
    }

    /**
     * The fast soft rank: {@code argmin_{mu in Perm(rho)} |mu - theta / eps|^2}; the largest
     * score gets a rank near {@code n}, the smallest near 1.
     *
     * @param theta {@code [batch, n]}
     * @param epsilon the regularisation strength
     * @return the soft ranks {@code [batch, n]}
     */
    public static Var fastSoftRank(Var theta, double epsilon) {
        int batch = theta.shape().dim(0);
        int n = theta.shape().dim(1);
        return project(theta.div((float) epsilon), Var.of(rho(batch, n, 1)));
    }

    private static Tensor rho(int batch, int n, double scale) {
        float[] r = new float[batch * n];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                r[b * n + i] = (float) ((n - i) * scale);
            }
        }
        return Tensor.of(Shape.of(batch, n), r);
    }

    /**
     * The Euclidean projection of {@code z} onto the permutahedron of {@code w}, row by row:
     * {@code mu = z - v} where {@code v} (in {@code z}'s sorted order) is the decreasing isotonic
     * regression of {@code sort(z) - sort(w)}, solved by pool-adjacent-violators. The backward pass
     * averages the incoming gradient within each pooled block.
     *
     * @param z the point {@code [batch, n]}
     * @param w the generating vector {@code [batch, n]}
     * @return the projection {@code [batch, n]}
     */
    public static Var project(Var z, Var w) {
        int batch = z.shape().dim(0);
        int n = z.shape().dim(1);
        int[][][] state = new int[batch][][];
        return Var.custom(in -> {
            double[] zv = in[0].toDoubleArray();
            double[] wv = in[1].toDoubleArray();
            float[] out = new float[batch * n];
            for (int b = 0; b < batch; b++) {
                double[] zr = Arrays.copyOfRange(zv, b * n, (b + 1) * n);
                double[] wr = Arrays.copyOfRange(wv, b * n, (b + 1) * n);
                int[] sz = argsortDescending(zr);
                int[] sw = argsortDescending(wr);
                double[] y = new double[n];
                for (int i = 0; i < n; i++) {
                    y[i] = zr[sz[i]] - wr[sw[i]];
                }
                int[] block = new int[n];
                double[] v = isotonicDecreasing(y, block);
                for (int i = 0; i < n; i++) {
                    out[b * n + sz[i]] = (float) (zr[sz[i]] - v[i]);
                }
                state[b] = new int[][] {sz, sw, block};
            }
            return Tensor.of(Shape.of(batch, n), out);
        }, (inputs, output, grad) -> {
            double[] g = grad.toDoubleArray();
            float[] gz = new float[batch * n];
            float[] gw = new float[batch * n];
            for (int b = 0; b < batch; b++) {
                int[] sz = state[b][0];
                int[] sw = state[b][1];
                int[] block = state[b][2];
                double[] gs = new double[n];
                for (int i = 0; i < n; i++) {
                    gs[i] = g[b * n + sz[i]];
                }
                double[] avg = blockAverage(gs, block);
                for (int i = 0; i < n; i++) {
                    gz[b * n + sz[i]] = (float) (gs[i] - avg[i]);
                    gw[b * n + sw[i]] = (float) avg[i];
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(batch, n), gz), Tensor.of(Shape.of(batch, n), gw)};
        }, z, w);
    }

    private static double[] blockAverage(double[] g, int[] block) {
        int n = g.length;
        double[] out = new double[n];
        int i = 0;
        while (i < n) {
            int j = i;
            double s = 0;
            while (j < n && block[j] == block[i]) {
                s += g[j];
                j++;
            }
            for (int t = i; t < j; t++) {
                out[t] = s / (j - i);
            }
            i = j;
        }
        return out;
    }

    /**
     * The decreasing isotonic regression {@code argmin_{v_1 >= ... >= v_n} |v - y|^2} by
     * pool-adjacent-violators.
     *
     * @param y the targets
     * @param block receives the pooled block id of every position
     * @return the fit
     */
    public static double[] isotonicDecreasing(double[] y, int[] block) {
        int n = y.length;
        double[] sum = new double[n];
        int[] count = new int[n];
        int[] start = new int[n];
        int top = -1;
        for (int i = 0; i < n; i++) {
            top++;
            sum[top] = y[i];
            count[top] = 1;
            start[top] = i;
            while (top > 0 && sum[top - 1] / count[top - 1] <= sum[top] / count[top]) {
                sum[top - 1] += sum[top];
                count[top - 1] += count[top];
                top--;
            }
        }
        double[] v = new double[n];
        for (int b = 0; b <= top; b++) {
            int end = b < top ? start[b + 1] : n;
            for (int i = start[b]; i < end; i++) {
                v[i] = sum[b] / count[b];
                block[i] = b;
            }
        }
        return v;
    }

    /**
     * The indices that sort a vector in descending order.
     *
     * @param x the vector
     * @return the order
     */
    public static int[] argsortDescending(double[] x) {
        Integer[] idx = new Integer[x.length];
        for (int i = 0; i < x.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(x[b], x[a]));
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = idx[i];
        }
        return out;
    }
}
