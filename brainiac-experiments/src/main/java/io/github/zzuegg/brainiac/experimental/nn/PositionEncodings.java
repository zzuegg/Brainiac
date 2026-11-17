package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Position encodings for attention that the core's RoPE and ALiBi
 * ({@code Functional.rotary}, {@code Functional.alibi}) do not cover:
 *
 * <ul>
 *   <li><b>xPos</b> ({@link #xpos}): RoPE plus an exponential decay per frequency, queries
 *       scaled by {@code zeta^(n/s)} and keys by {@code zeta^(-n/s)}, so attention scores
 *       decay smoothly with distance and length extrapolation is more stable. Sun et al., "A
 *       Length-Extrapolatable Transformer", <a href="https://arxiv.org/abs/2212.10554">arXiv:2212.10554</a>.</li>
 *   <li><b>2-D RoPE</b> ({@link #rotary2d}): half the features rotate with the patch row and
 *       half with the column, so image transformers see relative 2-D offsets. Heo et al.,
 *       "Rotary Position Embedding for Vision Transformer", <a href="https://arxiv.org/abs/2403.13298">arXiv:2403.13298</a>.</li>
 *   <li><b>T5 relative bias</b> ({@link #t5Bucket}, {@link #t5Bias}): one learned scalar
 *       per head and log-spaced distance bucket, added to the scores. Raffel et al.,
 *       <a href="https://arxiv.org/abs/1910.10683">arXiv:1910.10683</a>.</li>
 *   <li><b>CoPE</b> ({@link #cope}): contextual positions; each query counts the keys
 *       between itself and a key with sigmoid gates, so "position" can mean "the 3rd
 *       sentence back" rather than "the 30th token back", and interpolates learned position
 *       embeddings at that fractional count. Golovneva et al., "Contextual Position
 *       Encoding", <a href="https://arxiv.org/abs/2405.18719">arXiv:2405.18719</a>.</li>
 *   <li><b>NoPE</b>: no encoding at all; causal attention can infer positions itself
 *       (Kazemnejad et al., <a href="https://arxiv.org/abs/2305.19466">arXiv:2305.19466</a>),
 *       selected with {@link PositionalAttention.Position#NONE}.</li>
 * </ul>
 */
public final class PositionEncodings {

    private PositionEncodings() {
    }

    /**
     * Rotary embedding at explicit (possibly fractional) positions, half-split layout like
     * {@code Functional.rotary}: feature {@code i} pairs with {@code i + d/2}.
     *
     * @param x {@code [..., steps, d]} with even {@code d}
     * @param positions one position per step
     * @param base the frequency base
     * @return the rotated values
     */
    public static Var rotaryAt(Var x, double[] positions, float base) {
        Shape s = x.shape();
        int steps = s.dim(-2);
        int d = s.dim(-1);
        if (d % 2 != 0 || positions.length != steps) {
            throw new IllegalArgumentException("rotary needs an even width and one position per step, got " + s);
        }
        int half = d / 2;
        float[] cos = new float[steps * half];
        float[] sin = new float[steps * half];
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < half; i++) {
                double angle = positions[t] * Math.pow(base, -2.0 * i / d);
                cos[t * half + i] = (float) Math.cos(angle);
                sin[t * half + i] = (float) Math.sin(angle);
            }
        }
        Device device = x.value().device();
        Tensor c = Tensor.of(Shape.of(steps, half), cos, device);
        Tensor sn = Tensor.of(Shape.of(steps, half), sin, device);
        Var x1 = x.slice(-1, 0, half);
        Var x2 = x.slice(-1, half, d);
        return Var.concat(-1, x1.mul(c).sub(x2.mul(sn)), x2.mul(c).add(x1.mul(sn)));
    }

    /**
     * xPos: rotary embedding with a per-frequency exponential scale. Apply with
     * {@code query = true} to queries and {@code false} to keys; their product then carries
     * {@code zeta^((n - m) / scaleBase)}, a decay in the distance {@code n - m}.
     *
     * @param x {@code [..., steps, d]} with even {@code d}
     * @param base the rotary base, usually 10000
     * @param scaleBase the decay length {@code s}, 512 in the paper; smaller decays faster
     * @param offset the position of the first step
     * @param query whether these are queries (scaled up) or keys (scaled down)
     * @return the encoded values
     */
    public static Var xpos(Var x, float base, float scaleBase, int offset, boolean query) {
        int steps = x.shape().dim(-2);
        int d = x.shape().dim(-1);
        int half = d / 2;
        double[] positions = new double[steps];
        float[] scale = new float[steps * d];
        for (int t = 0; t < steps; t++) {
            positions[t] = t + offset;
            for (int i = 0; i < half; i++) {
                double zeta = (2.0 * i / d + 0.4) / 1.4;
                float v = (float) Math.pow(zeta, (query ? 1 : -1) * (t + offset) / (double) scaleBase);
                scale[t * d + i] = v;
                scale[t * d + i + half] = v;
            }
        }
        return rotaryAt(x, positions, base).mul(Tensor.of(Shape.of(steps, d), scale, x.value().device()));
    }

    /**
     * 2-D rotary embedding for image patches in row-major order: the first half of the
     * features rotates with the row, the second half with the column. The first
     * {@code prefix} tokens (such as a class token) stay unrotated.
     *
     * @param x {@code [..., prefix + height * width, d]} with {@code d} a multiple of 4
     * @param height patch rows
     * @param width patch columns
     * @param base the rotary base, usually 100 for small grids
     * @param prefix leading tokens that are not patches
     * @return the encoded values
     */
    public static Var rotary2d(Var x, int height, int width, float base, int prefix) {
        int steps = x.shape().dim(-2);
        int d = x.shape().dim(-1);
        if (steps != prefix + height * width || d % 4 != 0) {
            throw new IllegalArgumentException("2-D rotary needs " + prefix + " + " + height + "x" + width
                    + " tokens and a width divisible by 4, got " + x.shape());
        }
        int rank = x.shape().rank();
        Var patches = x.slice(rank - 2, prefix, steps);
        double[] rows = new double[height * width];
        double[] cols = new double[height * width];
        for (int p = 0; p < rows.length; p++) {
            rows[p] = p / width;
            cols[p] = p % width;
        }
        Var r = rotaryAt(patches.slice(-1, 0, d / 2), rows, base);
        Var c = rotaryAt(patches.slice(-1, d / 2, d), cols, base);
        Var rotated = Var.concat(-1, r, c);
        return prefix == 0 ? rotated : Var.concat(rank - 2, x.slice(rank - 2, 0, prefix), rotated);
    }

    /**
     * The T5 bucket of a relative position {@code key - query}: exact buckets for small
     * distances, logarithmic ones up to {@code maxDistance}, and (if bidirectional) separate
     * buckets for keys after the query.
     *
     * @param relative {@code key - query}
     * @param bidirectional whether future keys get their own buckets
     * @param buckets the number of buckets
     * @param maxDistance the distance from which all keys share the last bucket
     * @return the bucket, in {@code [0, buckets)}
     */
    public static int t5Bucket(int relative, boolean bidirectional, int buckets, int maxDistance) {
        int bucket = 0;
        int n;
        if (bidirectional) {
            buckets /= 2;
            if (relative > 0) {
                bucket += buckets;
            }
            n = Math.abs(relative);
        } else {
            n = Math.max(0, -relative);
        }
        int exact = Math.max(1, buckets / 2);
        if (n < exact) {
            return bucket + n;
        }
        int large = exact + (int) (Math.log(n / (double) exact) / Math.log(maxDistance / (double) exact)
                * (buckets - exact));
        return bucket + Math.min(large, buckets - 1);
    }

    /**
     * The T5 relative position bias for a square attention map.
     *
     * @param table learned biases {@code [heads, buckets]}
     * @param queries the number of queries
     * @param keys the number of keys
     * @param bidirectional whether future keys get their own buckets
     * @param maxDistance the distance of the last bucket
     * @return {@code [heads, queries, keys]}
     */
    public static Var t5Bias(Var table, int queries, int keys, boolean bidirectional, int maxDistance) {
        int buckets = table.shape().dim(1);
        float[] idx = new float[queries * keys];
        int offset = keys - queries;
        for (int i = 0; i < queries; i++) {
            for (int j = 0; j < keys; j++) {
                idx[i * keys + j] = t5Bucket(j - (i + offset), bidirectional, buckets, maxDistance);
            }
        }
        return table.gather(1, Tensor.of(Shape.of(queries, keys), idx, table.value().device()));
    }

    /**
     * The CoPE score bias. Gates {@code g_ij = sigmoid(q_i . k_j)} over the visible keys give
     * each key a contextual position {@code p_ij = sum_{t = j..i} g_it}; the bias is
     * {@code q_i . e[p_ij]}, interpolating the learned embeddings {@code e} between the two
     * nearest whole positions.
     *
     * @param q queries {@code [batch, heads, steps, d]}
     * @param k keys {@code [batch, heads, steps, d]}
     * @param embeddings learned position embeddings {@code [maxPositions, d]}
     * @param causal whether query {@code i} sees keys {@code <= i} only (otherwise all keys,
     *     counted from the end)
     * @return the score bias {@code [batch, heads, steps, steps]}
     */
    public static Var cope(Var q, Var k, Var embeddings, boolean causal) {
        int b = q.shape().dim(0);
        int h = q.shape().dim(1);
        int t = q.shape().dim(2);
        int maxPos = embeddings.shape().dim(0);
        Device device = q.value().device();
        Var gates = q.matmul(k.transpose()).sigmoid();
        float[] visible = new float[t * t];
        float[] upper = new float[t * t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                visible[i * t + j] = !causal || j <= i ? 1f : 0f;
                upper[i * t + j] = i >= j ? 1f : 0f;   // U[s, j] = 1 if s >= j: suffix sums over keys
            }
        }
        gates = gates.mul(Tensor.of(Shape.of(t, t), visible, device));
        Var pos = gates.matmul(Tensor.of(Shape.of(t, t), upper, device)).minimum(maxPos - 1);
        Var z = q.matmul(embeddings.transpose());   // [b, h, t, maxPos]
        int rows = b * h * t;
        float[] p = pos.value().toArray();
        int[][] lo = new int[rows][t];
        int[][] hi = new int[rows][t];
        float[] floor = new float[rows * t];
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < t; j++) {
                int f = (int) Math.floor(p[r * t + j]);
                f = Math.max(0, Math.min(maxPos - 1, f));
                lo[r][j] = f;
                hi[r][j] = Math.min(maxPos - 1, f + 1);
                floor[r * t + j] = f;
            }
        }
        Var zr = z.reshape(rows, maxPos);
        Var zl = TechniqueOps.takeAlong(zr, lo);
        Var zh = TechniqueOps.takeAlong(zr, hi);
        Var w = pos.reshape(rows, t).sub(Tensor.of(Shape.of(rows, t), floor, device));
        return zl.add(zh.sub(zl).mul(w)).reshape(b, h, t, t);
    }
}
