package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * QTIP-style trellis-coded weight quantization at 2 to 4 bits per weight. Scalar quantizers
 * at 2 bits have four levels per weight; vector quantizers do better but their codebooks grow
 * as {@code 2^(k d)}. Trellis-coded quantization (TCQ) gets the vector-quantizer gain with a
 * tiny state machine: a sequence of {@code T} weights is coded by a walk through a
 * <em>bit-shift trellis</em> with {@code 2^L} states, each step shifting {@code k} new bits
 * into an {@code L}-bit register ({@code s' = (s << k | b) mod 2^L}), and each state stands
 * for one reconstruction value. The value of a state is <em>computed</em>, not stored: a
 * hash of the state ("1MAD": a multiply-add, then the sum of the four bytes) gives an
 * approximately Gaussian value, so the codebook costs no memory and no lookups. The best walk
 * for a sequence is found by Viterbi dynamic programming; storage is {@code k} bits per weight
 * (plus the start state; QTIP's tail-biting trellis removes that overhead).
 *
 * <p>The Gaussian codebook fits because of <em>incoherence processing</em>: the weight is
 * rotated on both sides by random-sign Hadamard transforms ({@code W' = H S_in W S_out H}, as
 * QuIP# does), which spreads outliers and makes entries close to i.i.d. Gaussian; each column
 * is then scaled to unit RMS. {@link #quantize} returns the dequantized weight (rotations
 * undone), which is what a fused kernel computes on the fly. Both dimensions must be powers of
 * two for the rotation (otherwise only the scaling applies). This implements the trellis and
 * codebook of QTIP; its Hessian-aware BlockLDLQ rounding is left out (plain mean-squared
 * error per sequence).
 *
 * <p>Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing" (NeurIPS
 * 2024), <a href="https://arxiv.org/abs/2406.11235">arXiv:2406.11235</a>.
 */
public final class TrellisQuant {

    private final int bits;
    private final int stateBits;
    private final int sequence;
    private final long seed;
    private final float[] codebook;

    /**
     * @param bits bits per weight {@code k}, 1 to 4
     * @param stateBits the trellis register {@code L}, larger than {@code bits}; QTIP uses 16,
     *     10 to 12 is enough for small layers
     * @param sequence weights per trellis walk {@code T}
     * @param seed seeds the rotation signs
     */
    public TrellisQuant(int bits, int stateBits, int sequence, long seed) {
        if (bits < 1 || bits > 4 || stateBits <= bits || stateBits > 20 || sequence < 1) {
            throw new IllegalArgumentException("need 1 <= bits <= 4 < stateBits <= 20 and a positive sequence");
        }
        this.bits = bits;
        this.stateBits = stateBits;
        this.sequence = sequence;
        this.seed = seed;
        this.codebook = new float[1 << stateBits];
        for (int s = 0; s < codebook.length; s++) {
            codebook[s] = value(s);
        }
    }

    /**
     * QTIP's "1MAD" computed code: one multiply-add on the state, then the sum of the four
     * bytes of the result (a sum of four roughly uniform bytes is close to Gaussian), centred
     * and scaled to unit variance.
     *
     * @param state the trellis state
     * @return its reconstruction value, about {@code N(0, 1)}
     */
    public static float value(int state) {
        int x = state * 34038481 + 76625530;
        int sum = (x & 0xFF) + ((x >>> 8) & 0xFF) + ((x >>> 16) & 0xFF) + ((x >>> 24) & 0xFF);
        return (float) ((sum - 510) / 147.8);
    }

    /** @return bits per weight */
    public int bits() {
        return bits;
    }

    /**
     * Codes one sequence with the Viterbi algorithm.
     *
     * @param x the target values
     * @return the best state walk, one state per value
     */
    public int[] encode(float[] x) {
        int n = x.length;
        int states = 1 << stateBits;
        int mask = states - 1;
        int branch = 1 << bits;
        float[] cost = new float[states];
        float[] next = new float[states];
        byte[][] from = new byte[n][states];
        for (int s = 0; s < states; s++) {
            float d = x[0] - codebook[s];
            cost[s] = d * d;
        }
        int shift = stateBits - bits;
        for (int t = 1; t < n; t++) {
            float xt = x[t];
            for (int s = 0; s < states; s++) {
                // predecessors p with (p << k) & mask == s & ~(2^k - 1): p = (s >> k) | (top << shift)
                int low = s >>> bits;
                float best = Float.POSITIVE_INFINITY;
                int arg = 0;
                for (int top = 0; top < branch; top++) {
                    int p = low | (top << shift);
                    if (cost[p] < best) {
                        best = cost[p];
                        arg = top;
                    }
                }
                float d = xt - codebook[s];
                next[s] = best + d * d;
                from[t][s] = (byte) arg;
            }
            float[] tmp = cost;
            cost = next;
            next = tmp;
        }
        int end = 0;
        for (int s = 1; s < states; s++) {
            if (cost[s] < cost[end]) {
                end = s;
            }
        }
        int[] walk = new int[n];
        walk[n - 1] = end;
        for (int t = n - 1; t > 0; t--) {
            int s = walk[t];
            walk[t - 1] = ((s >>> bits) | (from[t][s] << shift)) & mask;
        }
        return walk;
    }

    /**
     * The reconstruction of a walk.
     *
     * @param walk states
     * @return their values
     */
    public float[] decode(int[] walk) {
        float[] out = new float[walk.length];
        for (int i = 0; i < walk.length; i++) {
            out[i] = codebook[walk[i]];
        }
        return out;
    }

    /**
     * Quantizes a weight and returns it dequantized.
     *
     * @param weight {@code [in, out]}
     * @return the reconstruction, same shape
     */
    public Tensor quantize(Tensor weight) {
        int in = weight.shape().dim(0);
        int out = weight.shape().dim(1);
        float[] w = weight.toArray();
        boolean rotIn = Integer.bitCount(in) == 1;
        boolean rotOut = Integer.bitCount(out) == 1;
        SplittableRandom r = new SplittableRandom(seed);
        float[] sIn = signs(in, r);
        float[] sOut = signs(out, r);
        if (rotIn) {
            rotateColumns(w, in, out, sIn, false);
        }
        if (rotOut) {
            rotateRows(w, in, out, sOut, false);
        }
        float[] col = new float[in];
        for (int c = 0; c < out; c++) {
            double ss = 0;
            for (int i = 0; i < in; i++) {
                col[i] = w[i * out + c];
                ss += col[i] * col[i];
            }
            float scale = (float) Math.sqrt(ss / in);
            if (scale == 0) {
                continue;
            }
            // fit the scale to the codebook's own spread: the best scale for unit-RMS targets is about 1
            for (int start = 0; start < in; start += sequence) {
                int len = Math.min(sequence, in - start);
                float[] x = new float[len];
                for (int i = 0; i < len; i++) {
                    x[i] = col[start + i] / scale;
                }
                float[] y = decode(encode(x));
                for (int i = 0; i < len; i++) {
                    w[(start + i) * out + c] = y[i] * scale;
                }
            }
        }
        if (rotOut) {
            rotateRows(w, in, out, sOut, true);
        }
        if (rotIn) {
            rotateColumns(w, in, out, sIn, true);
        }
        return Tensor.of(Shape.of(in, out), w, weight.device());
    }

    /**
     * Quantizes a dense layer's weight.
     *
     * @param dense the layer
     * @return a copy with the dequantized weight
     */
    public Dense quantize(Dense dense) {
        return WeightPruning.replaceWeight(dense, quantize(dense.weight().value()));
    }

    private static float[] signs(int n, SplittableRandom r) {
        float[] s = new float[n];
        for (int i = 0; i < n; i++) {
            s[i] = r.nextBoolean() ? 1 : -1;
        }
        return s;
    }

    /** {@code W <- H S W} (or its inverse {@code S H W}) on every column. */
    private static void rotateColumns(float[] w, int in, int out, float[] signs, boolean inverse) {
        float[] col = new float[in];
        for (int c = 0; c < out; c++) {
            for (int i = 0; i < in; i++) {
                col[i] = w[i * out + c] * (inverse ? 1 : signs[i]);
            }
            RotatedQuant.walshHadamard(col);
            for (int i = 0; i < in; i++) {
                w[i * out + c] = col[i] * (inverse ? signs[i] : 1);
            }
        }
    }

    /** {@code W <- W S H} (or its inverse {@code W H S}) on every row. */
    private static void rotateRows(float[] w, int in, int out, float[] signs, boolean inverse) {
        float[] row = new float[out];
        for (int i = 0; i < in; i++) {
            for (int c = 0; c < out; c++) {
                row[c] = w[i * out + c] * (inverse ? 1 : signs[c]);
            }
            RotatedQuant.walshHadamard(row);
            for (int c = 0; c < out; c++) {
                w[i * out + c] = row[c] * (inverse ? signs[c] : 1);
            }
        }
    }

    @Override
    public String toString() {
        return "TrellisQuant(" + bits + " bits, L=" + stateBits + ", T=" + sequence + ")";
    }
}
