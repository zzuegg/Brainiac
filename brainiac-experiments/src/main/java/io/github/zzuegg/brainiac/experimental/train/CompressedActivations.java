package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * ActNN (Chen et al. 2021, <a href="https://arxiv.org/abs/2104.14129">arXiv:2104.14129</a>):
 * activations saved for the backward pass are compressed to a few bits and decompressed only
 * when the gradients are computed. The forward pass is exact; each saved activation is
 * quantized per group of {@code group} consecutive values with the group's min and max (kept as
 * two 16-bit scales) and <em>stochastic</em> rounding, so the weight gradient
 * {@code x_hat^T g} stays unbiased with bounded extra variance. ReLU keeps a 1-bit mask, which is
 * exact.
 *
 * <p>Route dense layers through {@link #dense} and ReLUs through {@link #relu}; the meter
 * {@link #savedBytes()} adds up what the backward pass keeps (what a native implementation would
 * hold in memory; this simulation still references the float tensors in the graph). With
 * {@code bits = 32} it stores floats, the uncompressed baseline. It complements gradient
 * checkpointing (core {@code Recompute}), which trades compute for memory instead.
 */
public final class CompressedActivations {

    private final int bits;
    private final int group;
    private final SplittableRandom random;
    private long saved;

    /**
     * @param bits bits per saved activation: 1 to 8 for quantization, 16 for bf16 rounding, 32 for none
     * @param group values per min/max group, for example 64
     * @param seed the seed of the stochastic rounding
     */
    public CompressedActivations(int bits, int group, long seed) {
        if (!(bits >= 1 && bits <= 8 || bits == 16 || bits == 32)) {
            throw new IllegalArgumentException("bits must be 1-8, 16 or 32");
        }
        this.bits = bits;
        this.group = group;
        this.random = new SplittableRandom(seed);
    }

    /** @return the bytes saved for backward since the last {@link #resetMeter()} */
    public long savedBytes() {
        return saved;
    }

    /** Restarts the saved-bytes meter, for example at each step. */
    public void resetMeter() {
        saved = 0;
    }

    /**
     * Compresses and decompresses a tensor as it would be stored, and meters its bytes.
     *
     * @param x the activation
     * @return what the backward pass sees
     */
    public Tensor store(Tensor x) {
        long n = x.size();
        if (bits == 32) {
            saved += 4 * n;
            return x;
        }
        if (bits == 16) {
            saved += 2 * n;
            return StochasticRounding.nearestBf16(x);
        }
        float[] v = x.toArray();
        float[] out = new float[v.length];
        int levels = (1 << bits) - 1;
        for (int start = 0; start < v.length; start += group) {
            int end = Math.min(v.length, start + group);
            float lo = Float.POSITIVE_INFINITY;
            float hi = Float.NEGATIVE_INFINITY;
            for (int i = start; i < end; i++) {
                lo = Math.min(lo, v[i]);
                hi = Math.max(hi, v[i]);
            }
            float range = hi - lo;
            for (int i = start; i < end; i++) {
                if (range == 0) {
                    out[i] = lo;
                    continue;
                }
                double scaled = (v[i] - lo) / range * levels;
                int q = (int) Math.floor(scaled);
                if (random.nextDouble() < scaled - q) {
                    q++;
                }
                out[i] = lo + range * Math.min(q, levels) / levels;
            }
            saved += 4;  // two 16-bit scales per group
        }
        saved += (n * bits + 7) / 8;
        return Tensor.of(x.shape(), out);
    }

    /**
     * A dense layer {@code x W + b} whose saved input is compressed.
     *
     * @param x the input {@code [batch, in]}
     * @param w the weight {@code [in, out]}
     * @param b the bias {@code [1, out]}
     * @return the output
     */
    public Var dense(Var x, Var w, Var b) {
        Tensor stored = store(x.value());
        return Var.custom(in -> in[0].matmul(in[1]).add(in[2]), (in, out, g) -> new Tensor[] {
                g.matmul(in[1].transpose()),
                stored.transpose().matmul(g),
                g.sum(0, true)
        }, x, w, b);
    }

    /**
     * A ReLU that keeps only a 1-bit mask for the backward pass.
     *
     * @param x the input
     * @return {@code max(x, 0)}
     */
    public Var relu(Var x) {
        Tensor mask = x.value().greater(Tensor.zeros(io.github.zzuegg.brainiac.tensor.Shape.of()));
        saved += (x.value().size() + 7) / 8;
        return Var.custom(in -> in[0].relu(), (in, out, g) -> new Tensor[] {g.mul(mask)}, x);
    }

    @Override
    public String toString() {
        return "CompressedActivations(" + bits + " bits, group " + group + ")";
    }
}
