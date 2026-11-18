package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Unbiased stochastic gradient quantization, exchanged by all-gather:
 *
 * <ul>
 *   <li>{@link #qsgd} (Alistarh et al. 2017, <a href="https://arxiv.org/abs/1610.02132">arXiv:1610.02132</a>):
 *       per bucket of {@code d} values, {@code Q(v_j) = |v| sign(v_j) xi_j / s} with
 *       {@code xi_j} the stochastic rounding of {@code s |v_j| / |v|} to an integer level. The
 *       levels are Elias-gamma coded (small levels are frequent), plus a sign bit for nonzero
 *       levels and a 32-bit norm per bucket; the bytes count those bits.</li>
 *   <li>{@link #ternGrad} (Wen et al. 2017, <a href="https://arxiv.org/abs/1705.07878">arXiv:1705.07878</a>):
 *       {@code Q(v_j) = s sign(v_j) b_j} with {@code b_j ~ Bernoulli(|v_j| / s)} and
 *       {@code s = max |v|} after clipping at 2.5 standard deviations; two bits per value plus
 *       a 32-bit scale.</li>
 * </ul>
 *
 * <p>Both are unbiased, so no error feedback is needed; the price is variance that grows as the
 * levels get coarser ({@link #relativeVariance}). Stochastic rounding is the same idea that
 * {@link StochasticRounding} applies to weights.
 */
public final class QuantizedGradients implements GradientCompressor {

    private final int levels;
    private final int bucket;
    private final boolean ternary;
    private final SplittableRandom random;
    private long bits;

    private QuantizedGradients(int levels, int bucket, boolean ternary, long seed) {
        this.levels = levels;
        this.bucket = bucket;
        this.ternary = ternary;
        this.random = new SplittableRandom(seed);
    }

    /**
     * QSGD.
     *
     * @param levels the levels {@code s} per sign, for example 1, 4 or 16
     * @param bucket the bucket size {@code d}, for example 128
     * @param seed the seed of the rounding
     * @return the compressor
     */
    public static QuantizedGradients qsgd(int levels, int bucket, long seed) {
        return new QuantizedGradients(levels, bucket, false, seed);
    }

    /**
     * TernGrad.
     *
     * @param seed the seed of the rounding
     * @return the compressor
     */
    public static QuantizedGradients ternGrad(long seed) {
        return new QuantizedGradients(1, Integer.MAX_VALUE, true, seed);
    }

    /**
     * Quantizes one tensor.
     *
     * @param g the values
     * @return the dequantized values, same shape
     */
    public Tensor quantize(Tensor g) {
        float[] v = g.toArray();
        float[] q = new float[v.length];
        if (ternary) {
            double mean = 0;
            double sq = 0;
            for (float x : v) {
                mean += x;
                sq += (double) x * x;
            }
            mean /= v.length;
            double std = Math.sqrt(Math.max(0, sq / v.length - mean * mean));
            double limit = 2.5 * std;
            double s = 0;
            for (int i = 0; i < v.length; i++) {
                double c = limit > 0 ? Math.max(-limit, Math.min(limit, v[i])) : v[i];
                v[i] = (float) c;
                s = Math.max(s, Math.abs(c));
            }
            for (int i = 0; i < v.length; i++) {
                if (s > 0 && random.nextDouble() < Math.abs(v[i]) / s) {
                    q[i] = (float) Math.copySign(s, v[i]);
                }
            }
            bits += 2L * v.length + 32;
            return Tensor.of(g.shape(), q);
        }
        for (int start = 0; start < v.length; start += bucket) {
            int end = Math.min(v.length, start + bucket);
            double norm = 0;
            for (int i = start; i < end; i++) {
                norm += (double) v[i] * v[i];
            }
            norm = Math.sqrt(norm);
            bits += 32;
            for (int i = start; i < end; i++) {
                int level = 0;
                if (norm > 0) {
                    double scaled = Math.abs(v[i]) / norm * levels;
                    level = (int) Math.floor(scaled);
                    if (random.nextDouble() < scaled - level) {
                        level++;
                    }
                }
                q[i] = (float) (Math.copySign(norm * level / levels, v[i]));
                bits += eliasGamma(level + 1) + (level > 0 ? 1 : 0);
            }
        }
        return Tensor.of(g.shape(), q);
    }

    /**
     * The length of the Elias gamma code of a positive integer: {@code 2 floor(log2 n) + 1} bits.
     *
     * @param n the integer, at least 1
     * @return its code length
     */
    public static int eliasGamma(int n) {
        return 2 * (31 - Integer.numberOfLeadingZeros(n)) + 1;
    }

    /**
     * Estimates {@code E|Q(g) - g|^2 / |g|^2} by repeated quantization.
     *
     * @param g the values
     * @param samples the quantizations averaged
     * @return the relative variance
     */
    public double relativeVariance(Tensor g, int samples) {
        long saved = bits;
        float[] v = g.toArray();
        double norm = 0;
        for (float x : v) {
            norm += (double) x * x;
        }
        double err = 0;
        for (int s = 0; s < samples; s++) {
            float[] q = quantize(g).toArray();
            for (int i = 0; i < v.length; i++) {
                err += (q[i] - v[i]) * (double) (q[i] - v[i]);
            }
        }
        bits = saved;
        return err / samples / norm;
    }

    @Override
    public Tensor aggregate(Var parameter, List<Tensor> grads) {
        long before = bits;
        Tensor sum = null;
        for (Tensor g : grads) {
            Tensor q = quantize(g);
            sum = sum == null ? q : sum.add(q);
        }
        long sent = bits - before;  // all workers' encodings; each is received by the other workers
        bits = before + sent * grads.size();
        return sum.div(grads.size());
    }

    @Override
    public long bytes() {
        return (bits + 7) / 8;
    }

    @Override
    public String toString() {
        return ternary ? "TernGrad" : "QSGD(s=" + levels + ", d=" + bucket + ")";
    }
}
