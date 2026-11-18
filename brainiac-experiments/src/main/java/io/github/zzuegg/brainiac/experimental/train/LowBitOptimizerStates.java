package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.Fp8Format;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * AdamW whose two moments are stored in fewer bits than core {@code Optimizer.adam8bit}:
 *
 * <ul>
 *   <li>{@link #fourBit} (Li, Chen and Zhu 2023, <a href="https://arxiv.org/abs/2309.01507">arXiv:2309.01507</a>):
 *       the first moment in blocks of 128 with an absmax scale and a signed, non-linear
 *       (quadratically spaced, like the paper's dynamic-exponent map) 4-bit code; the second
 *       moment with <em>rank-1 normalization</em> (each entry scaled by the smaller of its row's
 *       and its column's maximum) and a linear 4-bit code that <em>excludes zero</em>. A zero
 *       code for the second moment turns {@code m / sqrt(v)} into a huge step, the zero-point
 *       failure of naive 4-bit states.</li>
 *   <li>{@link #coatFp8} (COAT, Xi et al., ICLR 2025, <a href="https://arxiv.org/abs/2410.19313">arXiv:2410.19313</a>):
 *       both moments in FP8 E4M3 ({@link Fp8Format}) per group of 128 after <em>dynamic range
 *       expansion</em>, {@code y -> y^k} with {@code k} chosen so the group's dynamic range matches
 *       E4M3's normal range: {@code k > 1} spreads a narrow first moment over all codes, and
 *       {@code k < 1} keeps the tiny entries of a wide second moment from underflowing to zero.</li>
 * </ul>
 *
 * <p>States are decoded, updated and re-encoded every step; {@link #stateBytes()} counts the
 * packed codes and scales.
 */
public final class LowBitOptimizerStates implements Optimizer {

    private enum Format { FOUR_BIT, NAIVE_FOUR_BIT, COAT_FP8 }

    private static final int BLOCK = 128;
    private static final float[] SIGNED_LEVELS = signedLevels();

    private final Format format;
    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final Map<Var, float[][]> states = new IdentityHashMap<>();
    private final Map<Var, Integer> steps = new IdentityHashMap<>();

    private LowBitOptimizerStates(Format format, float learningRate, float weightDecay) {
        this.format = format;
        this.learningRate = learningRate;
        this.beta1 = 0.9f;
        this.beta2 = 0.999f;
        this.epsilon = 1e-8f;
        this.weightDecay = weightDecay;
    }

    /**
     * AdamW with 4-bit states.
     *
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return the optimizer
     */
    public static LowBitOptimizerStates fourBit(float learningRate, float weightDecay) {
        return new LowBitOptimizerStates(Format.FOUR_BIT, learningRate, weightDecay);
    }

    /**
     * The ablation: 4-bit states with plain blockwise absmax scaling and a linear code that
     * includes zero for the second moment, which shows the zero-point failure.
     *
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return the optimizer
     */
    public static LowBitOptimizerStates naiveFourBit(float learningRate, float weightDecay) {
        return new LowBitOptimizerStates(Format.NAIVE_FOUR_BIT, learningRate, weightDecay);
    }

    /**
     * AdamW with COAT's FP8 states.
     *
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return the optimizer
     */
    public static LowBitOptimizerStates coatFp8(float learningRate, float weightDecay) {
        return new LowBitOptimizerStates(Format.COAT_FP8, learningRate, weightDecay);
    }

    private static float[] signedLevels() {
        float[] l = new float[15];
        for (int i = -7; i <= 7; i++) {
            l[i + 7] = Math.signum(i) * (i / 7f) * (i / 7f);
        }
        return l;
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var p : parameters) {
            if (p.grad() == null) {
                continue;
            }
            float[] g = p.grad().toArray();
            int n = g.length;
            int t = steps.merge(p, 1, Integer::sum);
            float[][] s = states.computeIfAbsent(p, k -> new float[][] {new float[n], new float[n]});
            float[] m = s[0];
            float[] v = s[1];
            float[] w = p.value().toArray();
            double c1 = 1 - Math.pow(beta1, t);
            double c2 = 1 - Math.pow(beta2, t);
            for (int i = 0; i < n; i++) {
                m[i] = beta1 * m[i] + (1 - beta1) * g[i];
                v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
            }
            int rows = p.shape().rank() == 2 ? p.shape().dim(0) : 1;
            if (format == Format.FOUR_BIT) {
                encodeSigned(m);
                encodeSecondMoment(v, rows, n / rows);
            } else if (format == Format.NAIVE_FOUR_BIT) {
                encodeSigned(m);
                encodeLinearWithZero(v);
            } else {
                encodeFp8(m);
                encodeFp8(v);
            }
            for (int i = 0; i < n; i++) {
                double step = (m[i] / c1) / (Math.sqrt(Math.max(0, v[i]) / c2) + epsilon);
                w[i] = (float) (w[i] * (1 - learningRate * weightDecay) - learningRate * step);
            }
            p.assign(Tensor.of(p.shape(), w));
            p.zeroGrad();
        }
    }

    /** Rounds the first moment, in place, through its blockwise signed 4-bit code. */
    private static void encodeSigned(float[] x) {
        for (int start = 0; start < x.length; start += BLOCK) {
            int end = Math.min(x.length, start + BLOCK);
            float max = 0;
            for (int i = start; i < end; i++) {
                max = Math.max(max, Math.abs(x[i]));
            }
            if (max == 0) {
                continue;
            }
            for (int i = start; i < end; i++) {
                x[i] = max * nearest(SIGNED_LEVELS, x[i] / max);
            }
        }
    }

    /** Rank-1 normalization and a zero-excluding linear 4-bit code, in place. */
    private static void encodeSecondMoment(float[] v, int rows, int cols) {
        float[] rowMax = new float[rows];
        float[] colMax = new float[cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float x = v[r * cols + c];
                rowMax[r] = Math.max(rowMax[r], x);
                colMax[c] = Math.max(colMax[c], x);
            }
        }
        if (rows == 1) {
            // vectors: blockwise maxima instead of row and column maxima
            for (int start = 0; start < cols; start += BLOCK) {
                int end = Math.min(cols, start + BLOCK);
                float max = 0;
                for (int i = start; i < end; i++) {
                    max = Math.max(max, v[i]);
                }
                for (int i = start; i < end; i++) {
                    v[i] = max == 0 ? 0 : max * linearNoZero(v[i] / max);
                }
            }
            return;
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float scale = Math.min(rowMax[r], colMax[c]);
                int i = r * cols + c;
                v[i] = scale == 0 ? 0 : scale * linearNoZero(Math.min(1f, v[i] / scale));
            }
        }
    }

    /** Blockwise absmax and 16 linear levels from 0 to the maximum, in place. */
    private static void encodeLinearWithZero(float[] v) {
        for (int start = 0; start < v.length; start += BLOCK) {
            int end = Math.min(v.length, start + BLOCK);
            float max = 0;
            for (int i = start; i < end; i++) {
                max = Math.max(max, v[i]);
            }
            for (int i = start; i < end; i++) {
                v[i] = max == 0 ? 0 : max * Math.round(v[i] / max * 15) / 15f;
            }
        }
    }

    private static float linearNoZero(float y) {
        int code = Math.max(0, Math.min(15, Math.round(y * 16) - 1));
        return (code + 1) / 16f;
    }

    private static float nearest(float[] levels, float y) {
        float best = levels[0];
        for (float l : levels) {
            if (Math.abs(l - y) < Math.abs(best - y)) {
                best = l;
            }
        }
        return best;
    }

    /** COAT: per group, expand the dynamic range by a power and round to E4M3, in place. */
    private static void encodeFp8(float[] x) {
        Fp8Format fp8 = Fp8Format.E4M3;
        // match the group's range to E4M3's normal range, so its smallest value stays representable
        double fp8Range = Math.log(fp8.max() / 0x1p-6);
        for (int start = 0; start < x.length; start += BLOCK) {
            int end = Math.min(x.length, start + BLOCK);
            float max = 0;
            float min = Float.POSITIVE_INFINITY;
            for (int i = start; i < end; i++) {
                float a = Math.abs(x[i]);
                max = Math.max(max, a);
                if (a > 0) {
                    min = Math.min(min, a);
                }
            }
            if (max == 0) {
                continue;
            }
            double range = Math.log(max / min);
            double k = range > 1e-6 ? Math.max(0.05, Math.min(8, fp8Range / range)) : 1;
            for (int i = start; i < end; i++) {
                double y = Math.pow(Math.abs(x[i]) / max, k) * fp8.max();
                double q = fp8.round((float) y) / fp8.max();
                x[i] = (float) Math.copySign(Math.pow(q, 1 / k) * max, x[i]);
            }
        }
    }

    /**
     * The bytes the packed states take: 4-bit codes plus block scales (first moment) and row and
     * column scales (second moment), or FP8 codes plus a scale and exponent per group.
     *
     * @return state bytes over every parameter seen
     */
    public long stateBytes() {
        long bytes = 0;
        for (Map.Entry<Var, float[][]> e : states.entrySet()) {
            long n = e.getValue()[0].length;
            long groups = (n + BLOCK - 1) / BLOCK;
            if (format == Format.NAIVE_FOUR_BIT) {
                bytes += n + 8 * groups;
            } else if (format == Format.FOUR_BIT) {
                var shape = e.getKey().shape();
                long secondScales = shape.rank() == 2 ? shape.dim(0) + shape.dim(1) : groups;
                bytes += n + 4 * groups + 4 * secondScales;  // two 4-bit codes per value = n bytes
            } else {
                bytes += 2 * n + 2 * 8 * groups;
            }
        }
        return bytes;
    }

    @Override
    public void forget(Var parameter) {
        states.remove(parameter);
        steps.remove(parameter);
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return "Adam " + format + " states(lr=" + learningRate + ")";
    }
}
