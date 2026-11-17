package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Simulated 8-bit floating point, the two formats of NVIDIA Hopper/Blackwell and the OCP
 * standard: <b>E4M3</b> (4 exponent bits, 3 mantissa bits, bias 7, max 448, no infinities)
 * for weights and activations, and <b>E5M2</b> (bias 15, max 57344) for gradients, which need
 * range more than precision. Values are rounded to the nearest representable number (ties
 * to even, subnormals included) and saturate at the maximum; {@link #quantize(Tensor)} first
 * scales a tensor so its largest magnitude maps to the format's maximum, as FP8 training
 * recipes do per tensor.
 *
 * <p>Paper: Micikevicius et al., "FP8 Formats for Deep Learning" (2022),
 * <a href="https://arxiv.org/abs/2209.05433">arXiv:2209.05433</a>.
 */
public enum Fp8Format {

    /** 4 exponent bits, 3 mantissa bits. */
    E4M3(3, -6, 448f),
    /** 5 exponent bits, 2 mantissa bits. */
    E5M2(2, -14, 57344f);

    private final int mantissaBits;
    private final int minExponent;
    private final float max;

    Fp8Format(int mantissaBits, int minExponent, float max) {
        this.mantissaBits = mantissaBits;
        this.minExponent = minExponent;
        this.max = max;
    }

    /** @return the largest finite value */
    public float max() {
        return max;
    }

    /** @return the smallest positive (subnormal) value */
    public float minSubnormal() {
        return (float) Math.scalb(1.0, minExponent - mantissaBits);
    }

    /**
     * Rounds one value to the nearest number of this format.
     *
     * @param v the value
     * @return the rounded value; NaN stays NaN, large values saturate at {@code +-max}
     */
    public float round(float v) {
        if (Float.isNaN(v)) {
            return v;
        }
        float a = Math.abs(v);
        if (a >= max) {
            return Math.copySign(max, v);
        }
        if (a == 0) {
            return v;
        }
        int e = Math.max(Math.getExponent(a), minExponent);
        double spacing = Math.scalb(1.0, e - mantissaBits);
        double r = Math.rint(a / spacing) * spacing;
        return (float) Math.copySign(Math.min(r, max), v);
    }

    /**
     * Rounds every value of a tensor without scaling.
     *
     * @param t the values
     * @return the rounded values, same shape
     */
    public Tensor roundAll(Tensor t) {
        float[] x = t.toArray();
        for (int i = 0; i < x.length; i++) {
            x[i] = round(x[i]);
        }
        return Tensor.of(t.shape(), x, t.device());
    }

    /**
     * Scales a tensor so its largest magnitude maps to {@link #max()}, rounds it, and scales
     * it back: what a value looks like after a round trip through per-tensor-scaled FP8.
     *
     * @param t the values
     * @return the quantized values, same shape and scale
     */
    public Tensor quantize(Tensor t) {
        float amax = t.abs().max().item();
        if (amax == 0) {
            return t;
        }
        float scale = max / amax;
        return roundAll(t.mul(scale)).div(scale);
    }

    /**
     * Fake quantization for FP8 training: the forward pass sees {@link #quantize(Tensor)},
     * the gradient passes straight through.
     *
     * @param x the values
     * @return the quantized values with an identity gradient
     */
    public Var fakeQuantize(Var x) {
        Tensor q = quantize(x.value());
        return x.add(q.sub(x.value()));
    }
}
