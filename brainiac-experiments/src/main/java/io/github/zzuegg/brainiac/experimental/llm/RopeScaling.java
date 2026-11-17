package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Rotary position embeddings stretched to longer contexts than a model was trained on:
 * <ul>
 * <li>{@link Method#LINEAR}: position interpolation, every frequency divided by the factor
 *     (Chen et al. 2023, <a href="https://arxiv.org/abs/2306.15595">arXiv:2306.15595</a>);</li>
 * <li>{@link Method#NTK}: "NTK-aware" scaling, the base raised to
 *     {@code theta * s^(d / (d - 2))}, which keeps high frequencies (local order) and
 *     stretches low ones (bloc97 2023);</li>
 * <li>{@link Method#DYNAMIC_NTK}: NTK scaling whose factor grows with the current length,
 *     so short inputs are untouched;</li>
 * <li>{@link Method#YARN}: YaRN (Peng et al. 2023, <a href="https://arxiv.org/abs/2309.00071">arXiv:2309.00071</a>),
 *     interpolating per frequency band: bands that rotate many times within the original
 *     context are kept, bands that rotate less than once are divided by the factor, with a
 *     linear ramp between ({@code beta_fast}, {@code beta_slow} rotations), plus an attention
 *     temperature {@code sqrt(1/t) = 0.1 ln s + 1}.</li>
 * </ul>
 * The frequencies plug into the core: {@link #freqFactors()} gives the per-frequency divisors
 * that {@code Rope.Settings.withFreqFactors} takes, so a Llama-family model uses YaRN with a
 * one-line change. {@link #apply} rotates directly, in the half-split layout of
 * {@code Functional.rotary}.
 */
public final class RopeScaling {

    /** How positions beyond the trained context are handled. */
    public enum Method {
        /** No scaling. */
        NONE,
        /** Position interpolation. */
        LINEAR,
        /** NTK-aware base scaling. */
        NTK,
        /** NTK scaling by the current length. */
        DYNAMIC_NTK,
        /** YaRN band-wise interpolation with attention temperature. */
        YARN
    }

    private final int dim;
    private final double theta;
    private final Method method;
    private final double factor;
    private final int originalContext;
    private final double betaFast;
    private final double betaSlow;

    private RopeScaling(int dim, double theta, Method method, double factor, int originalContext, double betaFast,
            double betaSlow) {
        if (dim % 2 != 0 || factor < 1) {
            throw new IllegalArgumentException("dim must be even and factor at least 1");
        }
        this.dim = dim;
        this.theta = theta;
        this.method = method;
        this.factor = factor;
        this.originalContext = originalContext;
        this.betaFast = betaFast;
        this.betaSlow = betaSlow;
    }

    /**
     * Scaling with the usual YaRN band limits (32 and 1 rotations).
     *
     * @param dim the rotated features per head
     * @param theta the frequency base, such as 10000
     * @param method the method
     * @param factor how many times longer the target context is, at least 1
     * @param originalContext the trained context length
     * @return the scaling
     */
    public static RopeScaling of(int dim, double theta, Method method, double factor, int originalContext) {
        return new RopeScaling(dim, theta, method, factor, originalContext, 32, 1);
    }

    /**
     * @param fast bands rotating more often than this within the original context are kept
     * @param slow bands rotating less often than this are fully interpolated
     * @return a copy with these YaRN band limits
     */
    public RopeScaling bands(double fast, double slow) {
        return new RopeScaling(dim, theta, method, factor, originalContext, fast, slow);
    }

    /** @return the unscaled inverse frequencies {@code theta^(-2i/d)} */
    public double[] baseFrequencies() {
        double[] f = new double[dim / 2];
        for (int i = 0; i < f.length; i++) {
            f[i] = Math.pow(theta, -2.0 * i / dim);
        }
        return f;
    }

    /**
     * The scaled inverse frequencies for inputs of a given length (only {@link Method#DYNAMIC_NTK}
     * depends on it).
     *
     * @param length the current sequence length
     * @return one frequency per rotated pair
     */
    public double[] frequencies(int length) {
        double[] base = baseFrequencies();
        double[] f = base.clone();
        switch (method) {
            case NONE -> { }
            case LINEAR -> {
                for (int i = 0; i < f.length; i++) {
                    f[i] /= factor;
                }
            }
            case NTK, DYNAMIC_NTK -> {
                double s = method == Method.NTK ? factor
                        : Math.max(1, factor * length / (double) originalContext - (factor - 1));
                double t = theta * Math.pow(s, dim / (dim - 2.0));
                for (int i = 0; i < f.length; i++) {
                    f[i] = Math.pow(t, -2.0 * i / dim);
                }
            }
            case YARN -> {
                for (int i = 0; i < f.length; i++) {
                    double rotations = originalContext * base[i] / (2 * Math.PI);
                    double keep = Math.clamp((rotations - betaSlow) / (betaFast - betaSlow), 0, 1);
                    f[i] = (1 - keep) * base[i] / factor + keep * base[i];
                }
            }
        }
        return f;
    }

    /** @return the frequencies for the full target context */
    public double[] frequencies() {
        return frequencies((int) Math.round(originalContext * factor));
    }

    /**
     * Divisors of the base frequencies, for {@code Rope.Settings.withFreqFactors}.
     *
     * @return {@code base[i] / scaled[i]}
     */
    public float[] freqFactors() {
        double[] base = baseFrequencies();
        double[] f = frequencies();
        float[] out = new float[f.length];
        for (int i = 0; i < f.length; i++) {
            out[i] = (float) (base[i] / f[i]);
        }
        return out;
    }

    /** @return YaRN's {@code 0.1 ln s + 1} (1 for the other methods), applied to queries and keys */
    public double mscale() {
        return method == Method.YARN && factor > 1 ? 0.1 * Math.log(factor) + 1 : 1;
    }

    /**
     * Rotates queries or keys (and multiplies them by {@link #mscale()}).
     *
     * @param x {@code [..., steps, dim]}
     * @param offset the position of the first step
     * @return the rotated values
     */
    public Var apply(Var x, int offset) {
        Shape s = x.shape();
        int steps = s.dim(-2);
        if (s.dim(-1) != dim) {
            throw new IllegalArgumentException("expected " + dim + " features, got " + s);
        }
        int half = dim / 2;
        double[] f = frequencies(offset + steps);
        double m = mscale();
        float[] cos = new float[steps * half];
        float[] sin = new float[steps * half];
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < half; i++) {
                double angle = (t + offset) * f[i];
                cos[t * half + i] = (float) (m * Math.cos(angle));
                sin[t * half + i] = (float) (m * Math.sin(angle));
            }
        }
        Device device = x.value().device();
        Tensor c = Tensor.of(Shape.of(steps, half), cos, device);
        Tensor sn = Tensor.of(Shape.of(steps, half), sin, device);
        Var x1 = x.slice(-1, 0, half);
        Var x2 = x.slice(-1, half, dim);
        return Var.concat(-1, x1.mul(c).sub(x2.mul(sn)), x2.mul(c).add(x1.mul(sn)));
    }

    @Override
    public String toString() {
        return "RopeScaling(" + method + ", x" + factor + " from " + originalContext + ")";
    }
}
