package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.Rope;
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
 * Linear, dynamic NTK and YaRN are the core {@link Rope}'s {@code "linear"}, {@code "dynamic"}
 * and {@code "yarn"} types, which this class delegates to; it adds static NTK scaling, which
 * the core has no type for, and a stateless {@link #apply} for any {@code [..., steps, dim]}
 * input. {@link #settings()} gives the matching core settings, and {@link #freqFactors()} the
 * per-frequency divisors that {@code Rope.Settings.withFreqFactors} takes.
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

    /**
     * The core rotary settings this scaling corresponds to: {@code "default"}, {@code "linear"},
     * {@code "dynamic"} and {@code "yarn"} for {@link Method#NONE}, {@link Method#LINEAR},
     * {@link Method#DYNAMIC_NTK} and {@link Method#YARN}, and {@code "default"} with the base
     * raised to {@code theta * s^(d / (d - 2))} for {@link Method#NTK}, which the core has no
     * type for.
     *
     * @return settings for {@code new Rope(...)}
     */
    public Rope.Settings settings() {
        return switch (method) {
            case NONE -> settings("default", 1, theta);
            case LINEAR -> settings("linear", factor, theta);
            case NTK -> settings("default", 1, ntkBase(factor));
            case DYNAMIC_NTK -> settings("dynamic", factor, theta);
            case YARN -> settings("yarn", factor, theta).withYarn(betaFast, betaSlow, 0);
        };
    }

    private Rope.Settings settings(String type, double f, double base) {
        return new Rope.Settings(dim, base, type, f, 1, 1, originalContext, null);
    }

    private double ntkBase(double s) {
        return theta * Math.pow(s, dim / (dim - 2.0));
    }

    /** @return the unscaled inverse frequencies {@code theta^(-2i/d)}, as the core {@link Rope} computes them */
    public double[] baseFrequencies() {
        return new Rope(settings("default", 1, theta)).frequencies();
    }

    /**
     * The scaled inverse frequencies for inputs of a given length (only {@link Method#DYNAMIC_NTK}
     * depends on it). Every method but {@link Method#NTK} comes from the core {@link Rope}.
     *
     * @param length the current sequence length
     * @return one frequency per rotated pair
     */
    public double[] frequencies(int length) {
        if (method == Method.NTK) {
            double t = ntkBase(factor);
            double[] f = new double[dim / 2];
            for (int i = 0; i < f.length; i++) {
                f[i] = Math.pow(t, -2.0 * i / dim);
            }
            return f;
        }
        Rope rope = new Rope(settings());
        if (method == Method.DYNAMIC_NTK && length > originalContext) {
            // the core grows its base from the positions it rotates
            rope.apply(Var.of(Tensor.zeros(Shape.of(1, 1, 1, dim))), new int[][] {{length - 1}});
        }
        return rope.frequencies();
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
        return new Rope(settings()).magnitude();
    }

    /**
     * Rotates queries or keys (and multiplies them by {@link #mscale()}) with a fresh core
     * {@link Rope}, so {@link Method#DYNAMIC_NTK} scales by this call's length alone.
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
        int[] positions = new int[steps];
        for (int t = 0; t < steps; t++) {
            positions[t] = offset + t;
        }
        int rows = Math.toIntExact(s.size() / ((long) steps * dim));
        Var rotated = new Rope(settings()).apply(x.reshape(1, rows, steps, dim), new int[][] {positions});
        return rotated.reshape(s.toArray());
    }

    @Override
    public String toString() {
        return "RopeScaling(" + method + ", x" + factor + " from " + originalContext + ")";
    }
}
