package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A normalization whose scale and offset come from a style vector: adaptive instance
 * normalization (AdaIN, Huang and Belongie, 2017) or adaptive layer normalization, as
 * StyleTTS2 and Kokoro condition their prosody predictors and decoders on a speaker's
 * style. A linear map of the style gives {@code gamma} and {@code beta} per channel, and the
 * output is {@code (1 + gamma) * norm(x) + beta}.
 *
 * <p>Inputs are a sequence {@code [channels, length]} and a style {@code [style]} per
 * sample; the output has the sequence's shape. Wire it with
 * {@code Node.merge(StyleNorm.instance(styleDim), style)}.
 */
public final class StyleNorm implements MultiInputLayer {

    /** What is normalized. */
    public enum Kind {
        /** each channel over the length (AdaIN) */
        INSTANCE,
        /** each position over the channels (adaptive layer norm) */
        LAYER
    }

    private final Kind kind;
    private final float eps;
    private final Var weight;
    private final Var bias;

    /**
     * A layer with random weights.
     *
     * @param kind what is normalized
     * @param channels the sequence's channels
     * @param style the style width
     * @param random the random source
     */
    public StyleNorm(Kind kind, int channels, int style, SplittableRandom random) {
        this(kind, 1e-5f, Var.param(Init.XAVIER.create(Shape.of(style, 2 * channels), style, 2 * channels,
                random)), Var.param(Tensor.zeros(Shape.of(2 * channels))));
    }

    /**
     * A layer from given weights.
     *
     * @param kind what is normalized
     * @param eps added to the variance
     * @param weight the style projection {@code [style, 2 channels]}: gammas, then betas
     * @param bias its bias {@code [2 channels]}
     */
    public StyleNorm(Kind kind, float eps, Var weight, Var bias) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.eps = eps;
        this.weight = weight;
        this.bias = bias;
        if (weight.shape().rank() != 2 || weight.shape().dim(1) % 2 != 0) {
            throw new IllegalArgumentException("a style norm's weight is [style, 2 channels], got " + weight.shape());
        }
    }

    /**
     * Describes an adaptive instance norm.
     *
     * @param style the style width
     * @return the spec
     */
    public static MultiInputLayer.Spec instance(int style) {
        return spec(Kind.INSTANCE, style);
    }

    /**
     * Describes an adaptive layer norm.
     *
     * @param style the style width
     * @return the spec
     */
    public static MultiInputLayer.Spec layer(int style) {
        return spec(Kind.LAYER, style);
    }

    private static MultiInputLayer.Spec spec(Kind kind, int style) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return check(inputs, style);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                check(inputs, style);
                return new StyleNorm(kind, inputs.get(0).dim(0), style, random);
            }
        };
    }

    static Shape check(List<Shape> inputs, int style) {
        if (inputs.size() != 2 || inputs.get(0).rank() != 2 || inputs.get(1).size() != style) {
            throw new IllegalArgumentException("a style-conditioned layer takes a sequence [channels, length] and a"
                    + " style [" + style + "], got " + inputs);
        }
        return inputs.get(0);
    }

    /** @return the number of channels */
    public int channels() {
        return weight.shape().dim(1) / 2;
    }

    /**
     * Normalizes and modulates.
     *
     * @param x {@code [batch, channels, length]}
     * @param style {@code [batch, style]}
     * @return {@code [batch, channels, length]}
     */
    public Var forward(Var x, Var style) {
        int c = channels();
        int n = x.shape().dim(0);
        Var h = style.reshape(n, weight.shape().dim(0)).matmul(weight).add(bias);
        Var gamma = h.slice(1, 0, c).reshape(n, c, 1);
        Var beta = h.slice(1, c, 2 * c).reshape(n, c, 1);
        int axis = kind == Kind.INSTANCE ? 2 : 1;
        Var centred = x.sub(x.mean(axis, true));
        Var normed = centred.div(centred.square().mean(axis, true).add(eps).sqrt());
        return normed.mul(gamma.add(1)).add(beta);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException(this + " takes a sequence and a style, got " + inputs.size()
                    + " inputs");
        }
        return forward(inputs.get(0), inputs.get(1));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return check(inputs, weight.shape().dim(0));
    }

    @Override
    public Shape outputShape(Shape input) {
        throw new IllegalStateException("a style norm needs a style; wire it with Node.merge(spec, style)");
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "StyleNorm[" + kind.name().toLowerCase(java.util.Locale.ROOT) + ", " + channels() + " channels, style "
                + weight.shape().dim(0) + "]";
    }

    static Config write(StyleNorm n) {
        return Config.builder().put("kind", n.kind.name()).put("eps", n.eps).put("channels", n.channels())
                .put("style", n.weight.shape().dim(0)).build();
    }

    static StyleNorm read(Config c) {
        int ch = c.getInt("channels");
        int st = c.getInt("style");
        return new StyleNorm(Kind.valueOf(c.getString("kind")), c.getFloat("eps"),
                Var.param(Tensor.zeros(Shape.of(st, 2 * ch))), Var.param(Tensor.zeros(Shape.of(2 * ch))));
    }
}
