package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The residual block of style-conditioned HiFi-GAN and iSTFTNet vocoders (StyleTTS2,
 * Kokoro): HiFi-GAN's multi-receptive-field block with each leaky ReLU replaced by adaptive
 * instance norm and a Snake activation {@code x + sin^2(a x) / a} with a learned frequency
 * {@code a} per channel (Ziyin et al., 2020), which suits periodic signals. Lengths stay the
 * same.
 *
 * <p>Inputs are a sequence {@code [channels, length]} and a style {@code [style]} per sample.
 * Wire it with {@code Node.merge(AdainSnakeBlock.of(kernel, style, 1, 3, 5), style)}.
 */
public final class AdainSnakeBlock implements MultiInputLayer {

    private final int[] dilations;
    private final StyleNorm[] norm1;
    private final StyleNorm[] norm2;
    private final Var[] alpha1;
    private final Var[] alpha2;
    private final Var[] w1;
    private final Var[] b1;
    private final Var[] w2;
    private final Var[] b2;

    /**
     * A block with random weights.
     *
     * @param channels the channels
     * @param kernel the kernel length (odd)
     * @param style the style width
     * @param dilations the dilations, such as {@code {1, 3, 5}}
     * @param random the random source
     */
    public AdainSnakeBlock(int channels, int kernel, int style, int[] dilations, SplittableRandom random) {
        this(dilations, norms(channels, style, dilations.length, random), norms(channels, style, dilations.length,
                random), ones(channels, dilations.length), ones(channels, dilations.length),
                HifiGanBlock.convs(channels, kernel, dilations.length, random),
                HifiGanBlock.biases(channels, dilations.length),
                HifiGanBlock.convs(channels, kernel, dilations.length, random),
                HifiGanBlock.biases(channels, dilations.length));
    }

    /**
     * A block from given weights; each array has one entry per dilation.
     *
     * @param dilations the dilations of the first convolution of each pair
     * @param norm1 the norms before the dilated convolutions
     * @param norm2 the norms before the plain convolutions
     * @param alpha1 the Snake frequencies before the dilated convolutions, each {@code [channels]}
     * @param alpha2 the Snake frequencies before the plain convolutions
     * @param w1 the dilated convolutions {@code [channels, channels, kernel]}
     * @param b1 their biases
     * @param w2 the plain convolutions
     * @param b2 their biases
     */
    public AdainSnakeBlock(int[] dilations, StyleNorm[] norm1, StyleNorm[] norm2, Var[] alpha1, Var[] alpha2,
            Var[] w1, Var[] b1, Var[] w2, Var[] b2) {
        int n = dilations.length;
        for (Object[] a : new Object[][] {norm1, norm2, alpha1, alpha2, w1, b1, w2, b2}) {
            if (a.length != n) {
                throw new IllegalArgumentException("one entry per dilation");
            }
        }
        if (w1[0].shape().dim(2) % 2 == 0) {
            throw new IllegalArgumentException("AdaIN Snake blocks need an odd kernel, got " + w1[0].shape());
        }
        this.dilations = dilations.clone();
        this.norm1 = norm1.clone();
        this.norm2 = norm2.clone();
        this.alpha1 = alpha1.clone();
        this.alpha2 = alpha2.clone();
        this.w1 = w1.clone();
        this.b1 = b1.clone();
        this.w2 = w2.clone();
        this.b2 = b2.clone();
    }

    private static StyleNorm[] norms(int channels, int style, int n, SplittableRandom random) {
        StyleNorm[] out = new StyleNorm[n];
        for (int i = 0; i < n; i++) {
            out[i] = new StyleNorm(StyleNorm.Kind.INSTANCE, channels, style, random);
        }
        return out;
    }

    private static Var[] ones(int channels, int n) {
        Var[] out = new Var[n];
        for (int i = 0; i < n; i++) {
            out[i] = Var.param(Tensor.ones(Shape.of(channels)));
        }
        return out;
    }

    /**
     * Describes a block.
     *
     * @param kernel the kernel length
     * @param style the style width
     * @param dilations the dilations
     * @return the spec
     */
    public static MultiInputLayer.Spec of(int kernel, int style, int... dilations) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return StyleNorm.check(inputs, style);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new AdainSnakeBlock(StyleNorm.check(inputs, style).dim(0), kernel, style, dilations, random);
            }
        };
    }

    /**
     * The Snake activation {@code x + sin^2(a x) / a}, with one frequency per channel.
     *
     * @param x {@code [batch, channels, length]}
     * @param alpha {@code [channels]}
     * @return the activations
     */
    public static Var snake(Var x, Var alpha) {
        Var a = alpha.reshape(1, (int) alpha.shape().size(), 1);
        return x.add(x.mul(a).sin().square().div(a));
    }

    /**
     * Runs the block.
     *
     * @param x {@code [batch, channels, length]}
     * @param style {@code [batch, style]}
     * @return {@code [batch, channels, length]}
     */
    public Var forward(Var x, Var style) {
        for (int i = 0; i < dilations.length; i++) {
            Var h = snake(norm1[i].forward(x, style), alpha1[i]);
            h = HifiGanBlock.conv(h, w1[i], b1[i], dilations[i]);
            h = snake(norm2[i].forward(h, style), alpha2[i]);
            h = HifiGanBlock.conv(h, w2[i], b2[i], 1);
            x = x.add(h);
        }
        return x;
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
        return StyleNorm.check(inputs, style());
    }

    @Override
    public Shape outputShape(Shape input) {
        throw new IllegalStateException("an AdaIN Snake block needs a style; wire it with Node.merge(spec, style)");
    }

    private int style() {
        return norm1[0].parameters().get(0).shape().dim(0);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < dilations.length; i++) {
            p.addAll(norm1[i].parameters());
            p.addAll(norm2[i].parameters());
            p.addAll(List.of(alpha1[i], alpha2[i], w1[i], b1[i], w2[i], b2[i]));
        }
        return p;
    }

    @Override
    public String toString() {
        return "AdainSnakeBlock[" + w1[0].shape().dim(0) + ", kernel " + w1[0].shape().dim(2) + ", dilations "
                + Arrays.toString(dilations) + "]";
    }

    static Config write(AdainSnakeBlock b) {
        return Config.builder().put("channels", b.w1[0].shape().dim(0)).put("kernel", b.w1[0].shape().dim(2))
                .put("style", b.style()).put("dilations", b.dilations).build();
    }

    static AdainSnakeBlock read(Config c) {
        return new AdainSnakeBlock(c.getInt("channels"), c.getInt("kernel"), c.getInt("style"), c.getInts("dilations"),
                new SplittableRandom());
    }
}
