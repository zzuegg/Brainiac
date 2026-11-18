package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The Snake activation with a learned frequency per channel (Ziyin et al., 2020):
 * {@code x + sin^2(a x) / a}. It is periodic yet keeps a linear trend, which suits audio;
 * BigVGAN, DAC and SNAC decoders use it in place of leaky ReLU.
 *
 * <p>Input and output are {@code [channels, length]} (or any shape with channels first).
 */
public final class Snake implements Layer {

    private final Var alpha;

    /**
     * A layer from given frequencies.
     *
     * @param alpha {@code [channels]}
     */
    public Snake(Var alpha) {
        this.alpha = alpha;
    }

    /**
     * A layer with every frequency 1.
     *
     * @param channels the channels
     */
    public Snake(int channels) {
        this(Var.param(Tensor.ones(Shape.of(channels))));
    }

    /**
     * Describes the layer for a model on {@code [channels, ...]}.
     *
     * @return the spec
     */
    public static LayerSpec of() {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new Snake(input.dim(0));
            }
        };
    }

    /**
     * Applies Snake.
     *
     * @param x {@code [batch, channels, ...]}
     * @param alpha {@code [channels]}
     * @return the activations
     */
    public static Var apply(Var x, Var alpha) {
        int[] dims = new int[x.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[1] = (int) alpha.shape().size();
        Var a = alpha.reshape(dims);
        return x.add(x.mul(a).sin().square().div(a.add(1e-9f)));
    }

    @Override
    public Var forward(Var input) {
        return apply(input, alpha);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        return List.of(alpha);
    }

    @Override
    public String toString() {
        return "Snake[" + alpha.shape().size() + " channels]";
    }

    static Config write(Snake s) {
        return Config.builder().put("channels", (int) s.alpha.shape().size()).build();
    }

    static Snake read(Config c) {
        return new Snake(c.getInt("channels"));
    }
}
