package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * HiFi-GAN's residual block (Kong et al., 2020): for each dilation, leaky ReLU, a dilated
 * convolution, leaky ReLU, a plain convolution, added back to the input. Lengths stay the
 * same. Several of these with different kernels, averaged, form the "multi-receptive
 * field fusion" of HiFi-GAN vocoders after each upsampling step. Input and output are
 * {@code [channels, length]} per sample.
 */
public final class HifiGanBlock implements Layer {

    private final int channels;
    private final int kernel;
    private final int[] dilations;
    private final float slope;
    private final Var[] w1;
    private final Var[] b1;
    private final Var[] w2;
    private final Var[] b2;

    /**
     * A new block with random weights.
     *
     * @param channels the channels
     * @param kernel the kernel length (odd)
     * @param dilations the dilations, such as {@code {1, 3, 5}}
     * @param slope the leaky ReLU slope, 0.1 in HiFi-GAN
     * @param random the random source
     */
    public HifiGanBlock(int channels, int kernel, int[] dilations, float slope, SplittableRandom random) {
        this(kernel, dilations, slope, convs(channels, kernel, dilations.length, random),
                biases(channels, dilations.length), convs(channels, kernel, dilations.length, random),
                biases(channels, dilations.length));
    }

    /**
     * A block from given weights.
     *
     * @param kernel the kernel length
     * @param dilations the dilations
     * @param slope the leaky ReLU slope
     * @param w1 the dilated convolutions' filters, each {@code [channels, channels, kernel]}
     * @param b1 their biases, each {@code [channels]}
     * @param w2 the plain convolutions' filters
     * @param b2 their biases
     */
    public HifiGanBlock(int kernel, int[] dilations, float slope, Var[] w1, Var[] b1, Var[] w2, Var[] b2) {
        if (kernel % 2 == 0) {
            throw new IllegalArgumentException("HiFi-GAN blocks need an odd kernel, got " + kernel);
        }
        if (w1.length != dilations.length || b1.length != dilations.length || w2.length != dilations.length
                || b2.length != dilations.length) {
            throw new IllegalArgumentException("one pair of convolutions per dilation");
        }
        this.channels = w1[0].shape().dim(0);
        this.kernel = kernel;
        this.dilations = dilations.clone();
        this.slope = slope;
        this.w1 = w1.clone();
        this.b1 = b1.clone();
        this.w2 = w2.clone();
        this.b2 = b2.clone();
    }

    static Var[] convs(int channels, int kernel, int n, SplittableRandom random) {
        Var[] w = new Var[n];
        for (int i = 0; i < n; i++) {
            w[i] = Var.param(Tensor.normal(Shape.of(channels, channels, kernel), 0, 0.01f, random));
        }
        return w;
    }

    static Var[] biases(int channels, int n) {
        Var[] b = new Var[n];
        for (int i = 0; i < n; i++) {
            b[i] = Var.param(Tensor.zeros(Shape.of(channels)));
        }
        return b;
    }

    /** A same-length 1-d convolution. */
    static Var conv(Var x, Var w, Var b, int dilation) {
        int k = w.shape().dim(2);
        int pad = (k * dilation - dilation) / 2;
        return Functional.conv(x, w, b, new int[] {1}, new int[] {pad}, new int[] {k * dilation - dilation - pad},
                new int[] {dilation}, 1);
    }

    @Override
    public Var forward(Var input) {
        boolean unbatched = input.shape().rank() == 2;
        Var x = unbatched ? input.reshape(1, input.shape().dim(0), input.shape().dim(1)) : input;
        for (int i = 0; i < dilations.length; i++) {
            Var h = conv(x.activation(ActivationOp.LEAKY_RELU, slope), w1[i], b1[i], dilations[i]);
            h = conv(h.activation(ActivationOp.LEAKY_RELU, slope), w2[i], b2[i], 1);
            x = x.add(h);
        }
        return unbatched ? x.reshape(input.shape().toArray()) : x;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < dilations.length; i++) {
            p.addAll(List.of(w1[i], b1[i], w2[i], b2[i]));
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "HifiGanBlock[" + channels + ", kernel " + kernel + ", dilations " + Arrays.toString(dilations) + "]";
    }

    static Config write(HifiGanBlock b) {
        return Config.builder().put("channels", b.channels).put("kernel", b.kernel).put("dilations", b.dilations)
                .put("slope", b.slope).build();
    }

    static HifiGanBlock read(Config c) {
        return new HifiGanBlock(c.getInt("channels"), c.getInt("kernel"), c.getInts("dilations"), c.getFloat("slope"),
                new SplittableRandom());
    }
}
