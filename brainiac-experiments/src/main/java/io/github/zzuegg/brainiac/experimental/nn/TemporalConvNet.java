package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A temporal convolutional network (Bai et al., 2018, <a href="https://arxiv.org/abs/1803.01271">arXiv:1803.01271</a>):
 * a stack of residual blocks of <b>dilated causal</b> 1-D convolutions. Causal means
 * output {@code t} sees only inputs up to {@code t}; dilations doubling per block
 * ({@code 1, 2, 4, ...}) make the receptive field {@code 1 + 2 (k - 1)(2^levels - 1)}
 * grow exponentially with depth at linear cost. A strong, parallel, easy-to-train
 * alternative to recurrent networks for sequence modelling and forecasting.
 *
 * <p>Input {@code [batch, steps, features]}, output {@code [batch, steps, channels]}.
 */
public final class TemporalConvNet implements Layer {

    private final int inputs;
    private final int channels;
    private final int kernel;
    private final int levels;
    private final List<Var[]> blocks = new ArrayList<>();

    /**
     * Creates the network.
     *
     * @param inputs features per step
     * @param channels channels of every block
     * @param kernel the convolution width
     * @param levels residual blocks, block {@code i} using dilation {@code 2^i}
     * @param random the initialization source
     */
    public TemporalConvNet(int inputs, int channels, int kernel, int levels, SplittableRandom random) {
        this.inputs = inputs;
        this.channels = channels;
        this.kernel = kernel;
        this.levels = levels;
        int in = inputs;
        for (int l = 0; l < levels; l++) {
            float s1 = (float) Math.sqrt(2.0 / (kernel * in));
            float s2 = (float) Math.sqrt(2.0 / (kernel * channels));
            Var w1 = Var.param(Tensor.normal(Shape.of(kernel * in, channels), 0, s1, random));
            Var b1 = Var.param(Tensor.zeros(Shape.of(1, channels)));
            Var w2 = Var.param(Tensor.normal(Shape.of(kernel * channels, channels), 0, s2, random));
            Var b2 = Var.param(Tensor.zeros(Shape.of(1, channels)));
            Var skip = in == channels ? null
                    : Var.param(Tensor.normal(Shape.of(in, channels), 0, (float) Math.sqrt(1.0 / in), random));
            blocks.add(new Var[] {w1, b1, w2, b2, skip});
            in = channels;
        }
    }

    /**
     * @param channels channels per block
     * @param kernel the convolution width
     * @param levels residual blocks
     * @return a spec for {@code Model.sequential().add(...)} on {@code [steps, features]} samples
     */
    public static LayerSpec spec(int channels, int kernel, int levels) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), channels);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new TemporalConvNet(input.dim(-1), channels, kernel, levels, random);
            }
        };
    }

    /**
     * A causal dilated convolution: {@code y_t = sum_j x_{t - j d} W_j + b}.
     *
     * @param x {@code [batch, steps, in]}
     * @param weight {@code [kernel * in, out]}, tap {@code j} (delay {@code j d}) in rows {@code j in ..}
     * @param bias {@code [1, out]}
     * @param kernel the width
     * @param dilation the dilation
     * @return {@code [batch, steps, out]}
     */
    public static Var causalConv(Var x, Var weight, Var bias, int kernel, int dilation) {
        int steps = x.shape().dim(1);
        int pad = (kernel - 1) * dilation;
        Var padded = pad == 0 ? x : Functional.padAxis(x, 1, pad, 0, PadMode.CONSTANT, 0f);
        Var[] taps = new Var[kernel];
        for (int j = 0; j < kernel; j++) {
            int start = pad - j * dilation;
            taps[j] = padded.slice(1, start, start + steps);
        }
        Var stacked = kernel == 1 ? taps[0] : Var.concat(2, taps);
        return stacked.matmul(weight).add(bias);
    }

    /** @return how many past steps (including the current one) an output can see */
    public int receptiveField() {
        return 1 + 2 * (kernel - 1) * ((1 << levels) - 1);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != inputs) {
            throw new IllegalArgumentException("TemporalConvNet expects [batch, steps, " + inputs + "], got " + s);
        }
        Var h = input;
        for (int l = 0; l < levels; l++) {
            Var[] b = blocks.get(l);
            int d = 1 << l;
            Var y = causalConv(h, b[0], b[1], kernel, d).relu();
            y = causalConv(y, b[2], b[3], kernel, d).relu();
            Var residual = b[4] == null ? h : h.matmul(b[4]);
            h = y.add(residual).relu();
        }
        return h;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var[] b : blocks) {
            for (Var v : b) {
                if (v != null) {
                    p.add(v);
                }
            }
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return channels;
    }

    @Override
    public String toString() {
        return "TemporalConvNet(" + levels + " levels x " + channels + ", kernel " + kernel + ")";
    }
}
