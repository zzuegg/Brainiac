package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A convolution whose filters are standardized on every forward pass: each filter has its
 * mean subtracted and is divided by its standard deviation before convolving. This smooths
 * the loss landscape like batch normalization does, but without any batch statistics, so it
 * works at batch size 1 and pairs with GroupNorm (the BiT / NFNet recipe). Spatial dimensions
 * keep their size ("same" padding for odd kernels) at stride 1.
 *
 * <p>Paper: Qiao et al., "Micro-Batch Training with Batch-Channel Normalization and Weight
 * Standardization" (2019), <a href="https://arxiv.org/abs/1903.10520">arXiv:1903.10520</a>.
 */
public final class WeightStandardizedConv implements Layer {

    private final int channels;
    private final int filters;
    private final int[] kernel;
    private final int[] stride;
    private final Var weight;
    private final Var bias;

    /**
     * Creates the layer.
     *
     * @param channels the input channels
     * @param filters the output channels
     * @param kernel the kernel size per spatial dimension (odd sizes keep the size)
     * @param stride the step per spatial dimension
     * @param random the random source
     */
    public WeightStandardizedConv(int channels, int filters, int[] kernel, int[] stride, RandomGenerator random) {
        this.channels = channels;
        this.filters = filters;
        this.kernel = kernel.clone();
        this.stride = stride.clone();
        int[] dims = new int[kernel.length + 2];
        dims[0] = filters;
        dims[1] = channels;
        int fanIn = channels;
        for (int i = 0; i < kernel.length; i++) {
            dims[i + 2] = kernel[i];
            fanIn *= kernel[i];
        }
        this.weight = Var.param(Tensor.normal(Shape.of(dims), 0, (float) Math.sqrt(2.0 / fanIn), random));
        this.bias = Var.param(Tensor.zeros(Shape.of(filters)));
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)} on {@code [channels, spatial...]}.
     *
     * @param filters the output channels
     * @param kernel the kernel size per spatial dimension
     * @return the spec
     */
    public static LayerSpec of(int filters, int... kernel) {
        int[] k = kernel.clone();
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] out = input.toArray();
                out[0] = filters;
                return Shape.of(out);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                int[] s = new int[k.length];
                Arrays.fill(s, 1);
                return new WeightStandardizedConv(input.dim(0), filters, k, s, random);
            }
        };
    }

    /**
     * Returns the standardized filters the convolution uses.
     *
     * @return {@code [filters, channels, kernel...]} with zero mean and unit variance per filter
     */
    public Var standardizedWeight() {
        int per = Math.toIntExact(weight.shape().size() / filters);
        Var w = weight.reshape(filters, per);
        Var c = w.sub(w.mean(1, true));
        Var std = c.square().mean(1, true).add(1e-5f).sqrt();
        return c.div(std).reshape(weight.shape().toArray());
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().dim(1) != channels) {
            throw new IllegalArgumentException(this + " expects " + channels + " channels, got " + input.shape());
        }
        int r = kernel.length;
        int[] before = new int[r];
        int[] after = new int[r];
        int[] dilation = new int[r];
        for (int i = 0; i < r; i++) {
            before[i] = (kernel[i] - 1) / 2;
            after[i] = kernel[i] - 1 - before[i];
            dilation[i] = 1;
        }
        return Functional.conv(input, standardizedWeight(), bias, stride, before, after, dilation, 1);
    }

    @Override
    public Shape outputShape(Shape input) {
        int[] out = input.toArray();
        out[0] = filters;
        for (int i = 0; i < stride.length; i++) {
            out[i + 1] = (out[i + 1] - 1) / stride[i] + 1;
        }
        return Shape.of(out);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "WeightStandardizedConv(" + channels + " -> " + filters + ", kernel " + Arrays.toString(kernel) + ")";
    }
}
