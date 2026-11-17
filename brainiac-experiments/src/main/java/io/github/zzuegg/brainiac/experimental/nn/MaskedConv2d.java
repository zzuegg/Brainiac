package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A masked 2-D convolution, the building block of PixelCNN: the kernel is zeroed below the
 * centre row and right of the centre in the centre row, so each output pixel sees only pixels
 * above it and to its left in raster order. Mask type {@code A} also hides the centre pixel
 * (for the first layer, which must not see the pixel it predicts); type {@code B} keeps it
 * (for later layers, whose centre already only summarizes earlier pixels). Stacking them gives
 * an autoregressive image model whose likelihood is computed for all pixels in one pass.
 * Same padding, stride 1.
 *
 * <p>Paper: van den Oord et al., "Pixel Recurrent Neural Networks" (2016),
 * <a href="https://arxiv.org/abs/1601.06759">arXiv:1601.06759</a>.
 */
public final class MaskedConv2d implements Layer {

    private final int channels;
    private final int filters;
    private final int kernel;
    private final boolean typeA;
    private final Var weight;
    private final Var bias;
    private final Tensor mask;

    /**
     * Creates the layer.
     *
     * @param channels input channels
     * @param filters output channels
     * @param kernel the kernel side, odd
     * @param typeA {@code true} for mask A (centre hidden), {@code false} for mask B
     * @param random the random source
     */
    public MaskedConv2d(int channels, int filters, int kernel, boolean typeA, RandomGenerator random) {
        if (kernel % 2 == 0) {
            throw new IllegalArgumentException("kernel must be odd, got " + kernel);
        }
        this.channels = channels;
        this.filters = filters;
        this.kernel = kernel;
        this.typeA = typeA;
        this.weight = Var.param(Tensor.normal(Shape.of(filters, channels, kernel, kernel), 0,
                (float) Math.sqrt(2.0 / (channels * kernel * kernel / 2.0 + 1)), random));
        this.bias = Var.param(Tensor.zeros(Shape.of(filters)));
        float[] m = new float[filters * channels * kernel * kernel];
        int c = kernel / 2;
        for (int f = 0; f < filters * channels; f++) {
            for (int y = 0; y < kernel; y++) {
                for (int x = 0; x < kernel; x++) {
                    boolean visible = y < c || (y == c && (x < c || (!typeA && x == c)));
                    m[(f * kernel + y) * kernel + x] = visible ? 1f : 0f;
                }
            }
        }
        this.mask = Tensor.of(Shape.of(filters, channels, kernel, kernel), m);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)} on {@code [channels, h, w]}.
     *
     * @param filters output channels
     * @param kernel the kernel side
     * @param typeA whether to hide the centre pixel
     * @return the spec
     */
    public static LayerSpec spec(int filters, int kernel, boolean typeA) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(filters, input.dim(1), input.dim(2));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new MaskedConv2d(input.dim(0), filters, kernel, typeA, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        int p = kernel / 2;
        return Functional.conv(input, weight.mul(mask.to(input.value().device())), bias, new int[] {1, 1},
                new int[] {p, p}, new int[] {p, p}, new int[] {1, 1}, 1);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(filters, input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "MaskedConv2d(" + (typeA ? "A" : "B") + ", " + channels + " -> " + filters + ", " + kernel + "x"
                + kernel + ")";
    }
}
