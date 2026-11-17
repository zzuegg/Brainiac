package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A ConvNeXt block over feature maps {@code [batch, channels, height, width]}: a large
 * depthwise convolution (7x7 by default) for spatial mixing, LayerNorm over channels, an
 * inverted-bottleneck MLP ({@code channels -> 4 channels -> channels}, GELU) applied per
 * pixel, LayerScale, stochastic depth and a residual connection. It is a ResNet modernized
 * step by step with transformer design choices, matching Swin transformers with pure
 * convolutions.
 *
 * <p>Paper: Liu et al., "A ConvNet for the 2020s" (2022),
 * <a href="https://arxiv.org/abs/2201.03545">arXiv:2201.03545</a>.
 */
public final class ConvNeXtBlock implements Layer {

    private final int channels;
    private final int kernel;
    private final float dropPath;
    private final Var depthwise;
    private final Var depthwiseBias;
    private final LayerNorm norm;
    private final Var up;
    private final Var upBias;
    private final Var down;
    private final Var downBias;
    private final LayerScale scale;
    private final RandomGenerator random;

    /**
     * Creates the block.
     *
     * @param channels the channels
     * @param kernel the depthwise kernel side, odd (7 in the paper)
     * @param expansion the MLP expansion (4 in the paper)
     * @param layerScale the LayerScale start value (1e-6 in the paper; 0 to disable)
     * @param dropPath the stochastic-depth rate
     * @param random the random source
     */
    public ConvNeXtBlock(int channels, int kernel, int expansion, float layerScale, float dropPath,
            RandomGenerator random) {
        if (kernel % 2 == 0) {
            throw new IllegalArgumentException("kernel must be odd, got " + kernel);
        }
        this.channels = channels;
        this.kernel = kernel;
        this.dropPath = dropPath;
        this.depthwise = Var.param(Tensor.normal(Shape.of(channels, 1, kernel, kernel), 0,
                (float) Math.sqrt(2.0 / (kernel * kernel)), random));
        this.depthwiseBias = Var.param(Tensor.zeros(Shape.of(channels)));
        this.norm = LayerNorm.of(channels);
        this.up = TechniqueOps.weight(random, channels, expansion * channels);
        this.upBias = TechniqueOps.filled(0f, expansion * channels);
        this.down = TechniqueOps.weight(random, expansion * channels, channels);
        this.downBias = TechniqueOps.filled(0f, channels);
        this.scale = layerScale > 0 ? LayerScale.perChannel(channels, layerScale) : null;
        this.random = random;
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)} on {@code [channels, h, w]}.
     *
     * @param kernel the depthwise kernel side
     * @return the spec
     */
    public static LayerSpec spec(int kernel) {
        return TechniqueOps.sameShape((Shape in, SplittableRandom r) -> new ConvNeXtBlock(in.dim(0), kernel, 4, 1e-6f,
                0f, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 4 || input.shape().dim(1) != channels) {
            throw new IllegalArgumentException(this + " expects [batch, " + channels + ", h, w], got " + input.shape());
        }
        int p = kernel / 2;
        Var h = Functional.conv(input, depthwise, depthwiseBias, new int[] {1, 1}, new int[] {p, p}, new int[] {p, p},
                new int[] {1, 1}, channels);
        h = h.permute(0, 2, 3, 1);                                     // channels last
        h = norm.forward(h);
        h = Activation.GELU.forward(h.matmul(up).add(upBias)).matmul(down).add(downBias);
        if (scale != null) {
            h = scale.forward(h);
        }
        h = h.permute(0, 3, 1, 2);
        return input.add(DropPath.apply(h, dropPath, training, random));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(depthwise, depthwiseBias));
        p.addAll(norm.parameters());
        p.addAll(List.of(up, upBias, down, downBias));
        if (scale != null) {
            p.addAll(scale.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "ConvNeXtBlock(" + channels + ", " + kernel + "x" + kernel + ")";
    }
}
