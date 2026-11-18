package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A mobile inverted-bottleneck block (MBConv, the EfficientNet / MobileNetV2 block) over
 * {@code [batch, channels, h, w]}: a 1x1 expansion to {@code expansion * channels} maps (SiLU),
 * a cheap depthwise {@code k x k} convolution (SiLU), optional {@link SqueezeExcitation}, a
 * linear 1x1 projection to {@code out} maps and, when shapes match, a residual connection
 * with stochastic depth. Expanding before the depthwise step and projecting linearly keeps
 * the block expressive at a fraction of a full convolution's multiply-adds.
 *
 * <p>{@link #multiplyAdds} counts the block's multiply-adds per example, for accuracy-vs-FLOPs
 * comparisons and EfficientNet's compound scaling.
 *
 * <p>Papers: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018),
 * <a href="https://arxiv.org/abs/1801.04381">arXiv:1801.04381</a>; Tan and Le, "EfficientNet:
 * Rethinking Model Scaling for Convolutional Neural Networks" (2019),
 * <a href="https://arxiv.org/abs/1905.11946">arXiv:1905.11946</a>.
 */
public final class MbConvBlock implements Layer {

    private final int channels;
    private final int out;
    private final int hidden;
    private final int kernel;
    private final float dropPath;
    private final Var expand;
    private final Var expandBias;
    private final Var depthwise;
    private final Var depthwiseBias;
    private final SqueezeExcitation se;
    private final Var project;
    private final Var projectBias;
    private final RandomGenerator random;

    /**
     * Creates the block.
     *
     * @param channels the input channels
     * @param out the output channels
     * @param expansion the expansion factor (6 in EfficientNet, 1 skips the expansion convolution)
     * @param kernel the depthwise kernel side, odd
     * @param seReduction the squeeze-excitation reduction relative to the input channels (0: no SE)
     * @param dropPath the stochastic-depth rate of the residual branch
     * @param random the source of weights and drops
     */
    public MbConvBlock(int channels, int out, int expansion, int kernel, int seReduction, float dropPath,
            RandomGenerator random) {
        this.channels = channels;
        this.out = out;
        this.hidden = channels * expansion;
        this.kernel = kernel;
        this.dropPath = dropPath;
        this.expand = expansion == 1 ? null : ClassicOps.convWeight(random, hidden, channels, 1);
        this.expandBias = expansion == 1 ? null : ClassicOps.zeros(hidden);
        this.depthwise = Var.param(Tensor.normal(Shape.of(hidden, 1, kernel, kernel), 0,
                (float) Math.sqrt(2.0 / (kernel * kernel)), random));
        this.depthwiseBias = ClassicOps.zeros(hidden);
        this.se = seReduction > 0 ? new SqueezeExcitation(hidden, Math.max(1, channels / seReduction), random) : null;
        this.project = ClassicOps.convWeight(random, out, hidden, 1);
        this.projectBias = ClassicOps.zeros(out);
        this.random = random;
    }

    /**
     * Describes the block for model builders (squeeze-excitation with reduction 4, no drop path).
     *
     * @param out the output channels
     * @param expansion the expansion factor
     * @param kernel the depthwise kernel side
     * @return the spec
     */
    public static LayerSpec spec(int out, int expansion, int kernel) {
        return ClassicOps.spec(s -> Shape.of(out, s.dim(1), s.dim(2)),
                (s, r) -> new MbConvBlock(s.dim(0), out, expansion, kernel, 4, 0f, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        ClassicOps.requireImages(input, channels, this);
        Var h = input;
        if (expand != null) {
            h = Activation.SILU.forward(ClassicOps.conv(h, expand, expandBias, 1));
        }
        h = Activation.SILU.forward(ClassicOps.conv(h, depthwise, depthwiseBias, hidden));
        if (se != null) {
            h = se.forward(h);
        }
        h = ClassicOps.conv(h, project, projectBias, 1);
        if (out == channels) {
            return input.add(DropPath.apply(h, dropPath, training, random));
        }
        return h;
    }

    /**
     * Multiply-adds per example on an {@code h x w} map.
     *
     * @param height the map height
     * @param width the map width
     * @return the count
     */
    public long multiplyAdds(int height, int width) {
        long px = (long) height * width;
        long m = (expand != null ? (long) channels * hidden : 0) + (long) hidden * kernel * kernel
                + (long) hidden * out;
        long seCost = se != null ? 2L * hidden * Math.max(1, channels / 4) : 0;
        return px * m + seCost;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(out, input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (expand != null) {
            p.add(expand);
            p.add(expandBias);
        }
        p.add(depthwise);
        p.add(depthwiseBias);
        if (se != null) {
            p.addAll(se.parameters());
        }
        p.add(project);
        p.add(projectBias);
        return p;
    }

    @Override
    public String toString() {
        return "MbConvBlock(" + channels + " -> " + hidden + " -> " + out + ", " + kernel + "x" + kernel + ")";
    }
}
