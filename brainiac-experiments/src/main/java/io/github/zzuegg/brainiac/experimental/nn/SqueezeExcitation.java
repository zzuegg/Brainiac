package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Squeeze-and-excitation channel attention over {@code [batch, channels, h, w]}: squeeze each
 * channel to its global mean, pass the channel vector through a bottleneck MLP
 * ({@code channels -> channels / reduction -> channels}, ReLU then sigmoid), and rescale every
 * channel by its gate. It lets a convolution's output reweight its channels by global
 * context for a tiny cost, and is a building block of MBConv / EfficientNet.
 *
 * <p>Paper: Hu, Shen and Sun, "Squeeze-and-Excitation Networks" (CVPR 2018),
 * <a href="https://arxiv.org/abs/1709.01507">arXiv:1709.01507</a>.
 */
public final class SqueezeExcitation implements Layer {

    private final int channels;
    private final Var down;
    private final Var downBias;
    private final Var up;
    private final Var upBias;

    /**
     * Creates the layer.
     *
     * @param channels the channels
     * @param squeezed the bottleneck width
     * @param random the source of initial weights
     */
    public SqueezeExcitation(int channels, int squeezed, RandomGenerator random) {
        this.channels = channels;
        this.down = ClassicOps.matrix(random, channels, squeezed);
        this.downBias = ClassicOps.zeros(squeezed);
        this.up = ClassicOps.matrix(random, squeezed, channels);
        this.upBias = ClassicOps.zeros(channels);
    }

    /**
     * Describes the layer for model builders.
     *
     * @param reduction the channel reduction of the bottleneck, such as 4 or 16
     * @return the spec
     */
    public static LayerSpec spec(int reduction) {
        return TechniqueOps.sameShape((s, r) -> new SqueezeExcitation(s.dim(0), Math.max(1, s.dim(0) / reduction), r));
    }

    /**
     * The per-channel gates for some feature maps.
     *
     * @param input {@code [batch, channels, h, w]}
     * @return {@code [batch, channels]} gates in {@code (0, 1)}
     */
    public Var gates(Var input) {
        Var s = ClassicOps.spatialMean(input);
        return s.matmul(down).add(downBias).relu().matmul(up).add(upBias).sigmoid();
    }

    @Override
    public Var forward(Var input) {
        ClassicOps.requireImages(input, channels, this);
        int b = input.shape().dim(0);
        return input.mul(gates(input).reshape(b, channels, 1, 1));
    }

    @Override
    public List<Var> parameters() {
        return List.of(down, downBias, up, upBias);
    }

    @Override
    public String toString() {
        return "SqueezeExcitation(" + channels + ")";
    }
}
