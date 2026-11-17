package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A DenseNet block over {@code [batch, channels, h, w]}: {@code layers} composite layers
 * (ReLU then a 3x3 convolution producing {@code growth} new maps, optionally preceded by a
 * 1x1 bottleneck to {@code 4 growth} maps), where each layer sees the concatenation of the
 * block input and <em>every</em> earlier layer's output. The block outputs all of them,
 * {@code channels + layers * growth} maps. Feature reuse makes DenseNets very parameter
 * efficient, and the short paths from every layer to the loss ease training.
 *
 * <p>Paper: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017),
 * <a href="https://arxiv.org/abs/1608.06993">arXiv:1608.06993</a>.
 */
public final class DenseBlock2d implements Layer {

    private final int channels;
    private final int growth;
    private final boolean bottleneck;
    private final List<Var[]> convs = new ArrayList<>();

    /**
     * Creates the block.
     *
     * @param channels the input channels
     * @param layers the composite layers
     * @param growth the maps each layer adds
     * @param bottleneck whether each layer starts with a 1x1 convolution to {@code 4 growth} maps (DenseNet-B)
     * @param random the source of initial weights
     */
    public DenseBlock2d(int channels, int layers, int growth, boolean bottleneck, RandomGenerator random) {
        this.channels = channels;
        this.growth = growth;
        this.bottleneck = bottleneck;
        for (int i = 0; i < layers; i++) {
            int in = channels + i * growth;
            if (bottleneck) {
                convs.add(new Var[] {ClassicOps.convWeight(random, 4 * growth, in, 1), ClassicOps.zeros(4 * growth),
                    ClassicOps.convWeight(random, growth, 4 * growth, 3), ClassicOps.zeros(growth)});
            } else {
                convs.add(new Var[] {ClassicOps.convWeight(random, growth, in, 3), ClassicOps.zeros(growth)});
            }
        }
    }

    /**
     * Describes the block for model builders.
     *
     * @param layers the composite layers
     * @param growth the growth rate
     * @return the spec
     */
    public static LayerSpec spec(int layers, int growth) {
        return ClassicOps.spec(s -> Shape.of(s.dim(0) + layers * growth, s.dim(1), s.dim(2)),
                (s, r) -> new DenseBlock2d(s.dim(0), layers, growth, false, r));
    }

    @Override
    public Var forward(Var input) {
        ClassicOps.requireImages(input, channels, this);
        Var features = input;
        for (Var[] w : convs) {
            Var h = features.relu();
            if (bottleneck) {
                h = ClassicOps.conv(h, w[0], w[1], 1).relu();
                h = ClassicOps.conv(h, w[2], w[3], 1);
            } else {
                h = ClassicOps.conv(h, w[0], w[1], 1);
            }
            features = Var.concat(1, features, h);
        }
        return features;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0) + convs.size() * growth, input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var[] w : convs) {
            p.addAll(List.of(w));
        }
        return p;
    }

    @Override
    public String toString() {
        return "DenseBlock2d(" + channels + " + " + convs.size() + " x " + growth + ")";
    }
}
