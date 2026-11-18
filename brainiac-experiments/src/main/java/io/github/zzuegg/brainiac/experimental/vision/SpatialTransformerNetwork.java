package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A spatial transformer (Jaderberg et al., 2015): a small localisation network regresses the
 * six parameters of an affine transform from the input image, and {@link AffineSampler}
 * builds the sampling grid and bilinearly resamples the image with it, so whatever follows
 * sees a canonicalised input. The last layer of the localisation network starts at zero
 * weights and the identity bias, so training starts from "no warp", and the whole module is
 * trained end to end from the downstream loss alone.
 *
 * <p>Input and output {@code [batch, channels, h, w]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1506.02025">arXiv:1506.02025</a>.
 */
public final class SpatialTransformerNetwork implements Layer {

    private final Var conv1;
    private final Var bias1;
    private final Var conv2;
    private final Var bias2;
    private final Dense hidden;
    private final Dense affine;

    /**
     * Creates the module.
     *
     * @param channels the image channels
     * @param width the channels of the localisation network
     * @param random the source of initial weights
     */
    public SpatialTransformerNetwork(int channels, int width, RandomGenerator random) {
        this.conv1 = VisionOps.convWeight(random, width, channels, 3);
        this.bias1 = VisionOps.zeros(width);
        this.conv2 = VisionOps.convWeight(random, 2 * width, width, 3);
        this.bias2 = VisionOps.zeros(2 * width);
        this.hidden = new Dense(2 * width, 32, Activation.RELU, Init.HE, random);
        this.affine = new Dense(32, 6, Activation.IDENTITY, Tensor.zeros(Shape.of(32, 6)),
                Tensor.of(Shape.of(6), new float[] {1, 0, 0, 0, 1, 0}));
    }

    /**
     * The affine transforms the localisation network predicts.
     *
     * @param images {@code [batch, channels, h, w]}
     * @return {@code [batch, 2, 3]}
     */
    public Var transform(Var images) {
        Var h = VisionOps.maxPool2(VisionOps.conv(images, conv1, bias1).relu());
        h = VisionOps.conv(h, conv2, bias2).relu();
        h = Functional.globalPool(h, Functional.PoolKind.AVG, false);
        return affine.forward(hidden.forward(h)).reshape(images.shape().dim(0), 2, 3);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        return AffineSampler.sample(input, AffineSampler.affineGrid(transform(input), s.dim(2), s.dim(3)));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(conv1, bias1, conv2, bias2));
        p.addAll(hidden.parameters());
        p.addAll(affine.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "SpatialTransformerNetwork";
    }
}
