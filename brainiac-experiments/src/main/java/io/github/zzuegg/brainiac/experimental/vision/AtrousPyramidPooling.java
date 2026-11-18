package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Atrous spatial pyramid pooling (DeepLabv3, Chen et al., 2017): parallel branches see the
 * same map at several receptive fields without downsampling it, a 1x1 convolution, 3x3
 * convolutions at several dilation rates (core {@code Conv} dilation, via
 * {@link Functional#conv}) and a global-average-pooling branch broadcast back over the map;
 * the branches (each with ReLU) are concatenated and projected by a 1x1 convolution. One
 * layer thus mixes local detail and whole-image context, which is what segmentation needs to
 * label a pixel by the object it belongs to.
 *
 * <p>Papers: Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation"
 * (2017), <a href="https://arxiv.org/abs/1706.05587">arXiv:1706.05587</a>; DeepLabv3+,
 * <a href="https://arxiv.org/abs/1802.02611">arXiv:1802.02611</a>.
 */
public final class AtrousPyramidPooling implements Layer {

    private final int inputs;
    private final int outputs;
    private final int[] rates;
    private final List<Var[]> branches = new ArrayList<>();
    private final Var pooled;
    private final Var pooledBias;
    private final Var project;
    private final Var projectBias;

    /**
     * Creates the layer.
     *
     * @param inputs the input channels
     * @param outputs the channels of every branch and of the output
     * @param rates the dilation rates of the 3x3 branches, such as {@code {1, 2, 4}}
     * @param random the source of initial weights
     */
    public AtrousPyramidPooling(int inputs, int outputs, int[] rates, RandomGenerator random) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.rates = rates.clone();
        branches.add(new Var[] {VisionOps.convWeight(random, outputs, inputs, 1), VisionOps.zeros(outputs)});
        for (int i = 0; i < rates.length; i++) {
            branches.add(new Var[] {VisionOps.convWeight(random, outputs, inputs, 3), VisionOps.zeros(outputs)});
        }
        this.pooled = VisionOps.convWeight(random, outputs, inputs, 1);
        this.pooledBias = VisionOps.zeros(outputs);
        this.project = VisionOps.convWeight(random, outputs, outputs * (rates.length + 2), 1);
        this.projectBias = VisionOps.zeros(outputs);
    }

    /**
     * Describes the layer for model builders.
     *
     * @param outputs the channels
     * @param rates the dilation rates
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int... rates) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs, input.dim(1), input.dim(2));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new AtrousPyramidPooling(input.dim(0), outputs, rates, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        int h = input.shape().dim(2);
        int w = input.shape().dim(3);
        List<Var> parts = new ArrayList<>();
        parts.add(VisionOps.conv(input, branches.get(0)[0], branches.get(0)[1]).relu());
        for (int i = 0; i < rates.length; i++) {
            Var[] b = branches.get(i + 1);
            parts.add(VisionOps.conv(input, b[0], b[1], 1, rates[i]).relu());
        }
        Var g = Functional.globalPool(input, Functional.PoolKind.AVG, true);
        parts.add(VisionOps.upsample(VisionOps.conv(g, pooled, pooledBias).relu(), h, w));
        return VisionOps.conv(Var.concat(1, parts.toArray(new Var[0])), project, projectBias).relu();
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(outputs, input.dim(1), input.dim(2));
    }

    /**
     * The multiply-adds of one map.
     *
     * @param side the map side
     * @return the count
     */
    public long macs(int side) {
        return VisionOps.convMacs(inputs, outputs, 1, side, side)
                + rates.length * VisionOps.convMacs(inputs, outputs, 3, side, side)
                + VisionOps.convMacs(inputs, outputs, 1, 1, 1)
                + VisionOps.convMacs(outputs * (rates.length + 2), outputs, 1, side, side);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var[] b : branches) {
            p.add(b[0]);
            p.add(b[1]);
        }
        p.addAll(List.of(pooled, pooledBias, project, projectBias));
        return p;
    }

    @Override
    public String toString() {
        return "AtrousPyramidPooling(" + inputs + " -> " + outputs + ", rates " + java.util.Arrays.toString(rates) + ")";
    }
}
