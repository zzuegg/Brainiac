package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A small multi-scale CNN backbone for the detection and dense-prediction heads: a stride-1
 * 3x3 stem, then one stride-2 3x3 convolution with ReLU per stage, so stage {@code i} has
 * stride {@code 2^(i+1)}. {@link #features} hands out every stage's map (fine to coarse),
 * which is what a {@link FeaturePyramid} consumes; as a plain {@link Layer} it returns the
 * coarsest map.
 */
public final class ConvBackbone implements Layer {

    private final int inputs;
    private final int[] channels;
    private final List<Var[]> convs = new ArrayList<>();

    /**
     * Creates the backbone.
     *
     * @param inputs the input channels
     * @param channels the stem's channels, then each stage's
     * @param random the source of initial weights
     */
    public ConvBackbone(int inputs, int[] channels, RandomGenerator random) {
        if (channels.length < 2) {
            throw new IllegalArgumentException("need a stem and at least one stage");
        }
        this.inputs = inputs;
        this.channels = channels.clone();
        int c = inputs;
        for (int ch : channels) {
            convs.add(new Var[] {VisionOps.convWeight(random, ch, c, 3), VisionOps.zeros(ch)});
            c = ch;
        }
    }

    /**
     * Describes a backbone for model builders.
     *
     * @param channels the stem's channels, then each stage's
     * @return the spec, mapping {@code [c, h, w]} to the coarsest map
     */
    public static LayerSpec spec(int... channels) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int s = 1 << (channels.length - 1);
                return Shape.of(channels[channels.length - 1], (input.dim(1) + s - 1) / s, (input.dim(2) + s - 1) / s);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new ConvBackbone(input.dim(0), channels, random);
            }
        };
    }

    /**
     * Runs the backbone.
     *
     * @param images {@code [batch, inputs, h, w]}
     * @return each stage's map, finest (stride 2) first
     */
    public List<Var> features(Var images) {
        List<Var> out = new ArrayList<>();
        Var x = VisionOps.conv(images, convs.get(0)[0], convs.get(0)[1]).relu();
        for (int i = 1; i < convs.size(); i++) {
            x = VisionOps.conv(x, convs.get(i)[0], convs.get(i)[1], 2, 1).relu();
            out.add(x);
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        List<Var> f = features(input);
        return f.get(f.size() - 1);
    }

    @Override
    public Shape outputShape(Shape input) {
        return spec(channels).outputShape(input);
    }

    /** @return each stage's channels, stem excluded */
    public int[] stageChannels() {
        return java.util.Arrays.copyOfRange(channels, 1, channels.length);
    }

    /**
     * The multiply-adds of one image.
     *
     * @param side the image side
     * @return the count
     */
    public long macs(int side) {
        long m = VisionOps.convMacs(inputs, channels[0], 3, side, side);
        int s = side;
        for (int i = 1; i < channels.length; i++) {
            s = (s + 1) / 2;
            m += VisionOps.convMacs(channels[i - 1], channels[i], 3, s, s);
        }
        return m;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var[] c : convs) {
            p.add(c[0]);
            p.add(c[1]);
        }
        return p;
    }

    @Override
    public String toString() {
        return "ConvBackbone(" + inputs + " -> " + java.util.Arrays.toString(channels) + ")";
    }
}
