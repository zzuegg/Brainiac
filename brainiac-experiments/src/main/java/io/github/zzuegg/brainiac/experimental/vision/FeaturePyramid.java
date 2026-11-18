package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A feature pyramid network (Lin et al., 2017): a backbone's maps at strides 2, 4, 8, ... are
 * each projected to a common width by a 1x1 lateral convolution; starting from the coarsest,
 * each level is upsampled (nearest neighbour) and added to the next finer lateral, and a 3x3
 * convolution smooths every sum. Every output level then carries the coarse levels' semantics
 * at its own resolution, which is what lets one shared detection head find small objects on
 * fine levels and large ones on coarse levels.
 *
 * <p>As a {@link MultiInputLayer} it takes the backbone maps finest first and returns the
 * finest pyramid level; {@link #levels} returns them all.
 *
 * <p>Paper: Lin et al., "Feature Pyramid Networks for Object Detection" (2017),
 * <a href="https://arxiv.org/abs/1612.03144">arXiv:1612.03144</a>.
 */
public final class FeaturePyramid implements MultiInputLayer {

    private final int width;
    private final List<Var[]> laterals = new ArrayList<>();
    private final List<Var[]> smooth = new ArrayList<>();

    /**
     * Creates the pyramid.
     *
     * @param inputChannels the channels of each backbone map, finest first
     * @param width the channels of every pyramid level
     * @param random the source of initial weights
     */
    public FeaturePyramid(int[] inputChannels, int width, RandomGenerator random) {
        this.width = width;
        for (int c : inputChannels) {
            laterals.add(new Var[] {VisionOps.convWeight(random, width, c, 1), VisionOps.zeros(width)});
            smooth.add(new Var[] {VisionOps.convWeight(random, width, width, 3), VisionOps.zeros(width)});
        }
    }

    /**
     * Describes a pyramid for graph builders.
     *
     * @param width the channels of every level
     * @return the spec
     */
    public static MultiInputLayer.Spec spec(int width) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return Shape.of(width, inputs.get(0).dim(1), inputs.get(0).dim(2));
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                int[] c = new int[inputs.size()];
                for (int i = 0; i < c.length; i++) {
                    c[i] = inputs.get(i).dim(0);
                }
                return new FeaturePyramid(c, width, random);
            }
        };
    }

    /**
     * Builds every pyramid level.
     *
     * @param features the backbone maps {@code [batch, c_i, h_i, w_i]}, finest first
     * @return the levels {@code [batch, width, h_i, w_i]}, finest first
     */
    public List<Var> levels(List<Var> features) {
        int n = features.size();
        if (n != laterals.size()) {
            throw new IllegalArgumentException("expected " + laterals.size() + " maps, got " + n);
        }
        Var[] merged = new Var[n];
        for (int i = n - 1; i >= 0; i--) {
            Var f = features.get(i);
            Var lat = VisionOps.conv(f, laterals.get(i)[0], laterals.get(i)[1]);
            if (i < n - 1) {
                lat = lat.add(VisionOps.upsample(merged[i + 1], f.shape().dim(2), f.shape().dim(3)));
            }
            merged[i] = lat;
        }
        List<Var> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(VisionOps.conv(merged[i], smooth.get(i)[0], smooth.get(i)[1]));
        }
        return out;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        return levels(inputs).get(0);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(width, inputs.get(0).dim(1), inputs.get(0).dim(2));
    }

    /**
     * The multiply-adds of one image.
     *
     * @param inputChannels each map's channels, finest first
     * @param sides each map's side
     * @return the count
     */
    public long macs(int[] inputChannels, int[] sides) {
        long m = 0;
        for (int i = 0; i < sides.length; i++) {
            m += VisionOps.convMacs(inputChannels[i], width, 1, sides[i], sides[i]);
            m += VisionOps.convMacs(width, width, 3, sides[i], sides[i]);
        }
        return m;
    }

    /** @return the channels of every level */
    public int width() {
        return width;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < laterals.size(); i++) {
            p.add(laterals.get(i)[0]);
            p.add(laterals.get(i)[1]);
            p.add(smooth.get(i)[0]);
            p.add(smooth.get(i)[1]);
        }
        return p;
    }

    @Override
    public String toString() {
        return "FeaturePyramid(" + laterals.size() + " levels, width " + width + ")";
    }
}
