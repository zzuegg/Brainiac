package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Stereo depth by a cost volume and soft-argmin, after GC-Net (Kendall et al., 2017). A
 * shared CNN embeds the left and right images; for every candidate disparity {@code d} the
 * unit-length left features are correlated with the right features shifted by {@code d}
 * ({@link AffineSampler#shift}) and scaled by a temperature of 10, giving a volume
 * {@code [1, disparities, h, w]} (GC-Net
 * concatenates instead; correlation, as in DispNet, is far cheaper on a CPU); two 3-D
 * convolutions regularise it into a residual on the negated correlation, the matching
 * cost, and the soft-argmin
 * {@code sum_d d softmax(-cost)_d} turns the costs into a differentiable, sub-pixel disparity.
 * The geometry (that a match lies on the same row, {@code d} pixels to the left) is built
 * in, which a network regressing disparity from the stacked pair must learn.
 *
 * <p>{@link #blockMatching} is the classical reference: the sum of absolute differences
 * over a window at every disparity, winner takes all.
 *
 * <p>As a {@link MultiInputLayer} it takes {@code [left, right]} images and returns the
 * disparity {@code [batch, 1, h, w]}.
 *
 * <p>Paper: Kendall et al., "End-to-End Learning of Geometry and Context for Deep Stereo
 * Regression" (2017), <a href="https://arxiv.org/abs/1703.04309">arXiv:1703.04309</a>.
 */
public final class StereoCostVolume implements MultiInputLayer {

    private static final float TEMPERATURE = 10f;

    private final int maxDisparity;
    private final int features;
    private final Var e1;
    private final Var e1b;
    private final Var e2;
    private final Var e2b;
    private final Var c1;
    private final Var c1b;
    private final Var c2;
    private final Var c2b;

    /**
     * Creates the network.
     *
     * @param maxDisparity the largest disparity considered
     * @param features the feature channels per image
     * @param random the source of initial weights
     */
    public StereoCostVolume(int maxDisparity, int features, RandomGenerator random) {
        this.maxDisparity = maxDisparity;
        this.features = features;
        this.e1 = VisionOps.convWeight(random, 8, 1, 3);
        this.e1b = VisionOps.zeros(8);
        this.e2 = VisionOps.convWeight(random, features, 8, 3);
        this.e2b = VisionOps.zeros(features);
        this.c1 = Var.param(Tensor.normal(Shape.of(4, 1, 3, 3, 3), 0, (float) Math.sqrt(2.0 / 27), random));
        this.c1b = VisionOps.zeros(4);
        this.c2 = Var.param(Tensor.normal(Shape.of(1, 4, 3, 3, 3), 0, (float) Math.sqrt(1.0 / (4 * 27)), random));
        this.c2b = VisionOps.zeros(1);
    }

    /** Features normalised to unit length per pixel, so correlations are cosines. */
    private Var embed(Var x) {
        Var f = VisionOps.conv(VisionOps.conv(x, e1, e1b).relu(), e2, e2b);
        return f.div(f.square().sum(1, true).add(1e-6f).sqrt());
    }

    private static Var conv3d(Var x, Var w, Var b) {
        return Functional.conv(x, w, b, new int[] {1, 1, 1}, new int[] {1, 1, 1}, new int[] {1, 1, 1},
                new int[] {1, 1, 1}, 1);
    }

    /**
     * The matching cost of every disparity.
     *
     * @param left {@code [batch, 1, h, w]}
     * @param right {@code [batch, 1, h, w]}
     * @return {@code [batch, maxDisparity + 1, h, w]}
     */
    public Var cost(Var left, Var right) {
        int b = left.shape().dim(0);
        int h = left.shape().dim(2);
        int w = left.shape().dim(3);
        Var fl = embed(left);
        Var fr = embed(right);
        Var[] slices = new Var[maxDisparity + 1];
        for (int d = 0; d <= maxDisparity; d++) {
            Var shifted = d == 0 ? fr : AffineSampler.shift(fr, d);
            slices[d] = fl.mul(shifted).sum(1, true).mul(TEMPERATURE).reshape(b, 1, 1, h, w);
        }
        Var volume = Var.concat(2, slices);
        Var c = conv3d(conv3d(volume, c1, c1b).relu(), c2, c2b);
        return c.reshape(b, maxDisparity + 1, h, w).add(volume.reshape(b, maxDisparity + 1, h, w).neg());
    }

    /**
     * The soft-argmin of a cost volume.
     *
     * @param cost {@code [batch, D + 1, h, w]}
     * @return the expected disparity {@code [batch, 1, h, w]}
     */
    public static Var softArgmin(Var cost) {
        int b = cost.shape().dim(0);
        int d = cost.shape().dim(1);
        int h = cost.shape().dim(2);
        int w = cost.shape().dim(3);
        float[] values = new float[d];
        for (int i = 0; i < d; i++) {
            values[i] = i;
        }
        Var p = Activation.SOFTMAX.forward(cost.neg().permute(0, 2, 3, 1));
        return p.mul(Tensor.of(Shape.of(d), values)).sum(3, false).reshape(b, 1, h, w);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        return softArgmin(cost(inputs.get(0), inputs.get(1)));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        Shape s = inputs.get(0);
        return Shape.of(1, s.dim(1), s.dim(2));
    }

    /**
     * Classical block matching: per left pixel the disparity whose window of absolute
     * differences to the right image is smallest.
     *
     * @param left the left image, {@code w * h}
     * @param right the right image
     * @param w the width
     * @param h the height
     * @param maxDisparity the largest disparity
     * @param radius the window radius
     * @return the disparity per left pixel
     */
    public static float[] blockMatching(float[] left, float[] right, int w, int h, int maxDisparity, int radius) {
        float[] out = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double best = Double.MAX_VALUE;
                int arg = 0;
                for (int d = 0; d <= maxDisparity; d++) {
                    double sad = 0;
                    int count = 0;
                    for (int yy = Math.max(0, y - radius); yy <= Math.min(h - 1, y + radius); yy++) {
                        for (int xx = Math.max(0, x - radius); xx <= Math.min(w - 1, x + radius); xx++) {
                            int rx = xx - d;
                            if (rx < 0) {
                                continue;
                            }
                            sad += Math.abs(left[yy * w + xx] - right[yy * w + rx]);
                            count++;
                        }
                    }
                    if (count > 0 && sad / count < best) {
                        best = sad / count;
                        arg = d;
                    }
                }
                out[y * w + x] = arg;
            }
        }
        return out;
    }

    /** @return the largest disparity */
    public int maxDisparity() {
        return maxDisparity;
    }

    @Override
    public List<Var> parameters() {
        return new ArrayList<>(List.of(e1, e1b, e2, e2b, c1, c1b, c2, c2b));
    }

    @Override
    public String toString() {
        return "StereoCostVolume(" + maxDisparity + " disparities, " + features + " features)";
    }
}
