package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Three ways to read keypoint coordinates from a CNN feature map:
 *
 * <ul>
 *   <li>{@link Mode#DIRECT}: DeepPose (Toshev and Szegedy, 2014), a dense layer on the
 *       flattened map regresses the {@code 2J} coordinates;</li>
 *   <li>{@link Mode#HEATMAP_ARGMAX}: one heatmap per joint trained against Gaussians
 *       ({@link CenterNetHead#splat}) with CenterNet's penalty-reduced focal loss
 *       ({@link CenterNetHead#focalLoss}), read out by the arg-max cell, so the estimate is
 *       quantised to the map's stride;</li>
 *   <li>{@link Mode#INTEGRAL}: integral regression (Sun et al., 2018), the expected
 *       coordinate of the softmax-normalised heatmap ({@link #softArgmax}), differentiable
 *       and sub-pixel, trained end to end with an L1 loss on coordinates.</li>
 * </ul>
 *
 * As a {@link Layer} it maps features {@code [batch, c, h, w]} to coordinates
 * {@code [batch, J, 2]} in image pixels.
 *
 * <p>Papers: Toshev and Szegedy, "DeepPose" (2014),
 * <a href="https://arxiv.org/abs/1312.4659">arXiv:1312.4659</a>; Sun et al., "Integral Human
 * Pose Regression" (2018), <a href="https://arxiv.org/abs/1711.08229">arXiv:1711.08229</a>.
 */
public final class KeypointHeads implements Layer {

    /** The readout. */
    public enum Mode {
        /** Dense regression of the coordinates. */
        DIRECT,
        /** Gaussian heatmaps read out by arg-max. */
        HEATMAP_ARGMAX,
        /** Heatmaps read out by soft-argmax, trained on coordinates. */
        INTEGRAL
    }

    private final Mode mode;
    private final int joints;
    private final int stride;
    private final Var weight;
    private final Var bias;

    /**
     * Creates a head.
     *
     * @param mode the readout
     * @param channels the feature channels
     * @param side the feature map side
     * @param joints the number of joints
     * @param stride image pixels per feature cell
     * @param random the source of initial weights
     */
    public KeypointHeads(Mode mode, int channels, int side, int joints, int stride, RandomGenerator random) {
        this.mode = mode;
        this.joints = joints;
        this.stride = stride;
        if (mode == Mode.DIRECT) {
            int in = channels * side * side;
            this.weight = Var.param(Tensor.normal(Shape.of(in, 2 * joints), 0, (float) Math.sqrt(1.0 / in), random));
            float[] b = new float[2 * joints];
            java.util.Arrays.fill(b, side * stride / 2f);
            this.bias = Var.param(Tensor.of(Shape.of(2 * joints), b));
        } else {
            this.weight = VisionOps.convWeight(random, joints, channels, 1);
            this.bias = VisionOps.zeros(joints);
        }
    }

    /** @return the readout */
    public Mode mode() {
        return mode;
    }

    /**
     * The heatmaps (not for {@link Mode#DIRECT}).
     *
     * @param features {@code [batch, c, h, w]}
     * @return {@code [batch, J, h, w]} logits
     */
    public Var heatmaps(Var features) {
        return VisionOps.conv(features, weight, bias);
    }

    /**
     * The expected cell coordinates of softmax-normalised heatmaps, scaled to pixels.
     *
     * @param heat {@code [batch, J, h, w]} logits
     * @param stride image pixels per cell
     * @return {@code [batch, J, 2]} as {@code (x, y)} with cell centres at
     *     {@code (i + 0.5) stride}
     */
    public static Var softArgmax(Var heat, int stride) {
        int b = heat.shape().dim(0);
        int j = heat.shape().dim(1);
        int h = heat.shape().dim(2);
        int w = heat.shape().dim(3);
        Var p = Activation.SOFTMAX.forward(heat.reshape(b, j, h * w));
        float[] coords = new float[h * w * 2];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                coords[(y * w + x) * 2] = (x + 0.5f) * stride;
                coords[(y * w + x) * 2 + 1] = (y + 0.5f) * stride;
            }
        }
        return p.matmul(Var.of(Tensor.of(Shape.of(h * w, 2), coords)));
    }

    /**
     * Gaussian heatmap targets.
     *
     * @param joints per image the {@code 2J} joint coordinates in pixels
     * @param side the map side
     * @param stride image pixels per cell
     * @param sigma the Gaussian's standard deviation in cells
     * @return {@code [batch, J, side, side]}
     */
    public static Tensor heatmapTargets(float[][] joints, int side, int stride, double sigma) {
        int b = joints.length;
        int j = joints[0].length / 2;
        float[] out = new float[b * j * side * side];
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < j; k++) {
                float[] plane = new float[side * side];
                int cx = Math.min(side - 1, (int) (joints[i][2 * k] / stride));
                int cy = Math.min(side - 1, (int) (joints[i][2 * k + 1] / stride));
                CenterNetHead.splat(plane, side, side, cx, cy, sigma);
                System.arraycopy(plane, 0, out, (i * j + k) * side * side, side * side);
            }
        }
        return Tensor.of(Shape.of(b, j, side, side), out);
    }

    @Override
    public Var forward(Var features) {
        int b = features.shape().dim(0);
        if (mode == Mode.DIRECT) {
            return features.reshape(b, (int) (features.value().size() / b)).matmul(weight).add(bias)
                    .reshape(b, joints, 2);
        }
        if (mode == Mode.INTEGRAL) {
            return softArgmax(heatmaps(features), stride);
        }
        Tensor heat = heatmaps(features).value();
        int h = heat.shape().dim(2);
        int w = heat.shape().dim(3);
        float[] v = heat.toArray();
        float[] out = new float[b * joints * 2];
        for (int i = 0; i < b * joints; i++) {
            int arg = 0;
            for (int p = 1; p < h * w; p++) {
                arg = v[i * h * w + p] > v[i * h * w + arg] ? p : arg;
            }
            out[2 * i] = (arg % w + 0.5f) * stride;
            out[2 * i + 1] = (arg / w + 0.5f) * stride;
        }
        return Var.of(Tensor.of(Shape.of(b, joints, 2), out));
    }

    /**
     * The training loss of the head's mode: L1 on coordinates for direct and integral
     * regression, the penalty-reduced focal loss against Gaussian targets for arg-max
     * heatmaps.
     *
     * @param features {@code [batch, c, h, w]}
     * @param joints per image the {@code 2J} joint coordinates in pixels
     * @return the loss
     */
    public Var loss(Var features, float[][] joints) {
        int b = joints.length;
        if (mode == Mode.HEATMAP_ARGMAX) {
            Var heat = heatmaps(features);
            int side = heat.shape().dim(2);
            return CenterNetHead.focalLoss(heat, heatmapTargets(joints, side, stride, 0.7));
        }
        float[] t = new float[b * this.joints * 2];
        for (int i = 0; i < b; i++) {
            System.arraycopy(joints[i], 0, t, i * this.joints * 2, this.joints * 2);
        }
        return forward(features).sub(Tensor.of(Shape.of(b, this.joints, 2), t)).abs().mean();
    }

    /**
     * The percentage of correct keypoints within a pixel threshold, and the mean joint
     * error.
     *
     * @param predicted {@code [batch, J, 2]} flattened
     * @param truth per image the {@code 2J} true coordinates
     * @param threshold the distance counted correct
     * @return {@code {PCK, mean error}}
     */
    public static double[] pck(float[] predicted, float[][] truth, double threshold) {
        int correct = 0;
        double err = 0;
        int n = 0;
        for (int i = 0; i < truth.length; i++) {
            for (int k = 0; k < truth[i].length / 2; k++) {
                int o = i * truth[i].length + 2 * k;
                double d = Math.hypot(predicted[o] - truth[i][2 * k], predicted[o + 1] - truth[i][2 * k + 1]);
                correct += d <= threshold ? 1 : 0;
                err += d;
                n++;
            }
        }
        return new double[] {(double) correct / n, err / n};
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(joints, 2);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "KeypointHeads(" + mode + ", " + joints + " joints)";
    }
}
