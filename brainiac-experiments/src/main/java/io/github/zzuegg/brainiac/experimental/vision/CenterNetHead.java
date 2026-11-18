package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * The CenterNet head (Zhou et al., 2019), an anchor-free detector that treats objects as
 * points: on a stride-{@code s} feature map it predicts a per-class heatmap of object
 * centres, a width and height and a sub-pixel centre offset. Heatmap targets are Gaussians
 * splatted at the centres ({@link #splat}), trained with the penalty-reduced focal loss
 * ({@link #focalLoss}): {@code -(1-p)^2 log p} at a centre and
 * {@code -(1-y)^4 p^2 log(1-p)} elsewhere, so pixels near a centre are punished less for
 * firing. Inference is a 3x3 max-pool peak extraction ({@link #peaks}), so neither anchors
 * nor non-maximum suppression are needed. Two objects of one class whose centres fall in
 * the same cell collapse into one peak, the method's known weakness on crowded scenes.
 *
 * <p>The heatmap helpers are static so other heatmap heads reuse them (keypoints in
 * {@link KeypointHeads}).
 *
 * <p>Paper: Zhou, Wang and Krähenbühl, "Objects as Points" (2019),
 * <a href="https://arxiv.org/abs/1904.07850">arXiv:1904.07850</a>.
 */
public final class CenterNetHead implements Layer {

    private final int inputs;
    private final int classes;
    private final Var hidden;
    private final Var hiddenBias;
    private final Var out;
    private final Var outBias;

    /**
     * Creates the head.
     *
     * @param inputs the feature channels
     * @param classes the object classes
     * @param width the hidden channels
     * @param random the source of initial weights
     */
    public CenterNetHead(int inputs, int classes, int width, RandomGenerator random) {
        this.inputs = inputs;
        this.classes = classes;
        this.hidden = VisionOps.convWeight(random, width, inputs, 3);
        this.hiddenBias = VisionOps.zeros(width);
        this.out = Var.param(Tensor.normal(Shape.of(classes + 4, width, 1, 1), 0, 0.01f, random));
        float[] b = new float[classes + 4];
        for (int c = 0; c < classes; c++) {
            b[c] = -2.19f;
        }
        this.outBias = Var.param(Tensor.of(Shape.of(classes + 4), b));
    }

    /**
     * Describes the head for model builders.
     *
     * @param classes the object classes
     * @param width the hidden channels
     * @return the spec
     */
    public static LayerSpec spec(int classes, int width) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(classes + 4, input.dim(1), input.dim(2));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new CenterNetHead(input.dim(0), classes, width, random);
            }
        };
    }

    /**
     * Runs the head.
     *
     * @param features {@code [batch, inputs, h, w]}
     * @return {@code [batch, classes + 4, h, w]}: heatmap logits, then width, height, x
     *     offset and y offset
     */
    @Override
    public Var forward(Var features) {
        return VisionOps.conv(VisionOps.conv(features, hidden, hiddenBias).relu(), out, outBias);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(classes + 4, input.dim(1), input.dim(2));
    }

    /**
     * Splats a Gaussian peak into a heatmap, keeping the maximum where peaks overlap.
     *
     * @param heat the heatmap, {@code h * w}, modified
     * @param w the map width
     * @param h the map height
     * @param cx the peak column
     * @param cy the peak row
     * @param sigma the Gaussian's standard deviation in cells
     */
    public static void splat(float[] heat, int w, int h, int cx, int cy, double sigma) {
        int r = (int) Math.ceil(3 * sigma);
        for (int y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++) {
            for (int x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) {
                double v = Math.exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * sigma * sigma));
                heat[y * w + x] = (float) Math.max(heat[y * w + x], v);
            }
        }
    }

    /**
     * The penalty-reduced pixel-wise focal loss of CenterNet (after CornerNet), normalised by
     * the number of centres.
     *
     * @param logits heatmap logits, any shape
     * @param heat Gaussian targets of the same shape, exactly 1 at centres
     * @return the loss, a scalar
     */
    public static Var focalLoss(Var logits, Tensor heat) {
        float[] y = heat.toArray();
        float[] pos = new float[y.length];
        float[] negW = new float[y.length];
        int centres = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] >= 1f) {
                pos[i] = 1;
                centres++;
            } else {
                negW[i] = (float) Math.pow(1 - y[i], 4);
            }
        }
        Var p = logits.sigmoid().clamp(1e-4f, 1 - 1e-4f);
        Var posTerm = p.neg().add(1).square().mul(p.log()).mul(Tensor.of(heat.shape(), pos));
        Var negTerm = p.square().mul(p.neg().add(1).log()).mul(Tensor.of(heat.shape(), negW));
        return posTerm.add(negTerm).sum().mul(-1f / Math.max(1, centres));
    }

    /**
     * The local maxima of a heatmap: cells that equal the maximum of their 3x3 window, best
     * first.
     *
     * @param heat {@code h * w} scores
     * @param w the width
     * @param h the height
     * @param k the most peaks to return
     * @param threshold the lowest score kept
     * @return the cell indices of the peaks, best first
     */
    public static int[] peaks(float[] heat, int w, int h, int k, float threshold) {
        List<Integer> found = new ArrayList<>();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float v = heat[y * w + x];
                if (v < threshold) {
                    continue;
                }
                boolean max = true;
                for (int dy = -1; dy <= 1 && max; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if (yy >= 0 && yy < h && xx >= 0 && xx < w && heat[yy * w + xx] > v) {
                            max = false;
                            break;
                        }
                    }
                }
                if (max) {
                    found.add(y * w + x);
                }
            }
        }
        found.sort((a, b) -> Float.compare(heat[b], heat[a]));
        int n = Math.min(k, found.size());
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = found.get(i);
        }
        return out;
    }

    /**
     * The training loss of a batch: focal heatmap loss plus 0.1 times the L1 size loss plus
     * the L1 offset loss, the last two at the centres only.
     *
     * @param output the head's output {@code [batch, classes + 4, h, w]}
     * @param stride image pixels per cell
     * @param boxes per image the boxes, {@code 4 k} values
     * @param labels per image the classes
     * @return the loss
     */
    public Var loss(Var output, int stride, float[][] boxes, int[][] labels) {
        int b = output.shape().dim(0);
        int h = output.shape().dim(2);
        int w = output.shape().dim(3);
        float[] heat = new float[b * classes * h * w];
        List<Integer> centres = new ArrayList<>();
        List<float[]> targets = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < labels[i].length; j++) {
                float x1 = boxes[i][4 * j] / stride;
                float y1 = boxes[i][4 * j + 1] / stride;
                float x2 = boxes[i][4 * j + 2] / stride;
                float y2 = boxes[i][4 * j + 3] / stride;
                float cx = (x1 + x2) / 2;
                float cy = (y1 + y2) / 2;
                int ix = Math.min(w - 1, (int) cx);
                int iy = Math.min(h - 1, (int) cy);
                double radius = Math.max(0, Math.floor(0.3 * Math.min(x2 - x1, y2 - y1)));
                double sigma = (2 * radius + 1) / 6;
                float[] plane = new float[h * w];
                splat(plane, w, h, ix, iy, sigma);
                int base = (i * classes + labels[i][j]) * h * w;
                for (int p = 0; p < h * w; p++) {
                    heat[base + p] = Math.max(heat[base + p], plane[p]);
                }
                centres.add((i * h + iy) * w + ix);
                targets.add(new float[] {x2 - x1, y2 - y1, cx - ix, cy - iy});
            }
        }
        Var loss = focalLoss(output.slice(1, 0, classes), Tensor.of(Shape.of(b, classes, h, w), heat));
        if (!centres.isEmpty()) {
            int p = centres.size();
            float[] idx = new float[p];
            float[] t = new float[4 * p];
            for (int i = 0; i < p; i++) {
                idx[i] = centres.get(i);
                System.arraycopy(targets.get(i), 0, t, 4 * i, 4);
            }
            Var reg = output.slice(1, classes, classes + 4).permute(0, 2, 3, 1).reshape(b * h * w, 4)
                    .gather(0, Tensor.of(Shape.of(p), idx));
            Var diff = reg.sub(Tensor.of(Shape.of(p, 4), t)).abs();
            loss = loss.add(diff.slice(1, 0, 2).mean().mul(0.1f)).add(diff.slice(1, 2, 4).mean());
        }
        return loss;
    }

    /**
     * Decodes detections from the head's output.
     *
     * @param output {@code [batch, classes + 4, h, w]}
     * @param stride image pixels per cell
     * @param maxObjects the most detections per image
     * @param firstImage the image index of the first image
     * @return the detections, peaks above 0.05
     */
    public List<DetectionMetrics.Detection> decode(Tensor output, int stride, int maxObjects, int firstImage) {
        int b = output.shape().dim(0);
        int h = output.shape().dim(2);
        int w = output.shape().dim(3);
        int hw = h * w;
        float[] o = output.toArray();
        List<DetectionMetrics.Detection> dets = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            List<DetectionMetrics.Detection> image = new ArrayList<>();
            for (int c = 0; c < classes; c++) {
                float[] heat = new float[hw];
                int base = (i * (classes + 4) + c) * hw;
                for (int p = 0; p < hw; p++) {
                    heat[p] = (float) (1 / (1 + Math.exp(-o[base + p])));
                }
                for (int p : peaks(heat, w, h, maxObjects, 0.05f)) {
                    int reg = (i * (classes + 4) + classes) * hw + p;
                    float bw = Math.max(0, o[reg]);
                    float bh = Math.max(0, o[reg + hw]);
                    float cx = p % w + o[reg + 2 * hw];
                    float cy = p / w + o[reg + 3 * hw];
                    image.add(new DetectionMetrics.Detection(firstImage + i, c, heat[p], new float[] {
                        (cx - bw / 2) * stride, (cy - bh / 2) * stride, (cx + bw / 2) * stride, (cy + bh / 2) * stride}));
                }
            }
            image.sort((x, y) -> Float.compare(y.score(), x.score()));
            dets.addAll(image.subList(0, Math.min(maxObjects, image.size())));
        }
        return dets;
    }

    /**
     * The multiply-adds of one feature map.
     *
     * @param width the hidden channels
     * @param side the map side
     * @return the count
     */
    public long macs(int width, int side) {
        return VisionOps.convMacs(inputs, width, 3, side, side) + VisionOps.convMacs(width, classes + 4, 1, side, side);
    }

    @Override
    public List<Var> parameters() {
        return List.of(hidden, hiddenBias, out, outBias);
    }

    @Override
    public String toString() {
        return "CenterNetHead(" + inputs + " -> " + classes + " classes)";
    }
}
