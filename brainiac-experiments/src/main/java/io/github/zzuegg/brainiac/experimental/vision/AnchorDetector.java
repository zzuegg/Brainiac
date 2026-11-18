package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Boxes;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A single-stage anchor detector in the style of RetinaNet (Lin et al., 2017): a
 * {@link ConvBackbone} feeds a {@link FeaturePyramid}, and one shared head predicts, for
 * every anchor of core {@link Boxes#anchors} on every pyramid level, a sigmoid score per
 * class and four box deltas ({@link Boxes#encode}). Anchors are matched to ground truth by
 * IoU (positive at 0.5 or more, background below 0.4, ignored in between, and every object's
 * best anchor positive); the class head is trained with the focal loss (core
 * {@link Loss#focal}), normalised by the number of positive anchors, so the thousands of easy
 * background anchors do not swamp the gradient; the box head with GIoU
 * ({@link Boxes#giouLoss}) or smooth L1 ({@link Loss#huber}) on positives. Detections are
 * decoded and filtered by per-class {@link Boxes#nms}.
 *
 * <p>The {@link Config} switches give the paper's ablations: no pyramid (one head on the
 * coarsest backbone map with every anchor size there), cross-entropy instead of focal loss
 * (focal with {@code gamma = 0}), and smooth L1 instead of GIoU. As a {@link Layer} it maps
 * images to the raw head output {@code [batch, anchors, classes + 4]}.
 *
 * <p>Paper: Lin et al., "Focal Loss for Dense Object Detection" (2017),
 * <a href="https://arxiv.org/abs/1708.02002">arXiv:1708.02002</a>.
 */
public final class AnchorDetector implements Layer {

    /**
     * The detector's switches.
     *
     * @param classes the object classes
     * @param pyramid whether to use the feature pyramid (otherwise the coarsest map only)
     * @param focal whether the class loss is the focal loss (otherwise cross-entropy)
     * @param giou whether the box loss is GIoU (otherwise smooth L1 on the deltas)
     * @param width the pyramid and head channels
     */
    public record Config(int classes, boolean pyramid, boolean focal, boolean giou, int width) {

        /**
         * RetinaNet: pyramid, focal loss and GIoU.
         *
         * @param classes the object classes
         * @return the configuration
         */
        public static Config retinaNet(int classes) {
            return new Config(classes, true, true, true, 16);
        }
    }

    private static final float[] RATIOS = {1 / 3f, 1f, 3f};

    private final Config config;
    private final int side;
    private final ConvBackbone backbone;
    private final FeaturePyramid pyramid;
    private final Var lateral;
    private final Var lateralBias;
    private final Var headConv;
    private final Var headBias;
    private final Var clsConv;
    private final Var clsBias;
    private final Var boxConv;
    private final Var boxBias;
    private final int perCell;
    private final float[] anchors;
    private final int[] levelSides;

    /**
     * Creates a detector for square images of one side.
     *
     * @param side the image side, divisible by 8
     * @param config the switches
     * @param random the source of initial weights
     */
    public AnchorDetector(int side, Config config, RandomGenerator random) {
        this.config = config;
        this.side = side;
        int w = config.width();
        this.backbone = new ConvBackbone(1, new int[] {8, 16, 16, 24}, random);
        int[] stages = backbone.stageChannels();
        List<Tensor> levelAnchors = new ArrayList<>();
        if (config.pyramid()) {
            this.pyramid = new FeaturePyramid(stages, w, random);
            this.lateral = null;
            this.lateralBias = null;
            this.levelSides = new int[] {side / 2, side / 4, side / 8};
            this.perCell = 2 * RATIOS.length;
            for (int l = 0; l < 3; l++) {
                float k = l + 1;
                levelAnchors.add(Boxes.anchors(levelSides[l], levelSides[l], 2 << l,
                        new float[] {3 * k, 1.73f * k}, RATIOS));
            }
        } else {
            this.pyramid = null;
            this.lateral = VisionOps.convWeight(random, w, stages[stages.length - 1], 1);
            this.lateralBias = VisionOps.zeros(w);
            this.levelSides = new int[] {side / 8};
            this.perCell = 6 * RATIOS.length;
            levelAnchors.add(Boxes.anchors(side / 8, side / 8, 8, new float[] {3, 1.73f, 6, 3.46f, 9, 5.2f}, RATIOS));
        }
        int total = 0;
        for (Tensor t : levelAnchors) {
            total += t.shape().dim(0);
        }
        this.anchors = new float[4 * total];
        int o = 0;
        for (Tensor t : levelAnchors) {
            float[] a = t.toArray();
            System.arraycopy(a, 0, anchors, o, a.length);
            o += a.length;
        }
        this.headConv = VisionOps.convWeight(random, w, w, 3);
        this.headBias = VisionOps.zeros(w);
        this.clsConv = Var.param(Tensor.normal(Shape.of(perCell * config.classes(), w, 1, 1), 0, 0.01f, random));
        float prior = (float) -Math.log((1 - 0.01) / 0.01);
        this.clsBias = Var.param(Tensor.full(Shape.of(perCell * config.classes()), prior));
        this.boxConv = Var.param(Tensor.normal(Shape.of(perCell * 4, w, 1, 1), 0, 0.01f, random));
        this.boxBias = VisionOps.zeros(perCell * 4);
    }

    /** @return the number of anchors per image */
    public int anchorCount() {
        return anchors.length / 4;
    }

    /** @return the switches */
    public Config config() {
        return config;
    }

    /**
     * Runs the network.
     *
     * @param images {@code [batch, 1, side, side]}
     * @return {@code [batch, anchors, classes + 4]}: class logits, then box deltas
     */
    @Override
    public Var forward(Var images) {
        int b = images.shape().dim(0);
        int k = config.classes();
        List<Var> feats = backbone.features(images);
        List<Var> levels;
        if (pyramid != null) {
            levels = pyramid.levels(feats);
        } else {
            levels = List.of(VisionOps.conv(feats.get(feats.size() - 1), lateral, lateralBias));
        }
        List<Var> outs = new ArrayList<>();
        for (Var p : levels) {
            int h = p.shape().dim(2);
            int ww = p.shape().dim(3);
            Var t = VisionOps.conv(p, headConv, headBias).relu();
            Var cls = VisionOps.conv(t, clsConv, clsBias).permute(0, 2, 3, 1).reshape(b, h * ww * perCell, k);
            Var box = VisionOps.conv(t, boxConv, boxBias).permute(0, 2, 3, 1).reshape(b, h * ww * perCell, 4);
            outs.add(Var.concat(2, cls, box));
        }
        return outs.size() == 1 ? outs.get(0) : Var.concat(1, outs.toArray(new Var[0]));
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(anchorCount(), config.classes() + 4);
    }

    /**
     * The training loss of a batch.
     *
     * @param images {@code [batch, 1, side, side]}
     * @param boxes per image the ground-truth boxes, {@code 4 k} values
     * @param classes per image the ground-truth classes
     * @return the class loss plus the box loss
     */
    public Var loss(Tensor images, float[][] boxes, int[][] classes) {
        int b = images.shape().dim(0);
        int n = anchorCount();
        int k = config.classes();
        Var out = forward(Var.of(images)).reshape(b * n, k + 4);
        List<Integer> keep = new ArrayList<>();
        List<Integer> pos = new ArrayList<>();
        List<float[]> keepTargets = new ArrayList<>();
        List<float[]> posBoxes = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            int[] match = match(boxes[i]);
            for (int a = 0; a < n; a++) {
                if (match[a] == -2) {
                    continue;
                }
                keep.add(i * n + a);
                float[] t = new float[k];
                if (match[a] >= 0) {
                    t[classes[i][match[a]]] = 1;
                    pos.add(i * n + a);
                    posBoxes.add(new float[] {boxes[i][4 * match[a]], boxes[i][4 * match[a] + 1],
                        boxes[i][4 * match[a] + 2], boxes[i][4 * match[a] + 3], anchors[4 * a], anchors[4 * a + 1],
                        anchors[4 * a + 2], anchors[4 * a + 3]});
                }
                keepTargets.add(t);
            }
        }
        Var logits = out.slice(1, 0, k).gather(0, indices(keep));
        float[] tgt = new float[keep.size() * k];
        for (int i = 0; i < keep.size(); i++) {
            System.arraycopy(keepTargets.get(i), 0, tgt, i * k, k);
        }
        Loss cls = config.focal() ? Loss.focal(2f, 0.25f) : Loss.focal(0f, 0.5f);
        float normaliser = (float) keep.size() * k / Math.max(1, pos.size());
        Var loss = cls.compute(logits.sigmoid(), Tensor.of(Shape.of(keep.size(), k), tgt)).mul(normaliser);
        if (!pos.isEmpty()) {
            int p = pos.size();
            float[] gt = new float[4 * p];
            float[] an = new float[4 * p];
            for (int i = 0; i < p; i++) {
                System.arraycopy(posBoxes.get(i), 0, gt, 4 * i, 4);
                System.arraycopy(posBoxes.get(i), 4, an, 4 * i, 4);
            }
            Var deltas = out.slice(1, k, k + 4).gather(0, indices(pos));
            Var anchorBoxes = Var.of(Tensor.of(Shape.of(p, 4), an));
            Var target = Var.of(Tensor.of(Shape.of(p, 4), gt));
            Var boxLoss = config.giou() ? Boxes.giouLoss(Boxes.decode(deltas, anchorBoxes), target)
                    : Loss.huber(0.11f).compute(deltas, Boxes.encode(target, anchorBoxes).value()).mul(4);
            loss = loss.add(boxLoss.mul(2));
        }
        return loss;
    }

    /** Per anchor: the matched object, -1 for background or -2 for ignored. */
    private int[] match(float[] boxes) {
        int n = anchorCount();
        int g = boxes.length / 4;
        int[] m = new int[n];
        double[] best = new double[n];
        int[] bestAnchor = new int[g];
        double[] bestIou = new double[g];
        java.util.Arrays.fill(bestIou, -1);
        for (int a = 0; a < n; a++) {
            float[] an = {anchors[4 * a], anchors[4 * a + 1], anchors[4 * a + 2], anchors[4 * a + 3]};
            int arg = -1;
            double max = 0;
            for (int j = 0; j < g; j++) {
                double o = DetectionMetrics.iou(an, new float[] {boxes[4 * j], boxes[4 * j + 1], boxes[4 * j + 2],
                    boxes[4 * j + 3]});
                if (o > max) {
                    max = o;
                    arg = j;
                }
                if (o > bestIou[j]) {
                    bestIou[j] = o;
                    bestAnchor[j] = a;
                }
            }
            best[a] = max;
            m[a] = max >= 0.5 ? arg : max < 0.4 ? -1 : -2;
        }
        for (int j = 0; j < g; j++) {
            m[bestAnchor[j]] = j;
        }
        return m;
    }

    private static Tensor indices(List<Integer> idx) {
        float[] f = new float[idx.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = idx.get(i);
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /**
     * Detects objects.
     *
     * @param images {@code [batch, 1, side, side]}
     * @param firstImage the image index of the first image, for the detections
     * @return the detections after score filtering and per-class NMS
     */
    public List<DetectionMetrics.Detection> detect(Tensor images, int firstImage) {
        int b = images.shape().dim(0);
        int n = anchorCount();
        int k = config.classes();
        float[] out = Var.noGrad(() -> forward(Var.of(images))).value().toArray();
        List<DetectionMetrics.Detection> dets = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            List<float[]> cand = new ArrayList<>();
            for (int a = 0; a < n; a++) {
                int o = (i * n + a) * (k + 4);
                for (int c = 0; c < k; c++) {
                    float s = (float) (1 / (1 + Math.exp(-out[o + c])));
                    if (s > 0.05f) {
                        cand.add(new float[] {s, c, a});
                    }
                }
            }
            cand.sort((x, y) -> Float.compare(y[0], x[0]));
            int m = Math.min(cand.size(), 300);
            float[] bx = new float[4 * m];
            float[] sc = new float[m];
            int[] cl = new int[m];
            for (int j = 0; j < m; j++) {
                float[] c = cand.get(j);
                int a = (int) c[2];
                int o = (i * n + a) * (k + 4) + k;
                float[] d = decode(out[o], out[o + 1], out[o + 2], out[o + 3], a);
                System.arraycopy(d, 0, bx, 4 * j, 4);
                sc[j] = c[0];
                cl[j] = (int) c[1];
            }
            for (int idx : Boxes.nms(bx, sc, cl, 0.5f, 0.05f, 20)) {
                dets.add(new DetectionMetrics.Detection(firstImage + i, cl[idx], sc[idx],
                        new float[] {bx[4 * idx], bx[4 * idx + 1], bx[4 * idx + 2], bx[4 * idx + 3]}));
            }
        }
        return dets;
    }

    private float[] decode(float dx, float dy, float dw, float dh, int a) {
        float aw = anchors[4 * a + 2] - anchors[4 * a];
        float ah = anchors[4 * a + 3] - anchors[4 * a + 1];
        float cx = anchors[4 * a] + aw / 2 + dx * aw;
        float cy = anchors[4 * a + 1] + ah / 2 + dy * ah;
        float w = (float) Math.exp(Math.min(4, dw)) * aw;
        float h = (float) Math.exp(Math.min(4, dh)) * ah;
        return new float[] {Math.max(0, cx - w / 2), Math.max(0, cy - h / 2), Math.min(side, cx + w / 2),
            Math.min(side, cy + h / 2)};
    }

    /** @return the multiply-adds of one image */
    public long macs() {
        long m = backbone.macs(side);
        long head = 0;
        int w = config.width();
        for (int s : levelSides) {
            head += VisionOps.convMacs(w, w, 3, s, s) + VisionOps.convMacs(w, perCell * (config.classes() + 4), 1, s, s);
        }
        if (pyramid != null) {
            m += pyramid.macs(backbone.stageChannels(), levelSides);
        } else {
            m += VisionOps.convMacs(backbone.stageChannels()[2], w, 1, levelSides[0], levelSides[0]);
        }
        return m + head;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(backbone.parameters());
        if (pyramid != null) {
            p.addAll(pyramid.parameters());
        } else {
            p.add(lateral);
            p.add(lateralBias);
        }
        p.addAll(List.of(headConv, headBias, clsConv, clsBias, boxConv, boxBias));
        return p;
    }

    @Override
    public String toString() {
        return "AnchorDetector(" + config + ", " + anchorCount() + " anchors)";
    }
}
