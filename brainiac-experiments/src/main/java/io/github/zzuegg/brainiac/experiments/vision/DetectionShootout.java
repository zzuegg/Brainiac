package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.experimental.vision.AnchorDetector;
import io.github.zzuegg.brainiac.experimental.vision.CenterNetHead;
import io.github.zzuegg.brainiac.experimental.vision.ConvBackbone;
import io.github.zzuegg.brainiac.experimental.vision.DetectionMetrics;
import io.github.zzuegg.brainiac.experimental.vision.DetectionMetrics.Detection;
import io.github.zzuegg.brainiac.experimental.vision.DiscriminativeEmbeddingLoss;
import io.github.zzuegg.brainiac.experimental.vision.EmbeddingClustering;
import io.github.zzuegg.brainiac.experimental.vision.FeaturePyramid;
import io.github.zzuegg.brainiac.experimental.vision.SetPredictionDetector;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Boxes;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 8 items 8.1.1 to 8.1.4: object detection and instance segmentation on
 * {@link ShapeScenes} (32x32, 1 to 5 glyphs of six classes at three scales).
 *
 * <ul>
 *   <li>8.1.1 RetinaNet-lite ({@link AnchorDetector}): pyramid, focal loss and GIoU, with
 *       the ablations no pyramid, cross-entropy instead of focal, smooth L1 instead of GIoU;</li>
 *   <li>8.1.2 CenterNet ({@link CenterNetHead} on the finest {@link FeaturePyramid} level),
 *       also on a crowded split where glyphs overlap;</li>
 *   <li>8.1.3 DETR-lite with and without denoising queries ({@link SetPredictionDetector}),
 *       as a learning curve;</li>
 *   <li>8.1.4 instance segmentation by a U-Net trained with the
 *       {@link DiscriminativeEmbeddingLoss} and read out by {@link EmbeddingClustering},
 *       on touching same-class pairs.</li>
 *   <li>10.6.10 ({@link #detrFollowUp}): two-stage query selection for DETR-lite against
 *       DETR, DN-DETR, H-DETR and the dense detectors, as learning curves on scenes and
 *       layouts.</li>
 * </ul>
 *
 * <p>Baselines: the {@code VisionShootout} CNN (two valid 3x3 convolutions, global average
 * pooling, a dense layer) trained on 8x8 crops and applied as a sliding-window classifier at
 * three window sizes (run fully convolutionally, which is exactly equivalent for valid
 * convolutions), its windows turned into the glyph box of the predicted class and filtered by
 * {@link Boxes#nms}; and a {@link UNet2d} semantic segmenter whose connected components are the
 * detections (or the instances). Reported: mAP@0.5, mAP@[.5:.95], the mAP@0.5 of the
 * smallest (scale 1) glyphs, multiply-adds per image and decode time.
 */
public final class DetectionShootout {

    /** The image side. */
    public static final int SIDE = 24;
    /** The glyph classes. */
    public static final int CLASSES = ShapeScenes.CLASSES;
    /** Scenes per training step. */
    public static final int BATCH = 4;

    private static final ShapeScenes SCENES = new ShapeScenes(SIDE, 0.1f);

    private DetectionShootout() {
    }

    /**
     * The outcome of one detector.
     *
     * @param method the method
     * @param map50 mAP@0.5
     * @param mapCoco mAP@[.5:.95]
     * @param mapSmall mAP@0.5 over the scale-1 glyphs
     * @param macs multiply-adds per image
     * @param decodeMillis milliseconds per image to run and decode
     */
    public record Score(String method, double map50, double mapCoco, double mapSmall, long macs,
            double decodeMillis) {
    }

    /**
     * Test scenes.
     *
     * @param n how many
     * @param layout their layout
     * @param seed the seed
     * @return the scenes
     */
    public static ShapeScenes.Scene[] testScenes(int n, ShapeScenes.Layout layout, long seed) {
        return SCENES.scenes(n, 1, 5, layout, seed);
    }

    private static float[][] boxes(ShapeScenes.Scene[] s) {
        float[][] b = new float[s.length][];
        for (int i = 0; i < s.length; i++) {
            b[i] = s[i].boxes();
        }
        return b;
    }

    private static int[][] classes(ShapeScenes.Scene[] s) {
        int[][] c = new int[s.length][];
        for (int i = 0; i < s.length; i++) {
            c[i] = s[i].classes();
        }
        return c;
    }

    /**
     * Scores a detector on scenes.
     *
     * @param method the name
     * @param detector images to detections (image indices from 0)
     * @param test the scenes
     * @param macs the detector's multiply-adds per image
     * @return the score
     */
    public static Score evaluate(String method, Function<Tensor, List<Detection>> detector, ShapeScenes.Scene[] test,
            long macs) {
        long start = System.nanoTime();
        List<Detection> dets = detector.apply(ShapeScenes.images(test));
        double ms = (System.nanoTime() - start) / 1e6 / test.length;
        float[][] b = boxes(test);
        int[][] c = classes(test);
        double map = DetectionMetrics.meanAp(dets, b, c, CLASSES, 0.5);
        double coco = DetectionMetrics.meanApCoco(dets, b, c, CLASSES);
        float[][] sb = new float[test.length][];
        int[][] sc = new int[test.length][];
        for (int i = 0; i < test.length; i++) {
            List<Integer> keep = new ArrayList<>();
            for (int k = 0; k < test[i].count(); k++) {
                if (test[i].scales()[k] == 1) {
                    keep.add(k);
                }
            }
            sb[i] = new float[4 * keep.size()];
            sc[i] = new int[keep.size()];
            for (int j = 0; j < keep.size(); j++) {
                System.arraycopy(test[i].boxes(), 4 * keep.get(j), sb[i], 4 * j, 4);
                sc[i][j] = test[i].classes()[keep.get(j)];
            }
        }
        List<Detection> small = new ArrayList<>();
        for (Detection d : dets) {
            if ((d.box()[2] - d.box()[0]) * (d.box()[3] - d.box()[1]) <= 16) {
                small.add(d);
            }
        }
        return new Score(method, map, coco, DetectionMetrics.meanAp(small, sb, sc, CLASSES, 0.5), macs, ms);
    }

    // ------------------------------------------------------------------ 8.1.1 RetinaNet-lite

    /**
     * Trains and scores the anchor detector.
     *
     * @param config the switches
     * @param steps Adam steps of {@link #BATCH} scenes
     * @param seed the seed
     * @return the score on 48 sparse test scenes
     */
    public static Score anchor(AnchorDetector.Config config, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        AnchorDetector det = new AnchorDetector(SIDE, config, r);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            ShapeScenes.Scene[] batch = SCENES.scenes(BATCH, 1, 5, ShapeScenes.Layout.SPARSE, r.nextLong());
            det.loss(ShapeScenes.images(batch), boxes(batch), classes(batch)).backward();
            opt.step(det.parameters());
        }
        String name = "RetinaNet-lite" + (config.pyramid() ? "" : " no pyramid") + (config.focal() ? "" : " CE")
                + (config.giou() ? "" : " smooth-L1");
        return evaluate(name, x -> det.detect(x, 0), testScenes(48, ShapeScenes.Layout.SPARSE, seed + 99), det.macs());
    }

    // ------------------------------------------------------------------ 8.1.2 CenterNet

    /**
     * Trains CenterNet and scores it on sparse and crowded scenes.
     *
     * @param steps Adam steps of {@link #BATCH} scenes
     * @param seed the seed
     * @return the scores on 48 sparse and 48 crowded test scenes
     */
    public static Score[] centerNet(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        ConvBackbone backbone = new ConvBackbone(1, new int[] {8, 16, 16, 24}, r);
        FeaturePyramid fpn = new FeaturePyramid(backbone.stageChannels(), 16, r);
        CenterNetHead head = new CenterNetHead(16, CLASSES, 16, r);
        List<Var> params = new ArrayList<>(backbone.parameters());
        params.addAll(fpn.parameters());
        params.addAll(head.parameters());
        Function<Var, Var> net = x -> head.forward(fpn.levels(backbone.features(x)).get(0));
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            ShapeScenes.Scene[] batch = SCENES.scenes(BATCH, 1, 5,
                    s % 2 == 0 ? ShapeScenes.Layout.SPARSE : ShapeScenes.Layout.CROWDED, r.nextLong());
            head.loss(net.apply(Var.of(ShapeScenes.images(batch))), 2, boxes(batch), classes(batch)).backward();
            opt.step(params);
        }
        long macs = backbone.macs(SIDE) + fpn.macs(backbone.stageChannels(), new int[] {12, 6, 3}) + head.macs(16, 12);
        Function<Tensor, List<Detection>> detect = x -> head.decode(Var.noGrad(() -> net.apply(Var.of(x))).value(),
                2, 20, 0);
        return new Score[] {
            evaluate("CenterNet", detect, testScenes(48, ShapeScenes.Layout.SPARSE, seed + 99), macs),
            evaluate("CenterNet (crowded)", detect, testScenes(48, ShapeScenes.Layout.CROWDED, seed + 98), macs)};
    }

    /**
     * Trains RetinaNet-lite on the same mix as {@link #centerNet} and scores it on crowded
     * scenes, the like-for-like row for CenterNet's crowded split.
     *
     * @param steps Adam steps of {@link #BATCH} scenes
     * @param seed the seed
     * @return the score on 48 crowded test scenes
     */
    public static Score anchorCrowded(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        AnchorDetector det = new AnchorDetector(SIDE, AnchorDetector.Config.retinaNet(CLASSES), r);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            ShapeScenes.Scene[] batch = SCENES.scenes(BATCH, 1, 5,
                    s % 2 == 0 ? ShapeScenes.Layout.SPARSE : ShapeScenes.Layout.CROWDED, r.nextLong());
            det.loss(ShapeScenes.images(batch), boxes(batch), classes(batch)).backward();
            opt.step(det.parameters());
        }
        return evaluate("RetinaNet-lite (crowded)", x -> det.detect(x, 0),
                testScenes(48, ShapeScenes.Layout.CROWDED, seed + 98), det.macs());
    }

    // ------------------------------------------------------------------ 8.1.3 DETR-lite

    /**
     * Trains DETR-lite and records its learning curve.
     *
     * @param denoising whether to add denoising queries
     * @param checkpoints the step counts at which to score, increasing
     * @param seed the seed
     * @return the mAP@0.5 on 48 sparse test scenes at each checkpoint
     */
    public static double[] detr(boolean denoising, int[] checkpoints, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SetPredictionDetector det = new SetPredictionDetector(SIDE, CLASSES, 8, 32, 2, denoising, 5, r);
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        ShapeScenes.Scene[] test = testScenes(48, ShapeScenes.Layout.SPARSE, seed + 99);
        double[] curve = new double[checkpoints.length];
        int step = 0;
        for (int c = 0; c < checkpoints.length; c++) {
            for (; step < checkpoints[c]; step++) {
                ShapeScenes.Scene[] batch = SCENES.scenes(BATCH, 1, 5, ShapeScenes.Layout.SPARSE, r.nextLong());
                det.loss(ShapeScenes.images(batch), boxes(batch), classes(batch)).backward();
                opt.step(det.parameters());
            }
            curve[c] = evaluate("DETR", x -> det.detect(x, 0), test, det.macs()).map50();
        }
        return curve;
    }

    // ------------------------------------------------------------------ 9.4.14 DETR follow-up

    /**
     * Wave 9 item 9.4.14: DETR-lite learning curves with the convergence options of
     * {@link SetPredictionDetector.Options} (one-to-many hybrid matching, anchor-box queries,
     * per-layer auxiliary losses), on {@link ShapeScenes} or on {@link GlyphDocuments} layout
     * pages (text-block, table and figure regions, few large boxes). The wave 8 rows
     * ({@link #detr}) and the baselines ({@link #centerNet}, {@link #anchor}) are unchanged.
     *
     * @param options the detector options ({@link SetPredictionDetector.Options#none()} is plain DETR)
     * @param layouts whether to train and test on layout pages instead of scenes
     * @param checkpoints the step counts at which to score, increasing
     * @param seed the seed
     * @return the mAP@0.5 on 48 test images at each checkpoint
     */
    public static double[] detrCurve(SetPredictionDetector.Options options, boolean layouts, int[] checkpoints,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int classes = layouts ? GlyphDocuments.REGION_KINDS : CLASSES;
        SetPredictionDetector det = new SetPredictionDetector(SIDE, classes, 8, 32, 2, false, 5, options, r);
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        Object[] test = layouts ? layoutBatch(48, new SplittableRandom(seed + 99))
                : sceneBatch(testScenes(48, ShapeScenes.Layout.SPARSE, seed + 99));
        double[] curve = new double[checkpoints.length];
        int step = 0;
        for (int c = 0; c < checkpoints.length; c++) {
            for (; step < checkpoints[c]; step++) {
                Object[] batch = layouts ? layoutBatch(BATCH, r)
                        : sceneBatch(SCENES.scenes(BATCH, 1, 5, ShapeScenes.Layout.SPARSE, r.nextLong()));
                det.loss((Tensor) batch[0], (float[][]) batch[1], (int[][]) batch[2]).backward();
                opt.step(det.parameters());
            }
            List<Detection> dets = det.detect((Tensor) test[0], 0);
            curve[c] = DetectionMetrics.meanAp(dets, (float[][]) test[1], (int[][]) test[2], classes, 0.5);
        }
        return curve;
    }

    /** Images, boxes and classes of scenes. */
    private static Object[] sceneBatch(ShapeScenes.Scene[] s) {
        return new Object[] {ShapeScenes.images(s), boxes(s), classes(s)};
    }

    /** Images, boxes and classes of layout pages with 1 to 3 regions. */
    private static Object[] layoutBatch(int n, SplittableRandom r) {
        List<GlyphDocuments.Page> pages = new ArrayList<>();
        float[][] b = new float[n][];
        int[][] c = new int[n][];
        for (int i = 0; i < n; i++) {
            GlyphDocuments.Page p = GlyphDocuments.layout(SIDE, 3, 0.02, r);
            pages.add(p);
            b[i] = new float[4 * p.regions().size()];
            c[i] = new int[p.regions().size()];
            for (int k = 0; k < c[i].length; k++) {
                System.arraycopy(p.regions().get(k).box(), 0, b[i], 4 * k, 4);
                c[i][k] = p.regions().get(k).kind();
            }
        }
        return new Object[] {GlyphDocuments.images(pages, SIDE, SIDE), b, c};
    }

    // ------------------------------------------------------------------ sliding-window CNN

    /** The VisionShootout CNN, as raw parameters so it can also run fully convolutionally. */
    private static final class WindowCnn {
        final Var w1;
        final Var b1;
        final Var w2;
        final Var b2;
        final Var dense;
        final Var bias;

        WindowCnn(SplittableRandom r) {
            w1 = VisionOps.convWeight(r, 16, 1, 3);
            b1 = VisionOps.zeros(16);
            w2 = VisionOps.convWeight(r, 32, 16, 3);
            b2 = VisionOps.zeros(32);
            dense = Var.param(Tensor.normal(Shape.of(CLASSES + 1, 32, 1, 1), 0, (float) Math.sqrt(1 / 32.0), r));
            bias = VisionOps.zeros(CLASSES + 1);
        }

        List<Var> parameters() {
            return List.of(w1, b1, w2, b2, dense, bias);
        }

        /** Valid convolutions, a 4x4 average pool and the dense layer as a 1x1 convolution. */
        Var map(Var x) {
            Var h = Functional.conv(x, w1, b1, new int[] {1, 1}, new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, 1)
                    .relu();
            h = Functional.conv(h, w2, b2, new int[] {1, 1}, new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, 1)
                    .relu();
            h = Functional.pool(h, Functional.PoolKind.AVG, new int[] {4, 4}, new int[] {1, 1}, new int[] {0, 0},
                    new int[] {0, 0}, new int[] {1, 1}, false, false, 1f);
            return VisionOps.conv(h, dense, bias);
        }
    }

    private static float bilinear(float[] img, int side, float x, float y) {
        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);
        float fx = x - x0;
        float fy = y - y0;
        float v = 0;
        for (int dy = 0; dy <= 1; dy++) {
            for (int dx = 0; dx <= 1; dx++) {
                int xx = x0 + dx;
                int yy = y0 + dy;
                if (xx >= 0 && yy >= 0 && xx < side && yy < side) {
                    v += img[yy * side + xx] * (dx == 1 ? fx : 1 - fx) * (dy == 1 ? fy : 1 - fy);
                }
            }
        }
        return v;
    }

    private static void crop(float[] img, float x0, float y0, float size, float[] out, int offset) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                out[offset + y * 8 + x] = bilinear(img, SIDE, x0 + (x + 0.5f) * size / 8 - 0.5f,
                        y0 + (y + 0.5f) * size / 8 - 0.5f);
            }
        }
    }

    /**
     * Trains the sliding-window CNN on 8x8 crops and scores it.
     *
     * @param steps Adam steps of crops from 4 scenes
     * @param seed the seed
     * @return the score on 48 sparse test scenes
     */
    public static Score slidingWindow(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        WindowCnn cnn = new WindowCnn(r);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            ShapeScenes.Scene[] batch = SCENES.scenes(4, 1, 5, ShapeScenes.Layout.SPARSE, r.nextLong());
            List<float[]> crops = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            for (ShapeScenes.Scene sc : batch) {
                for (int k = 0; k < sc.count(); k++) {
                    float[] sq = sc.square(k);
                    float[] c = new float[64];
                    crop(sc.image(), sq[0] - 1, sq[1] - 1, sq[2] - sq[0] + 2, c, 0);
                    crops.add(c);
                    labels.add(sc.classes()[k] + 1);
                }
                for (int neg = 0; neg < 2 * sc.count(); neg++) {
                    int scale = 1 + r.nextInt(3);
                    float size = 3 * scale + 2;
                    float x = r.nextFloat() * (SIDE - size);
                    float y = r.nextFloat() * (SIDE - size);
                    float[] win = {x + 1, y + 1, x + size - 1, y + size - 1};
                    boolean clear = true;
                    for (int k = 0; k < sc.count(); k++) {
                        clear &= DetectionMetrics.iou(win, sc.square(k)) < 0.3;
                    }
                    if (clear) {
                        float[] c = new float[64];
                        crop(sc.image(), x, y, size, c, 0);
                        crops.add(c);
                        labels.add(0);
                    }
                }
            }
            int n = crops.size();
            float[] x = new float[n * 64];
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(crops.get(i), 0, x, i * 64, 64);
                y[i] = labels.get(i);
            }
            Var logits = cnn.map(Var.of(Tensor.of(Shape.of(n, 1, 8, 8), x))).reshape(n, CLASSES + 1);
            Functional.crossEntropy(logits, Tensor.of(Shape.of(n), y)).backward();
            opt.step(cnn.parameters());
        }
        long macs = 0;
        for (int scale = 1; scale <= 3; scale++) {
            int rs = Math.round(SIDE * 8f / (3 * scale + 2));
            macs += VisionOps.convMacs(1, 16, 3, rs - 2, rs - 2) + VisionOps.convMacs(16, 32, 3, rs - 4, rs - 4)
                    + VisionOps.convMacs(32, CLASSES + 1, 1, rs - 7, rs - 7);
        }
        return evaluate("sliding-window CNN", x -> slidingDetect(cnn, x), testScenes(48, ShapeScenes.Layout.SPARSE,
                seed + 99), macs);
    }

    private static List<Detection> slidingDetect(WindowCnn cnn, Tensor images) {
        int n = images.shape().dim(0);
        List<Detection> dets = new ArrayList<>();
        List<float[]> boxes = new ArrayList<>();
        for (int scale = 1; scale <= 3; scale++) {
            float size = 3 * scale + 2;
            int rs = Math.round(SIDE * 8f / size);
            float factor = SIDE / (float) rs;
            int finalScale = scale;
            Tensor probs = Var.noGrad(() -> Activation.SOFTMAX.forward(cnn.map(Functional.resize(Var.of(images),
                    new int[] {rs, rs}, Functional.Interpolation.LINEAR, false)).permute(0, 2, 3, 1))).value();
            int m = rs - 7;
            float[] p = probs.toArray();
            for (int i = 0; i < n; i++) {
                for (int y = 0; y < m; y++) {
                    for (int x = 0; x < m; x++) {
                        int o = ((i * m + y) * m + x) * (CLASSES + 1);
                        int best = 1;
                        for (int c = 2; c <= CLASSES; c++) {
                            best = p[o + c] > p[o + best] ? c : best;
                        }
                        if (p[o + best] < 0.2f) {
                            continue;
                        }
                        float wx = x * factor + 1;
                        float wy = y * factor + 1;
                        boxes.add(ShapeScenes.Scene.glyphBox(best - 1, finalScale, wx, wy));
                        dets.add(new Detection(i, best - 1, p[o + best], null));
                    }
                }
            }
        }
        List<Detection> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> idx = new ArrayList<>();
            for (int k = 0; k < dets.size(); k++) {
                if (dets.get(k).image() == i) {
                    idx.add(k);
                }
            }
            float[] bx = new float[4 * idx.size()];
            float[] sc = new float[idx.size()];
            int[] cl = new int[idx.size()];
            for (int j = 0; j < idx.size(); j++) {
                System.arraycopy(boxes.get(idx.get(j)), 0, bx, 4 * j, 4);
                sc[j] = dets.get(idx.get(j)).score();
                cl[j] = dets.get(idx.get(j)).cls();
            }
            for (int k : Boxes.nms(bx, sc, cl, 0.3f, 0.2f, 20)) {
                out.add(new Detection(i, cl[k], sc[k], new float[] {bx[4 * k], bx[4 * k + 1], bx[4 * k + 2],
                    bx[4 * k + 3]}));
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ U-Net + connected components

    private static Tensor semanticTargets(ShapeScenes.Scene[] s) {
        int p = s[0].side() * s[0].side();
        float[] t = new float[s.length * p];
        for (int i = 0; i < s.length; i++) {
            int[] sem = s[i].semantic();
            for (int j = 0; j < p; j++) {
                t[i * p + j] = sem[j];
            }
        }
        return Tensor.of(Shape.of(s.length * p), t);
    }

    private static UNet2d trainSegmenter(ShapeScenes gen, int steps, ShapeScenes.Layout layout, SplittableRandom r) {
        int side = gen.side();
        UNet2d net = new UNet2d(1, CLASSES + 1, 4, 2, true, 0, r);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            ShapeScenes.Scene[] batch = gen.scenes(BATCH, 1, 5, layout, r.nextLong());
            Var logits = net.forward(Var.of(ShapeScenes.images(batch))).permute(0, 2, 3, 1)
                    .reshape(BATCH * side * side, CLASSES + 1);
            Functional.crossEntropy(logits, semanticTargets(batch)).backward();
            opt.step(net.parameters());
        }
        return net;
    }

    /** Per image: the U-Net's per-pixel class (0 background) and per-pixel class probabilities. */
    private static float[][] segment(UNet2d net, Tensor images, int[][] labelsOut) {
        int n = images.shape().dim(0);
        int side = images.shape().dim(2);
        int p = side * side;
        float[] probs = Var.noGrad(() -> Activation.SOFTMAX.forward(net.forward(Var.of(images)).permute(0, 2, 3, 1)))
                .value().toArray();
        float[][] best = new float[n][p];
        for (int i = 0; i < n; i++) {
            labelsOut[i] = new int[p];
            for (int j = 0; j < p; j++) {
                int o = (i * p + j) * (CLASSES + 1);
                int arg = 0;
                for (int c = 1; c <= CLASSES; c++) {
                    arg = probs[o + c] > probs[o + arg] ? c : arg;
                }
                labelsOut[i][j] = arg;
                best[i][j] = probs[o + arg];
            }
        }
        return best;
    }

    /**
     * Trains the U-Net segmenter and detects connected components of its labels.
     *
     * @param steps Adam steps of {@link #BATCH} scenes
     * @param seed the seed
     * @return the score on 48 sparse test scenes
     */
    public static Score unetComponents(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        UNet2d net = trainSegmenter(SCENES, steps, ShapeScenes.Layout.SPARSE, r);
        long macs = unetMacs(SIDE);
        return evaluate("U-Net + connected components", x -> {
            int n = x.shape().dim(0);
            int[][] labels = new int[n][];
            float[][] conf = segment(net, x, labels);
            List<Detection> dets = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int[] comp = EmbeddingClustering.connectedComponents(labels[i], SIDE, true, 1);
                int k = 0;
                for (int v : comp) {
                    k = Math.max(k, v);
                }
                for (int c = 1; c <= k; c++) {
                    float[] box = {SIDE, SIDE, 0, 0};
                    double score = 0;
                    int count = 0;
                    int cls = 0;
                    for (int j = 0; j < comp.length; j++) {
                        if (comp[j] == c) {
                            box[0] = Math.min(box[0], j % SIDE);
                            box[1] = Math.min(box[1], j / SIDE);
                            box[2] = Math.max(box[2], j % SIDE + 1);
                            box[3] = Math.max(box[3], j / SIDE + 1);
                            score += conf[i][j];
                            count++;
                            cls = labels[i][j] - 1;
                        }
                    }
                    dets.add(new Detection(i, cls, (float) (score / count), box));
                }
            }
            return dets;
        }, testScenes(48, ShapeScenes.Layout.SPARSE, seed + 99), macs);
    }

    private static long unetMacs(int side) {
        // UNet2d(1, 7, 4, 2): two 3x3 convolutions per stage
        long m = 0;
        int[] c = {1, 4, 8, 16};
        for (int l = 0; l < 3; l++) {
            int s = side >> l;
            m += VisionOps.convMacs(c[l], c[l + 1], 3, s, s) + VisionOps.convMacs(c[l + 1], c[l + 1], 3, s, s);
        }
        for (int l = 1; l >= 0; l--) {
            int s = side >> l;
            m += VisionOps.convMacs(c[l + 2], c[l + 1], 3, s, s) + VisionOps.convMacs(2 * c[l + 1], c[l + 1], 3, s, s)
                    + VisionOps.convMacs(c[l + 1], c[l + 1], 3, s, s);
        }
        return m + VisionOps.convMacs(4, CLASSES + 1, 1, side, side);
    }

    // ------------------------------------------------------------------ 8.1.4 instance segmentation

    /**
     * The outcome of an instance segmenter.
     *
     * @param method the method
     * @param maskAp mask AP@0.5 (class-agnostic)
     * @param fgAri the foreground adjusted Rand index
     */
    public record InstanceScore(String method, double maskAp, double fgAri) {
    }

    private static InstanceScore scoreInstances(String method, ShapeScenes.Scene[] test, int[][] predicted,
            float[][] confidence) {
        List<DetectionMetrics.MaskDetection> masks = new ArrayList<>();
        int[][] truth = new int[test.length][];
        double ari = 0;
        for (int i = 0; i < test.length; i++) {
            truth[i] = test[i].instances();
            ari += DetectionMetrics.foregroundAri(predicted[i], truth[i]);
            int k = 0;
            for (int v : predicted[i]) {
                k = Math.max(k, v);
            }
            for (int c = 1; c <= k; c++) {
                boolean[] m = new boolean[predicted[i].length];
                double s = 0;
                int count = 0;
                for (int j = 0; j < m.length; j++) {
                    if (predicted[i][j] == c) {
                        m[j] = true;
                        s += confidence[i][j];
                        count++;
                    }
                }
                if (count > 0) {
                    masks.add(new DetectionMetrics.MaskDetection(i, (float) (s / count * Math.min(1, count / 4.0)), m));
                }
            }
        }
        return new InstanceScore(method, DetectionMetrics.maskAp(masks, truth, 0.5), ari / test.length);
    }

    /**
     * The U-Net plus connected-components instance baseline on touching same-class pairs (glyph scales 1 and 2).
     *
     * @param side the scene side, divisible by 4
     * @param steps Adam steps of {@link #BATCH} scenes
     * @param seed the seed
     * @return the score on 32 touching test scenes
     */
    public static InstanceScore instanceComponents(int side, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        ShapeScenes gen = new ShapeScenes(side, 0.1f, 2);
        UNet2d net = trainSegmenter(gen, steps, ShapeScenes.Layout.TOUCHING, r);
        ShapeScenes.Scene[] test = gen.scenes(32, 2, 4, ShapeScenes.Layout.TOUCHING, seed + 99);
        int[][] labels = new int[test.length][];
        float[][] conf = segment(net, ShapeScenes.images(test), labels);
        int[][] inst = new int[test.length][];
        for (int i = 0; i < test.length; i++) {
            inst[i] = EmbeddingClustering.connectedComponents(labels[i], side, true, 1);
        }
        return scoreInstances("U-Net + connected components", test, inst, conf);
    }

    /**
     * A U-Net with a foreground head and an embedding head trained with the discriminative
     * loss, read out by greedy seed clustering, on touching same-class pairs (glyph scales 1 and 2).
     *
     * @param side the scene side, divisible by 4
     * @param steps Adam steps of {@link #BATCH} scenes
     * @param seed the seed
     * @return the score on 32 touching test scenes
     */
    public static InstanceScore instanceEmbedding(int side, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int dims = 4;
        ShapeScenes gen = new ShapeScenes(side, 0.1f, 2);
        UNet2d net = new UNet2d(1, 2 + dims, 4, 2, true, 0, r);
        DiscriminativeEmbeddingLoss disc = new DiscriminativeEmbeddingLoss(0.5f, 1.5f, 0.001f);
        Optimizer opt = Optimizer.adam(3e-3f);
        int p = side * side;
        for (int s = 0; s < steps; s++) {
            ShapeScenes.Scene[] batch = gen.scenes(BATCH, 2, 4, ShapeScenes.Layout.TOUCHING, r.nextLong());
            Var out = net.forward(Var.of(ShapeScenes.images(batch)));
            float[] fg = new float[BATCH * p];
            float[] ids = new float[BATCH * p];
            for (int i = 0; i < BATCH; i++) {
                for (int j = 0; j < p; j++) {
                    ids[i * p + j] = batch[i].instances()[j];
                    fg[i * p + j] = batch[i].instances()[j] > 0 ? 1 : 0;
                }
            }
            Var fgLoss = Functional.crossEntropy(out.slice(1, 0, 2).permute(0, 2, 3, 1).reshape(BATCH * p, 2),
                    Tensor.of(Shape.of(BATCH * p), fg));
            fgLoss.mul(5f).add(disc.compute(out.slice(1, 2, 2 + dims), Tensor.of(Shape.of(BATCH, side, side), ids))).backward();
            opt.step(net.parameters());
        }
        ShapeScenes.Scene[] test = gen.scenes(32, 2, 4, ShapeScenes.Layout.TOUCHING, seed + 99);
        float[] out = Var.noGrad(() -> net.forward(Var.of(ShapeScenes.images(test)))).value().toArray();
        int[][] inst = new int[test.length][];
        float[][] conf = new float[test.length][p];
        for (int i = 0; i < test.length; i++) {
            int base = i * (2 + dims) * p;
            boolean[] fg = new boolean[p];
            float[] emb = new float[dims * p];
            for (int j = 0; j < p; j++) {
                double a = out[base + j];
                double b = out[base + p + j];
                conf[i][j] = (float) (1 / (1 + Math.exp(a - b)));
                fg[j] = b > a;
            }
            System.arraycopy(out, base + 2 * p, emb, 0, dims * p);
            inst[i] = EmbeddingClustering.seeds(emb, dims, fg, disc.deltaVar(), disc.deltaDist(), 2);
        }
        return scoreInstances("discriminative embedding", test, inst, conf);
    }

    // ------------------------------------------------------------------ 10.6.10 query selection

    /**
     * A learning curve.
     *
     * @param method the detector
     * @param dataset "scenes" or "layouts"
     * @param steps the checkpoints
     * @param map50 mAP@0.5 at each checkpoint
     */
    public record DetrCurve(String method, String dataset, int[] steps, double[] map50) {

        /**
         * Copies the arrays.
         *
         * @param method the detector
         * @param dataset the dataset
         * @param steps the checkpoints
         * @param map50 the scores
         */
        public DetrCurve {
            steps = steps.clone();
            map50 = map50.clone();
        }

        @Override
        public int[] steps() {
            return steps.clone();
        }

        @Override
        public double[] map50() {
            return map50.clone();
        }

        @Override
        public String toString() {
            StringBuilder b = new StringBuilder(method + " on " + dataset + ":");
            for (int i = 0; i < steps.length; i++) {
                b.append(String.format(" %d:%.3f", steps[i], map50[i]));
            }
            return b.toString();
        }
    }

    /**
     * Wave 10 item 10.6.10: DETR-lite with two-stage query selection
     * ({@link SetPredictionDetector.Options#querySelection()}) against DETR, DN-DETR (scenes
     * only, as in wave 8) and H-DETR with anchors and auxiliary losses, on {@link ShapeScenes}
     * and {@link GlyphDocuments} layouts; with {@code baselines} also CenterNet and
     * RetinaNet-lite on scenes, retrained to each checkpoint. Wave 8 and 9 rows are unchanged.
     *
     * @param checkpoints the step counts at which to score, increasing
     * @param scenes whether to run the scene curves
     * @param layouts whether to run the layout curves
     * @param baselines whether to add the CenterNet and RetinaNet-lite scene curves
     * @param seed the seed
     * @return the curves
     */
    public static List<DetrCurve> detrFollowUp(int[] checkpoints, boolean scenes, boolean layouts,
            boolean baselines, long seed) {
        SetPredictionDetector.Options hybrid = new SetPredictionDetector.Options(16, 3, true, true);
        List<DetrCurve> out = new ArrayList<>();
        for (boolean lay : new boolean[] {false, true}) {
            if (lay ? !layouts : !scenes) {
                continue;
            }
            String data = lay ? "layouts" : "scenes";
            out.add(new DetrCurve("DETR", data, checkpoints,
                    detrCurve(SetPredictionDetector.Options.none(), lay, checkpoints, seed)));
            if (!lay) {
                out.add(new DetrCurve("DN-DETR", data, checkpoints, detr(true, checkpoints, seed)));
            }
            out.add(new DetrCurve("H-DETR + anchors + aux", data, checkpoints,
                    detrCurve(hybrid, lay, checkpoints, seed)));
            out.add(new DetrCurve("DETR + query selection", data, checkpoints,
                    detrCurve(SetPredictionDetector.Options.none().withQuerySelection(true), lay, checkpoints, seed)));
            out.add(new DetrCurve("H-DETR + query selection + aux", data, checkpoints,
                    detrCurve(hybrid.withQuerySelection(true), lay, checkpoints, seed)));
        }
        if (baselines && scenes) {
            double[] cn = new double[checkpoints.length];
            double[] rn = new double[checkpoints.length];
            for (int c = 0; c < checkpoints.length; c++) {
                cn[c] = centerNet(checkpoints[c], seed)[0].map50();
                rn[c] = anchor(AnchorDetector.Config.retinaNet(CLASSES), checkpoints[c], seed).map50();
            }
            out.add(new DetrCurve("CenterNet", "scenes", checkpoints, cn));
            out.add(new DetrCurve("RetinaNet-lite", "scenes", checkpoints, rn));
        }
        return out;
    }

    /**
     * The long 10.6.10 Run: curves to 5,000 steps on both datasets with the dense baselines.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (DetrCurve c : detrFollowUp(new int[] {250, 500, 1000, 2000, 5000}, true, true, true, 1)) {
            System.out.println(c);
        }
    }
}
