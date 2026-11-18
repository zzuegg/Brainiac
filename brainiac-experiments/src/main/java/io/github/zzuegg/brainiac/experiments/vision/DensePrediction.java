package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.experimental.vision.AtrousPyramidPooling;
import io.github.zzuegg.brainiac.experimental.vision.ClassicalFlow;
import io.github.zzuegg.brainiac.experimental.vision.ConvBackbone;
import io.github.zzuegg.brainiac.experimental.vision.FlowMetrics;
import io.github.zzuegg.brainiac.experimental.vision.KeypointHeads;
import io.github.zzuegg.brainiac.experimental.vision.RecurrentFlowRefiner;
import io.github.zzuegg.brainiac.experimental.vision.StereoCostVolume;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 8 items 8.1.5 to 8.1.9: dense prediction on synthetic scenes.
 *
 * <ul>
 *   <li>8.1.5 DeepLab-lite: {@link AtrousPyramidPooling} on a stride-2 map with a one-skip
 *       decoder, against {@code ShapeSegmentation}'s {@link UNet2d} and its no-skip network,
 *       with a loss sweep over cross-entropy, Dice ({@link Loss#dice}) and Lovász-softmax
 *       ({@link Functional#lovaszSoftmax});</li>
 *   <li>8.1.6 Horn-Schunck and pyramidal Lucas-Kanade ({@link ClassicalFlow}) against zero
 *       flow and a CNN regressing flow from the stacked frames;</li>
 *   <li>8.1.7 RAFT-lite ({@link RecurrentFlowRefiner}) against the classical methods and a
 *       FlowNet-style {@link UNet2d}, on small (0-4 px) and large (6-10 px) motions;</li>
 *   <li>8.1.8 GC-Net-lite stereo ({@link StereoCostVolume}) against block matching and a
 *       {@link UNet2d} regressing disparity from the stacked pair;</li>
 *   <li>8.1.9 keypoints: direct regression (DeepPose), arg-max heatmaps and integral
 *       regression ({@link KeypointHeads}) on {@link ShapeScenes#pose} stick figures.</li>
 * </ul>
 */
public final class DensePrediction {

    /** The side of the flow, stereo and pose scenes. */
    public static final int SIDE = 24;

    private DensePrediction() {
    }

    // ------------------------------------------------------------------ 8.1.5 segmentation

    /** The segmentation networks. */
    public enum Segmenter {
        /** Stride-2 encoder, ASPP and a one-skip decoder. */
        DEEPLAB,
        /** {@code ShapeSegmentation}'s U-Net. */
        UNET,
        /** The U-Net without skips. */
        UNET_NO_SKIPS
    }

    /** The segmentation losses. */
    public enum SegLoss {
        /** Pixel-wise cross-entropy. */
        CROSS_ENTROPY,
        /** Soft Dice per class (core {@code Loss.dice}). */
        DICE,
        /** Lovász-softmax (core {@code Functional.lovaszSoftmax}). */
        LOVASZ,
        /** Cross-entropy plus Lovász-softmax. */
        CROSS_ENTROPY_LOVASZ
    }

    /**
     * The outcome of a segmenter.
     *
     * @param miou the mean IoU of the shape classes
     * @param pixelAccuracy the pixel accuracy
     * @param parameters the trainable scalars
     * @param macs multiply-adds per image
     */
    public record SegScore(double miou, double pixelAccuracy, long parameters, long macs) {
    }

    /** DeepLabv3+-lite: stem, stride-2 convolution, ASPP, upsampling, skip from the stem. */
    private static final class DeepLab {
        final Var stem;
        final Var stemB;
        final Var down;
        final Var downB;
        final AtrousPyramidPooling aspp;
        final Var fuse;
        final Var fuseB;
        final Var head;
        final Var headB;

        DeepLab(int classes, SplittableRandom r) {
            stem = VisionOps.convWeight(r, 8, 1, 3);
            stemB = VisionOps.zeros(8);
            down = VisionOps.convWeight(r, 16, 8, 3);
            downB = VisionOps.zeros(16);
            aspp = new AtrousPyramidPooling(16, 16, new int[] {1, 2, 3}, r);
            fuse = VisionOps.convWeight(r, 16, 24, 3);
            fuseB = VisionOps.zeros(16);
            head = VisionOps.convWeight(r, classes, 16, 1);
            headB = VisionOps.zeros(classes);
        }

        Var forward(Var x) {
            int h = x.shape().dim(2);
            int w = x.shape().dim(3);
            Var s = VisionOps.conv(x, stem, stemB).relu();
            Var d = VisionOps.conv(s, down, downB, 2, 1).relu();
            Var a = VisionOps.upsample(aspp.forward(d), h, w);
            Var f = VisionOps.conv(Var.concat(1, s, a), fuse, fuseB).relu();
            return VisionOps.conv(f, head, headB);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(stem, stemB, down, downB));
            p.addAll(aspp.parameters());
            p.addAll(List.of(fuse, fuseB, head, headB));
            return p;
        }

        long macs(int side) {
            int half = (side + 1) / 2;
            return VisionOps.convMacs(1, 8, 3, side, side) + VisionOps.convMacs(8, 16, 3, half, half) + aspp.macs(half)
                    + VisionOps.convMacs(24, 16, 3, side, side) + VisionOps.convMacs(16, head.shape().dim(0), 1, side,
                    side);
        }
    }

    private static long unetMacs(int in, int out, int base, int side) {
        long m = 0;
        int c = in;
        for (int l = 0; l < 3; l++) {
            int f = base << l;
            int s = side >> l;
            m += VisionOps.convMacs(c, f, 3, s, s) + VisionOps.convMacs(f, f, 3, s, s);
            c = f;
        }
        for (int l = 1; l >= 0; l--) {
            int f = base << l;
            int s = side >> l;
            m += VisionOps.convMacs(c, f, 3, s, s) + VisionOps.convMacs(2 * f, f, 3, s, s)
                    + VisionOps.convMacs(f, f, 3, s, s);
            c = f;
        }
        return m + VisionOps.convMacs(base, out, 1, side, side);
    }

    private static Tensor[] segmentationData(boolean multiScale, int n, long seed) {
        if (!multiScale) {
            return ShapeSegmentation.scenes(n, seed);
        }
        ShapeScenes gen = new ShapeScenes(SIDE, 0.15f);
        ShapeScenes.Scene[] s = gen.scenes(n, 1, 5, ShapeScenes.Layout.SPARSE, seed);
        int p = SIDE * SIDE;
        float[] lab = new float[n * p];
        for (int i = 0; i < n; i++) {
            int[] sem = s[i].semantic();
            for (int j = 0; j < p; j++) {
                lab[i * p + j] = sem[j];
            }
        }
        return new Tensor[] {ShapeScenes.images(s), Tensor.of(Shape.of(n * p), lab)};
    }

    private static Var segLoss(SegLoss kind, Var logits, Tensor labels, int classes) {
        return switch (kind) {
            case CROSS_ENTROPY -> Functional.crossEntropy(logits, labels);
            case DICE -> {
                Var p = Activation.SOFTMAX.forward(logits).permute(1, 0);
                Tensor oneHot = Tensor.of(Shape.of(classes, labels.shape().dim(0)), oneHot(labels, classes));
                yield Loss.dice(1f).compute(p, oneHot);
            }
            case LOVASZ -> Functional.lovaszSoftmax(Activation.SOFTMAX.forward(logits), labels);
            case CROSS_ENTROPY_LOVASZ -> Functional.crossEntropy(logits, labels)
                    .add(Functional.lovaszSoftmax(Activation.SOFTMAX.forward(logits), labels));
        };
    }

    private static float[] oneHot(Tensor labels, int classes) {
        float[] l = labels.toArray();
        float[] out = new float[classes * l.length];
        for (int i = 0; i < l.length; i++) {
            out[(int) l[i] * l.length + i] = 1;
        }
        return out;
    }

    /**
     * Trains a segmenter.
     *
     * @param net the network
     * @param loss the loss
     * @param multiScale false for {@code ShapeSegmentation}'s 12x12 two-shape scenes, true for
     *     24x24 {@link ShapeScenes} with 1 to 5 glyphs at three scales
     * @param steps Adam steps of 16 scenes
     * @param seed the seed
     * @return mean IoU and pixel accuracy on 128 fresh scenes
     */
    public static SegScore segment(Segmenter net, SegLoss loss, boolean multiScale, int steps, long seed) {
        int classes = ShapeSegmentation.CLASSES;
        int side = multiScale ? SIDE : ShapeSegmentation.SIDE;
        SplittableRandom r = new SplittableRandom(seed);
        Function<Var, Var> f;
        List<Var> params;
        long macs;
        if (net == Segmenter.DEEPLAB) {
            DeepLab d = new DeepLab(classes, r);
            f = d::forward;
            params = d.parameters();
            macs = d.macs(side);
        } else {
            UNet2d u = new UNet2d(1, classes, 8, 2, net == Segmenter.UNET, 0, r);
            f = u::forward;
            params = u.parameters();
            macs = unetMacs(1, classes, 8, side);
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom data = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = segmentationData(multiScale, 16, data.nextLong());
            Var logits = f.apply(Var.of(d[0])).permute(0, 2, 3, 1).reshape(16 * side * side, classes);
            segLoss(loss, logits, d[1], classes).backward();
            opt.step(params);
        }
        Tensor[] test = segmentationData(multiScale, 128, seed + 99);
        float[] pred = Var.noGrad(() -> f.apply(Var.of(test[0]))).value().permute(0, 2, 3, 1)
                .reshape(128 * side * side, classes).argmax(1).toArray();
        float[] truth = test[1].toArray();
        long[] inter = new long[classes];
        long[] union = new long[classes];
        int correct = 0;
        for (int i = 0; i < truth.length; i++) {
            int p = (int) pred[i];
            int t = (int) truth[i];
            correct += p == t ? 1 : 0;
            union[t]++;
            if (p == t) {
                inter[t]++;
            } else {
                union[p]++;
            }
        }
        double miou = 0;
        int counted = 0;
        for (int c = 1; c < classes; c++) {
            if (union[c] > 0) {
                miou += (double) inter[c] / union[c];
                counted++;
            }
        }
        return new SegScore(miou / counted, (double) correct / truth.length, VisionOps.count(params), macs);
    }

    // ------------------------------------------------------------------ 8.1.6 / 8.1.7 optical flow

    /** The flow estimators. */
    public enum FlowMethod {
        /** Zero flow. */
        ZERO,
        /** Horn-Schunck, 3 pyramid levels. */
        HORN_SCHUNCK,
        /** Pyramidal Lucas-Kanade, 3 levels. */
        LUCAS_KANADE
    }

    /**
     * Frame pairs with exact flow.
     *
     * @param n how many
     * @param large true for 6-10 px motions, false for 0-4 px
     * @param noise the pixel noise
     * @param seed the seed
     * @return {@code {first [n,1,s,s], second [n,1,s,s], flow [n,2,s,s]}}
     */
    public static Tensor[] flowPairs(int n, boolean large, float noise, long seed) {
        ShapeScenes gen = new ShapeScenes(SIDE, noise);
        SplittableRandom r = new SplittableRandom(seed);
        int p = SIDE * SIDE;
        float[] a = new float[n * p];
        float[] b = new float[n * p];
        float[] f = new float[n * 2 * p];
        for (int i = 0; i < n; i++) {
            ShapeScenes.Video v = gen.video(2, 3, 8, large ? 6 : 0, large ? 10 : 4, false, r);
            System.arraycopy(v.frames()[0], 0, a, i * p, p);
            System.arraycopy(v.frames()[1], 0, b, i * p, p);
            System.arraycopy(v.flow()[0], 0, f, i * 2 * p, 2 * p);
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 1, SIDE, SIDE), a), Tensor.of(Shape.of(n, 1, SIDE, SIDE), b),
            Tensor.of(Shape.of(n, 2, SIDE, SIDE), f)};
    }

    /** {all-pixel EPE, EPE over the pixels that move}. */
    private static double[] meanEpe(float[] flows, float[] truth, int n) {
        int p = SIDE * SIDE;
        double all = 0;
        double moving = 0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                int u = i * 2 * p + j;
                double e = Math.hypot(flows[u] - truth[u], flows[u + p] - truth[u + p]);
                all += e;
                if (truth[u] != 0 || truth[u + p] != 0) {
                    moving += e;
                    count++;
                }
            }
        }
        return new double[] {all / (n * p), moving / Math.max(1, count)};
    }

    /**
     * Scores a training-free flow method.
     *
     * @param method the method
     * @param large whether motions are 6-10 px
     * @param noise the pixel noise
     * @param seed the seed of the 32 test pairs
     * @return the average end-point error over all pixels and over the moving pixels
     */
    public static double[] classicalFlow(FlowMethod method, boolean large, float noise, long seed) {
        Tensor[] t = flowPairs(32, large, noise, seed);
        int p = SIDE * SIDE;
        float[] a = t[0].toArray();
        float[] b = t[1].toArray();
        float[] out = new float[32 * 2 * p];
        for (int i = 0; i < 32; i++) {
            float[] f1 = java.util.Arrays.copyOfRange(a, i * p, (i + 1) * p);
            float[] f2 = java.util.Arrays.copyOfRange(b, i * p, (i + 1) * p);
            float[] f = switch (method) {
                case ZERO -> new float[2 * p];
                case HORN_SCHUNCK -> ClassicalFlow.hornSchunck(f1, f2, SIDE, SIDE, 0.3f, 60, 3);
                case LUCAS_KANADE -> ClassicalFlow.lucasKanade(f1, f2, SIDE, SIDE, 2, 3, 3);
            };
            System.arraycopy(f, 0, out, i * 2 * p, 2 * p);
        }
        return meanEpe(out, t[2].toArray(), 32);
    }

    /** The learned flow regressors. */
    public enum FlowNet {
        /** The VisionShootout CNN layers, same-padded, on the stacked frames. */
        CNN,
        /** A FlowNet-style U-Net on the stacked frames. */
        UNET
    }

    /**
     * Trains a network that regresses flow from the stacked frames.
     *
     * @param net the network
     * @param large whether motions are 6-10 px
     * @param steps Adam steps of 8 pairs
     * @param seed the seed
     * @return the average end-point error over all and over moving pixels of 32 test pairs
     */
    public static double[] regressedFlow(FlowNet net, boolean large, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Function<Var, Var> f;
        List<Var> params;
        if (net == FlowNet.UNET) {
            UNet2d u = new UNet2d(2, 2, 8, 2, true, 0, r);
            f = u::forward;
            params = u.parameters();
        } else {
            Var w1 = VisionOps.convWeight(r, 16, 2, 3);
            Var b1 = VisionOps.zeros(16);
            Var w2 = VisionOps.convWeight(r, 32, 16, 3);
            Var b2 = VisionOps.zeros(32);
            Var w3 = VisionOps.convWeight(r, 2, 32, 3);
            Var b3 = VisionOps.zeros(2);
            f = x -> VisionOps.conv(VisionOps.conv(VisionOps.conv(x, w1, b1).relu(), w2, b2).relu(), w3, b3);
            params = List.of(w1, b1, w2, b2, w3, b3);
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = flowPairs(8, large, 0.02f, r.nextLong());
            Var in = Var.concat(1, Var.of(d[0]), Var.of(d[1]));
            f.apply(in).sub(d[2]).square().sum(1, false).mean().backward();
            opt.step(params);
        }
        Tensor[] t = flowPairs(32, large, 0.02f, seed + 99);
        Tensor out = Var.noGrad(() -> f.apply(Var.concat(1, Var.of(t[0]), Var.of(t[1])))).value();
        return meanEpe(out.toArray(), t[2].toArray(), 32);
    }

    /**
     * Trains RAFT-lite and scores it at several test-time iteration counts.
     *
     * @param large whether motions are 6-10 px
     * @param steps Adam steps of 4 pairs
     * @param trainIterations the refinement iterations while training
     * @param testIterations the iteration counts to score
     * @param seed the seed
     * @return per test iteration count the end-point error over all and over moving pixels
     *     of 32 test pairs
     */
    public static double[][] raft(boolean large, int steps, int trainIterations, int[] testIterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        RecurrentFlowRefiner raft = new RecurrentFlowRefiner(12, 24, 3, 1, r);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = flowPairs(4, large, 0.02f, r.nextLong());
            RecurrentFlowRefiner.sequenceLoss(raft.predict(Var.of(d[0]), Var.of(d[1]), trainIterations), d[2], 0.8f, true)
                    .backward();
            opt.step(raft.parameters());
        }
        Tensor[] t = flowPairs(32, large, 0.02f, seed + 99);
        int max = 0;
        for (int it : testIterations) {
            max = Math.max(max, it);
        }
        int most = max;
        List<Var> preds = Var.noGrad(() -> raft.predict(Var.of(t[0]), Var.of(t[1]), most));
        double[][] out = new double[testIterations.length][];
        for (int i = 0; i < out.length; i++) {
            out[i] = meanEpe(preds.get(testIterations[i] - 1).value().toArray(), t[2].toArray(), 32);
        }
        return out;
    }

    // ------------------------------------------------------------------ 8.1.8 stereo

    /** The largest disparity of the stereo scenes. */
    public static final int MAX_DISPARITY = 6;

    /**
     * Stereo pairs with exact disparity.
     *
     * @param n how many
     * @param seed the seed
     * @return {@code {left [n,1,s,s], right [n,1,s,s], disparity [n,1,s,s]}}
     */
    public static Tensor[] stereoPairs(int n, long seed) {
        ShapeScenes gen = new ShapeScenes(SIDE, 0.05f);
        SplittableRandom r = new SplittableRandom(seed);
        int p = SIDE * SIDE;
        float[] l = new float[n * p];
        float[] rr = new float[n * p];
        float[] d = new float[n * p];
        for (int i = 0; i < n; i++) {
            ShapeScenes.Stereo s = gen.stereo(MAX_DISPARITY, r);
            System.arraycopy(s.left(), 0, l, i * p, p);
            System.arraycopy(s.right(), 0, rr, i * p, p);
            System.arraycopy(s.disparity(), 0, d, i * p, p);
        }
        Shape shape = Shape.of(n, 1, SIDE, SIDE);
        return new Tensor[] {Tensor.of(shape, l), Tensor.of(shape, rr), Tensor.of(shape, d)};
    }

    /** The stereo methods. */
    public enum StereoMethod {
        /** Sum-of-absolute-differences block matching, winner takes all. */
        BLOCK_MATCHING,
        /** The cost volume with soft-argmin. */
        COST_VOLUME,
        /** A U-Net regressing disparity from the stacked pair. */
        UNET
    }

    /**
     * Scores a stereo method.
     *
     * @param method the method
     * @param steps Adam steps of 4 pairs (ignored by block matching)
     * @param seed the seed
     * @return {@code {end-point error, share of pixels off by more than 1 px}} on 32 test
     *     pairs
     */
    public static double[] stereo(StereoMethod method, int steps, long seed) {
        Tensor[] t = stereoPairs(32, seed + 99);
        float[] est;
        if (method == StereoMethod.BLOCK_MATCHING) {
            int p = SIDE * SIDE;
            float[] l = t[0].toArray();
            float[] rr = t[1].toArray();
            est = new float[32 * p];
            for (int i = 0; i < 32; i++) {
                float[] d = StereoCostVolume.blockMatching(java.util.Arrays.copyOfRange(l, i * p, (i + 1) * p),
                        java.util.Arrays.copyOfRange(rr, i * p, (i + 1) * p), SIDE, SIDE, MAX_DISPARITY, 2);
                System.arraycopy(d, 0, est, i * p, p);
            }
        } else {
            SplittableRandom r = new SplittableRandom(seed);
            Function<Var[], Var> f;
            List<Var> params;
            if (method == StereoMethod.COST_VOLUME) {
                StereoCostVolume net = new StereoCostVolume(MAX_DISPARITY, 8, r);
                f = x -> net.forward(List.of(x[0], x[1]), true);
                params = net.parameters();
            } else {
                UNet2d net = new UNet2d(2, 1, 8, 2, true, 0, r);
                f = x -> net.forward(Var.concat(1, x[0], x[1]));
                params = net.parameters();
            }
            Optimizer opt = Optimizer.adam(3e-3f);
            for (int s = 0; s < steps; s++) {
                Tensor[] d = stereoPairs(4, r.nextLong());
                f.apply(new Var[] {Var.of(d[0]), Var.of(d[1])}).sub(d[2]).abs().mean().backward();
                opt.step(params);
            }
            est = Var.noGrad(() -> f.apply(new Var[] {Var.of(t[0]), Var.of(t[1])})).value().toArray();
        }
        float[] truth = t[2].toArray();
        return new double[] {FlowMetrics.meanAbsoluteError(est, truth), FlowMetrics.badShare(est, truth, 1)};
    }

    // ------------------------------------------------------------------ 8.1.9 keypoints

    /** The number of joints of the stick figures. */
    public static final int JOINTS = 5;

    /**
     * The outcome of a keypoint head.
     *
     * @param pck the share of joints within 2 px
     * @param meanError the mean joint error in pixels
     * @param rareError the mean joint error on the rare (upside-down) poses
     */
    public record PoseScore(double pck, double meanError, double rareError) {
    }

    private static Object[] poses(int n, double rare, long seed) {
        ShapeScenes gen = new ShapeScenes(SIDE, 0.05f);
        SplittableRandom r = new SplittableRandom(seed);
        float[] img = new float[n * SIDE * SIDE];
        float[][] joints = new float[n][];
        boolean[] isRare = new boolean[n];
        for (int i = 0; i < n; i++) {
            ShapeScenes.Pose p = gen.pose(JOINTS, 1, rare, r);
            System.arraycopy(p.image(), 0, img, i * SIDE * SIDE, SIDE * SIDE);
            joints[i] = p.joints();
            isRare[i] = p.rare();
        }
        return new Object[] {Tensor.of(Shape.of(n, 1, SIDE, SIDE), img), joints, isRare};
    }

    /**
     * Trains a keypoint head on a small backbone (stride-2 features with the stride-4 map
     * added back, or for direct regression the flattened stride-4 map) with 10% rare poses.
     *
     * @param mode the readout
     * @param steps Adam steps of 16 figures
     * @param seed the seed
     * @return the score on 128 test figures (half of them rare)
     */
    public static PoseScore keypoints(KeypointHeads.Mode mode, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        ConvBackbone backbone = new ConvBackbone(1, new int[] {8, 16, 16}, r);
        KeypointHeads head = mode == KeypointHeads.Mode.DIRECT
                ? new KeypointHeads(mode, 16, SIDE / 4, JOINTS, 4, r)
                : new KeypointHeads(mode, 16, SIDE / 2, JOINTS, 2, r);
        Function<Var, Var> features = x -> {
            List<Var> f = backbone.features(x);
            if (mode == KeypointHeads.Mode.DIRECT) {
                return f.get(1);
            }
            return f.get(0).add(VisionOps.upsample(f.get(1), SIDE / 2, SIDE / 2));
        };
        List<Var> params = new ArrayList<>(backbone.parameters());
        params.addAll(head.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            Object[] d = poses(16, 0.1, r.nextLong());
            head.loss(features.apply(Var.of((Tensor) d[0])), (float[][]) d[1]).backward();
            opt.step(params);
        }
        Object[] test = poses(128, 0.5, seed + 99);
        float[] pred = Var.noGrad(() -> head.forward(features.apply(Var.of((Tensor) test[0])))).value().toArray();
        float[][] truth = (float[][]) test[1];
        boolean[] rare = (boolean[]) test[2];
        double[] all = KeypointHeads.pck(pred, truth, 2);
        double rareErr = 0;
        int rareCount = 0;
        for (int i = 0; i < truth.length; i++) {
            if (rare[i]) {
                double[] one = KeypointHeads.pck(java.util.Arrays.copyOfRange(pred, i * 2 * JOINTS,
                        (i + 1) * 2 * JOINTS), new float[][] {truth[i]}, 2);
                rareErr += one[1];
                rareCount++;
            }
        }
        return new PoseScore(all[0], all[1], rareErr / Math.max(1, rareCount));
    }
}
