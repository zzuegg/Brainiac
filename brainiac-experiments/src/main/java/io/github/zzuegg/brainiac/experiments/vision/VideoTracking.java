package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.experimental.vision.ConvLstmCell;
import io.github.zzuegg.brainiac.experimental.vision.FrameMetrics;
import io.github.zzuegg.brainiac.experimental.vision.SimVp;
import io.github.zzuegg.brainiac.experimental.vision.TrackingByDetection;
import io.github.zzuegg.brainiac.experimental.vision.TrackingMetrics;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 8 items 8.1.10 and 8.1.11 on {@link ShapeScenes#video} sequences.
 *
 * <p>8.1.10 multi-object tracking: five 8-pixel tiles cross each other in 40x40 frames for 30
 * frames, and pass behind a static occluding band. Detections are the true boxes with 0.5 px
 * noise and 5% dropout, scored 0.6 to 1; boxes partly hidden by the band are detected with
 * 80% probability only weakly (score 0.15 to 0.45, 0.8 px noise); and one false positive in
 * two frames scores 0.1 to 0.4. This separates tracker errors from detector errors. Compared:
 * {@link TrackingByDetection} SORT, ByteTrack, SORT without Kalman prediction and greedy
 * nearest-centroid association, by {@link TrackingMetrics} (MOTA, IDF1, ID switches).
 *
 * <p>8.1.11 video prediction: two tiles bouncing in 16x16 frames; four context frames, four
 * predicted. Compared: copying the last frame, an autoregressive {@link UNet2d} next-frame
 * predictor, a {@link ConvLstmCell} predictor (stride-2 encoder, ConvLSTM, decoder, its own
 * predictions fed back) and {@link SimVp}, by per-frame MSE and SSIM against the horizon and
 * by training time.
 */
public final class VideoTracking {

    private VideoTracking() {
    }

    // ------------------------------------------------------------------ 8.1.10 tracking

    /**
     * The tracking metrics summed over sequences.
     *
     * @param mota the mean MOTA
     * @param idf1 the mean IDF1
     * @param idSwitches the total ID switches
     */
    public record TrackScore(double mota, double idf1, int idSwitches) {
    }

    /**
     * Tracks synthetic detections.
     *
     * @param mode the association scheme
     * @param videos the number of sequences
     * @param seed the seed
     * @return the metrics
     */
    public static TrackScore track(TrackingByDetection.Mode mode, int videos, long seed) {
        ShapeScenes gen = new ShapeScenes(40, 0f);
        SplittableRandom r = new SplittableRandom(seed);
        double mota = 0;
        double idf1 = 0;
        int switches = 0;
        for (int v = 0; v < videos; v++) {
            ShapeScenes.Video video = gen.video(30, 5, 8, 1f, 2f, true, r);
            TrackingByDetection tracker = new TrackingByDetection(mode, 0.5f, 0.1f, 0.3, 3);
            List<List<TrackingByDetection.Output>> truth = new ArrayList<>();
            List<List<TrackingByDetection.Output>> outputs = new ArrayList<>();
            for (int t = 0; t < video.length(); t++) {
                List<TrackingByDetection.Output> gt = new ArrayList<>();
                List<TrackingByDetection.Detection> dets = new ArrayList<>();
                for (int o = 0; o < video.objects(); o++) {
                    float[] box = java.util.Arrays.copyOfRange(video.boxes()[t], 4 * o, 4 * o + 4);
                    gt.add(new TrackingByDetection.Output(o, box));
                    if (video.visible()[t][o]) {
                        if (r.nextDouble() >= 0.05) {
                            dets.add(new TrackingByDetection.Detection(jitter(box, 0.5, r), 0.6f + 0.4f * r.nextFloat()));
                        }
                    } else if (r.nextDouble() < 0.8) {
                        dets.add(new TrackingByDetection.Detection(jitter(box, 0.8, r), 0.15f + 0.3f * r.nextFloat()));
                    }
                }
                if (r.nextBoolean()) {
                    float x = r.nextFloat() * 32;
                    float y = r.nextFloat() * 32;
                    dets.add(new TrackingByDetection.Detection(new float[] {x, y, x + 8, y + 8},
                            0.1f + 0.3f * r.nextFloat()));
                }
                truth.add(gt);
                outputs.add(tracker.update(dets));
            }
            TrackingMetrics.Result res = TrackingMetrics.evaluate(truth, outputs);
            mota += res.mota();
            idf1 += res.idf1();
            switches += res.idSwitches();
        }
        return new TrackScore(mota / videos, idf1 / videos, switches);
    }

    private static float[] jitter(float[] box, double sigma, SplittableRandom r) {
        float[] b = box.clone();
        for (int i = 0; i < 4; i++) {
            b[i] += (float) (sigma * r.nextGaussian());
        }
        return b;
    }

    // ------------------------------------------------------------------ 8.1.11 video prediction

    /** The frame side of the prediction videos. */
    public static final int SIDE = 16;
    /** Context frames. */
    public static final int CONTEXT = 4;
    /** Predicted frames. */
    public static final int FUTURE = 4;

    /** The video predictors. */
    public enum Predictor {
        /** Repeat the last context frame. */
        COPY_LAST,
        /** A U-Net from the last four frames to the next, rolled out. */
        UNET,
        /** A ConvLSTM, its predictions fed back. */
        CONV_LSTM,
        /** SimVP, all future frames at once. */
        SIM_VP
    }

    /**
     * The outcome of a predictor.
     *
     * @param mse the mean squared error at each horizon
     * @param ssim the SSIM at each horizon
     * @param seconds the training time
     */
    public record PredictionScore(double[] mse, double[] ssim, double seconds) {
    }

    /**
     * Videos as tensors.
     *
     * @param n how many
     * @param seed the seed
     * @return {@code [n, CONTEXT + FUTURE, SIDE, SIDE]}
     */
    public static Tensor videos(int n, long seed) {
        ShapeScenes gen = new ShapeScenes(SIDE, 0f);
        SplittableRandom r = new SplittableRandom(seed);
        int len = CONTEXT + FUTURE;
        int p = SIDE * SIDE;
        float[] out = new float[n * len * p];
        for (int i = 0; i < n; i++) {
            ShapeScenes.Video v = gen.video(len, 2, 4, 0.5f, 1.5f, false, r);
            for (int t = 0; t < len; t++) {
                System.arraycopy(v.frames()[t], 0, out, (i * len + t) * p, p);
            }
        }
        return Tensor.of(Shape.of(n, len, SIDE, SIDE), out);
    }

    /** The ConvLSTM predictor: stride-2 encoder, cell, decoder. */
    private static final class LstmPredictor {
        final Var enc;
        final Var encB;
        final ConvLstmCell cell;
        final Var dec1;
        final Var dec1B;
        final Var dec2;
        final Var dec2B;

        LstmPredictor(SplittableRandom r) {
            enc = VisionOps.convWeight(r, 8, 1, 3);
            encB = VisionOps.zeros(8);
            cell = new ConvLstmCell(8, 16, 3, r);
            dec1 = VisionOps.convWeight(r, 8, 16, 3);
            dec1B = VisionOps.zeros(8);
            dec2 = VisionOps.convWeight(r, 1, 9, 3);
            dec2B = VisionOps.zeros(1);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(enc, encB));
            p.addAll(cell.parameters());
            p.addAll(List.of(dec1, dec1B, dec2, dec2B));
            return p;
        }

        /** Next-frame predictions for steps 1 .. CONTEXT + FUTURE - 1: teacher-forced in the context, then fed back. */
        List<Var> rollout(Tensor video) {
            int b = video.shape().dim(0);
            Var v = Var.of(video);
            Var[] state = cell.zeroState(b, SIDE / 2, SIDE / 2);
            List<Var> preds = new ArrayList<>();
            Var frame = v.slice(1, 0, 1);
            for (int t = 0; t < CONTEXT + FUTURE - 1; t++) {
                state = cell.step(VisionOps.conv(frame, enc, encB, 2, 1).relu(), state);
                Var h = VisionOps.conv(VisionOps.upsample(state[0], SIDE, SIDE), dec1, dec1B).relu();
                Var next = VisionOps.conv(Var.concat(1, h, frame), dec2, dec2B).add(frame);
                preds.add(next);
                frame = t + 1 < CONTEXT ? v.slice(1, t + 1, t + 2) : next;
            }
            return preds;
        }
    }

    /**
     * Trains and scores a predictor.
     *
     * @param kind the predictor
     * @param steps Adam steps of 8 videos
     * @param seed the seed
     * @return per-horizon MSE and SSIM on 32 test videos, and the training time
     */
    public static PredictionScore predict(Predictor kind, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor test = videos(32, seed + 99);
        long start = System.nanoTime();
        Tensor predicted;
        Tensor context = test.slice(1, 0, CONTEXT);
        switch (kind) {
            case COPY_LAST -> predicted = context.slice(1, CONTEXT - 1, CONTEXT).expand(Shape.of(32, FUTURE, SIDE, SIDE));
            case SIM_VP -> {
                SimVp net = new SimVp(CONTEXT, FUTURE, 8, 2, r);
                Optimizer opt = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    Tensor v = videos(8, r.nextLong());
                    net.forward(Var.of(v.slice(1, 0, CONTEXT))).sub(v.slice(1, CONTEXT, CONTEXT + FUTURE)).square()
                            .mean().backward();
                    opt.step(net.parameters());
                }
                predicted = Var.noGrad(() -> net.forward(Var.of(context))).value();
            }
            case UNET -> {
                UNet2d net = new UNet2d(CONTEXT, 1, 8, 2, true, 0, r);
                Optimizer opt = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    Tensor v = videos(8, r.nextLong());
                    int t0 = r.nextInt(FUTURE);
                    net.forward(Var.of(v.slice(1, t0, t0 + CONTEXT))).sub(v.slice(1, t0 + CONTEXT, t0 + CONTEXT + 1))
                            .square().mean().backward();
                    opt.step(net.parameters());
                }
                predicted = Var.noGrad(() -> {
                    Var window = Var.of(context);
                    List<Var> outs = new ArrayList<>();
                    for (int t = 0; t < FUTURE; t++) {
                        Var next = net.forward(window);
                        outs.add(next);
                        window = Var.concat(1, window.slice(1, 1, CONTEXT), next);
                    }
                    return Var.concat(1, outs.toArray(new Var[0]));
                }).value();
            }
            case CONV_LSTM -> {
                LstmPredictor net = new LstmPredictor(r);
                Optimizer opt = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    Tensor v = videos(8, r.nextLong());
                    List<Var> preds = net.rollout(v);
                    Var loss = null;
                    for (int t = 0; t < preds.size(); t++) {
                        Var l = preds.get(t).sub(v.slice(1, t + 1, t + 2)).square().mean();
                        loss = loss == null ? l : loss.add(l);
                    }
                    loss.backward();
                    opt.step(net.parameters());
                }
                predicted = Var.noGrad(() -> {
                    List<Var> preds = net.rollout(test);
                    return Var.concat(1, preds.subList(CONTEXT - 1, CONTEXT + FUTURE - 1).toArray(new Var[0]));
                }).value();
            }
            default -> throw new IllegalArgumentException(kind.toString());
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        float[] pred = predicted.toArray();
        float[] truth = test.slice(1, CONTEXT, CONTEXT + FUTURE).toArray();
        int p = SIDE * SIDE;
        double[] mse = new double[FUTURE];
        double[] ssim = new double[FUTURE];
        for (int i = 0; i < 32; i++) {
            for (int t = 0; t < FUTURE; t++) {
                int o = (i * FUTURE + t) * p;
                float[] a = java.util.Arrays.copyOfRange(pred, o, o + p);
                float[] b = java.util.Arrays.copyOfRange(truth, o, o + p);
                mse[t] += FrameMetrics.mse(a, b) / 32;
                ssim[t] += FrameMetrics.ssim(a, b, SIDE, SIDE) / 32;
            }
        }
        return new PredictionScore(mse, ssim, seconds);
    }
}
