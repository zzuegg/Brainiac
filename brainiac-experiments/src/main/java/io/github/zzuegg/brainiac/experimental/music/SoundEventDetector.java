package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Polyphonic sound event detection with a convolutional-recurrent network (CRNN): 3x3
 * convolutions over the log-mel spectrogram pooled over frequency only (time resolution is
 * kept), a bidirectional GRU (the core {@link Recurrent}) over time, and per-frame sigmoids
 * per class, so overlapping events are detected independently. Post-processing median-filters
 * each class's probabilities and thresholds them into events. {@link Architecture#FRAME_MLP}
 * (one frame at a time) and {@link Architecture#WINDOW_CNN} (the same convolutions without the
 * recurrence, so a fixed local context) are the baselines.
 *
 * <p>Paper: Cakir, Parascandolo, Heittola, Huttunen, Virtanen, "Convolutional Recurrent Neural
 * Networks for Polyphonic Sound Event Detection", IEEE/ACM TASLP 25(6), 2017,
 * <a href="https://arxiv.org/abs/1702.06286">arXiv:1702.06286</a>.
 */
public final class SoundEventDetector {

    /** Architectures. */
    public enum Architecture {
        /** Convolutions, frequency pooling and a bidirectional GRU. */
        CRNN,
        /** Convolutions and frequency pooling only. */
        WINDOW_CNN,
        /** A per-frame MLP. */
        FRAME_MLP
    }

    private final Architecture arch;
    private final int mels;
    private final int classes;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Layer gru;
    private final Var head;
    private final Var headBias;
    private final Var mlp1;
    private final Var mlpB;
    private final List<Var> params = new ArrayList<>();

    /**
     * Creates an untrained detector.
     *
     * @param arch the architecture
     * @param mels mel bands (divisible by 4)
     * @param classes event classes
     * @param frames frames per training clip
     * @param seed the seed
     */
    public SoundEventDetector(Architecture arch, int mels, int classes, int frames, long seed) {
        this.arch = arch;
        this.mels = mels;
        this.classes = classes;
        SplittableRandom r = new SplittableRandom(seed);
        w1 = VisionOps.convWeight(r, 8, 1, 3);
        b1 = VisionOps.zeros(8);
        w2 = VisionOps.convWeight(r, 16, 8, 3);
        b2 = VisionOps.zeros(16);
        int conv = 16 * (mels / 4);
        int units = 24;
        gru = arch == Architecture.CRNN ? Recurrent.gru(units).returnSequences(true)
                .bidirectional(Recurrent.Merge.CONCAT).create(Shape.of(frames, conv), r) : null;
        int in = arch == Architecture.CRNN ? 2 * units : conv;
        mlp1 = Var.param(Tensor.normal(Shape.of(mels, 64), 0, (float) Math.sqrt(2.0 / mels), r));
        mlpB = Var.param(Tensor.zeros(Shape.of(1, 64)));
        int headIn = arch == Architecture.FRAME_MLP ? 64 : in;
        head = Var.param(Tensor.normal(Shape.of(headIn, classes), 0, (float) Math.sqrt(1.0 / headIn), r));
        headBias = Var.param(Tensor.full(Shape.of(1, classes), -1f));
        if (arch == Architecture.FRAME_MLP) {
            params.addAll(List.of(mlp1, mlpB));
        } else {
            params.addAll(List.of(w1, b1, w2, b2));
        }
        if (gru != null) {
            params.addAll(gru.parameters());
        }
        params.addAll(List.of(head, headBias));
    }

    /** @return every parameter */
    public List<Var> parameters() {
        return params;
    }

    private static Var poolFrequency(Var x) {
        return Functional.pool(x, Functional.PoolKind.MAX, new int[] {1, 2}, new int[] {1, 2}, new int[] {0, 0},
                new int[] {0, 0}, new int[] {1, 1}, false, false, 1f);
    }

    /**
     * Per-frame event logits.
     *
     * @param x log-mel clips {@code [batch, frames, mels]}
     * @return {@code [batch, frames, classes]}
     */
    public Var logits(Tensor x) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        if (arch == Architecture.FRAME_MLP) {
            Var h = Var.of(x.reshape(b * t, mels)).matmul(mlp1).add(mlpB).relu();
            return h.matmul(head).add(headBias).reshape(b, t, classes);
        }
        Var h = Var.of(x.reshape(b, 1, t, mels));
        h = poolFrequency(VisionOps.conv(h, w1, b1).relu());
        h = poolFrequency(VisionOps.conv(h, w2, b2).relu());              // [b, 16, t, mels / 4]
        h = h.permute(0, 2, 1, 3).reshape(b, t, 16 * (mels / 4));
        if (gru != null) {
            h = gru.forward(h);
        }
        return h.matmul(head).add(headBias);
    }

    /**
     * Trains with per-frame, per-class binary cross-entropy.
     *
     * @param clips per clip {@code [frames][mels]}, all of one length
     * @param labels per clip {@code [frames][classes]} activity
     * @param steps steps
     * @param batch clips per step
     * @param lr learning rate
     * @param seed the seed
     * @return the last loss
     */
    public float train(List<float[][]> clips, List<boolean[][]> labels, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        int t = clips.get(0).length;
        float last = 0;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[batch * t * mels];
            float[] y = new float[batch * t * classes];
            for (int i = 0; i < batch; i++) {
                int c = r.nextInt(clips.size());
                for (int j = 0; j < t; j++) {
                    System.arraycopy(clips.get(c)[j], 0, x, (i * t + j) * mels, mels);
                    for (int k = 0; k < classes; k++) {
                        y[(i * t + j) * classes + k] = labels.get(c)[j][k] ? 1 : 0;
                    }
                }
            }
            params.forEach(Var::zeroGrad);
            Var z = logits(Tensor.of(Shape.of(batch, t, mels), x));
            Var loss = z.softplus().sub(z.mul(Tensor.of(Shape.of(batch, t, classes), y))).mean();
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
        return last;
    }

    /**
     * Detects events in one clip: probabilities, a median filter and a threshold.
     *
     * @param clip {@code [frames][mels]}
     * @param median the median filter width in frames (odd)
     * @param threshold the probability threshold
     * @return {@code [frames][classes]} activity
     */
    public boolean[][] detect(float[][] clip, int median, double threshold) {
        int t = clip.length;
        float[] x = new float[t * mels];
        for (int j = 0; j < t; j++) {
            System.arraycopy(clip[j], 0, x, j * mels, mels);
        }
        float[] p = Var.noGrad(() -> logits(Tensor.of(Shape.of(1, t, mels), x)).sigmoid()).value().toArray();
        boolean[][] out = new boolean[t][classes];
        float[] window = new float[median];
        for (int k = 0; k < classes; k++) {
            for (int j = 0; j < t; j++) {
                for (int d = 0; d < median; d++) {
                    int jj = Math.min(t - 1, Math.max(0, j + d - median / 2));
                    window[d] = p[jj * classes + k];
                }
                java.util.Arrays.sort(window);
                out[j][k] = window[median / 2] > threshold;
            }
        }
        return out;
    }
}
