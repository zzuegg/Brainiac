package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Clip classifiers sharing one small 2-D trunk (two {@code 3 x 3} convolutions, each followed by
 * ReLU and {@code 2 x 2} max pooling, then global average pooling and a linear head):
 *
 * <ul>
 *   <li>{@link Kind#SINGLE_FRAME}: the middle frame only (baseline);
 *   <li>{@link Kind#LATE_FUSION}: per-frame features averaged over time, so blind to order;
 *       {@link Kind#FRAME_DIFFERENCE}: the same on frame differences {@code x_{t+1} - x_t};
 *   <li>{@link Kind#TSM}: a residual temporal shift before the second convolution, a quarter
 *       of the channels moved one frame forward and a quarter one frame back (zero padded), at
 *       no extra multiply-adds;
 *   <li>{@link Kind#R2PLUS1D}: a temporal convolution (kernel 3 over frames) with ReLU between
 *       the two spatial convolutions, a factorised 3-D convolution;
 *   <li>{@link Kind#CONV_LSTM}: a {@link ConvLstmCell} over the first stage's maps, the last
 *       hidden state feeding the second stage (baseline).
 * </ul>
 *
 * <p>Papers: Lin, Gan, Han, "TSM: Temporal Shift Module for Efficient Video Understanding",
 * ICCV 2019, <a href="https://arxiv.org/abs/1811.08383">arXiv:1811.08383</a>; Tran et al., "A
 * Closer Look at Spatiotemporal Convolutions for Action Recognition", CVPR 2018,
 * <a href="https://arxiv.org/abs/1711.11248">arXiv:1711.11248</a>; Karpathy et al., "Large-scale
 * Video Classification with Convolutional Neural Networks", CVPR 2014.
 */
public final class VideoClassifiers {

    /** Architectures. */
    public enum Kind {
        /** the middle frame */
        SINGLE_FRAME,
        /** average of per-frame features */
        LATE_FUSION,
        /** late fusion on frame differences */
        FRAME_DIFFERENCE,
        /** temporal shift module */
        TSM,
        /** (2+1)-D convolution */
        R2PLUS1D,
        /** convolutional LSTM */
        CONV_LSTM
    }

    private final Kind kind;
    private final int channels;
    private final int classes;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var skip;
    private final Var temporal;
    private final ConvLstmCell lstm;
    private final Var head;
    private final List<Var> params = new ArrayList<>();

    /**
     * Creates a classifier.
     *
     * @param kind the architecture
     * @param channels the first stage width (the second is twice that)
     * @param classes the number of classes
     * @param seed the seed
     */
    public VideoClassifiers(Kind kind, int channels, int classes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.kind = kind;
        this.channels = channels;
        this.classes = classes;
        w1 = VisionOps.convWeight(r, channels, 1, 3);
        b1 = VisionOps.zeros(channels);
        w2 = VisionOps.convWeight(r, 2 * channels, channels, 3);
        b2 = VisionOps.zeros(2 * channels);
        head = Var.param(Tensor.normal(Shape.of(2 * channels, classes), 0, (float) Math.sqrt(1.0 / (2 * channels)), r));
        params.addAll(List.of(w1, b1, w2, b2, head));
        skip = kind == Kind.TSM ? VisionOps.convWeight(r, 2 * channels, channels, 1) : null;
        temporal = kind == Kind.R2PLUS1D ? Var.param(Tensor.normal(Shape.of(channels, channels, 3, 1, 1), 0,
                (float) Math.sqrt(2.0 / (3 * channels)), r)) : null;
        lstm = kind == Kind.CONV_LSTM ? new ConvLstmCell(channels, channels, 3, r) : null;
        if (skip != null) {
            params.add(skip);
        }
        if (temporal != null) {
            params.add(temporal);
        }
        if (lstm != null) {
            params.addAll(lstm.parameters());
        }
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    private static Var shift(Var x, int b, int t, int c, int h, int w) {
        Var v = x.reshape(b, t, c, h, w);
        int q = Math.max(1, c / 4);
        Var zero = Var.of(Tensor.zeros(Shape.of(b, 1, q, h, w)));
        Var fwd = Var.concat(1, zero, v.slice(1, 0, t - 1).slice(2, 0, q));          // frame t sees t - 1
        Var bwd = Var.concat(1, v.slice(1, 1, t).slice(2, q, 2 * q), zero);           // frame t sees t + 1
        Var rest = v.slice(2, 2 * q, c);
        return Var.concat(2, fwd, bwd, rest).reshape(b * t, c, h, w);
    }

    /**
     * Logits for clips.
     *
     * @param clips {@code [b][t * side * side]}
     * @param t frames per clip
     * @param side the frame side
     * @return {@code [b, classes]}
     */
    public Var logits(float[][] clips, int t, int side) {
        int b = clips.length;
        int n = side * side;
        int frames = switch (kind) {
            case SINGLE_FRAME -> 1;
            case FRAME_DIFFERENCE -> t - 1;
            default -> t;
        };
        float[] x = new float[b * frames * n];
        for (int i = 0; i < b; i++) {
            for (int f = 0; f < frames; f++) {
                for (int p = 0; p < n; p++) {
                    float v = switch (kind) {
                        case SINGLE_FRAME -> clips[i][(t / 2) * n + p];
                        case FRAME_DIFFERENCE -> clips[i][(f + 1) * n + p] - clips[i][f * n + p];
                        default -> clips[i][f * n + p];
                    };
                    x[(i * frames + f) * n + p] = v;
                }
            }
        }
        Var in = Var.of(Tensor.of(Shape.of(b * frames, 1, side, side), x));
        Var a = VisionOps.maxPool2(VisionOps.conv(in, w1, b1).relu());
        int h = side / 2;
        int c = channels;
        if (kind == Kind.R2PLUS1D) {
            Var v = a.reshape(b, frames, c, h, h).permute(0, 2, 1, 3, 4);
            v = Functional.conv(v, temporal, null, new int[] {1, 1, 1}, new int[] {1, 0, 0}, new int[] {1, 0, 0},
                    new int[] {1, 1, 1}, 1).relu();
            a = v.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, h);
        }
        if (kind == Kind.CONV_LSTM) {
            Var v = a.reshape(b, frames, c, h, h);
            Var[] state = lstm.zeroState(b, h, h);
            for (int f = 0; f < frames; f++) {
                state = lstm.step(v.slice(1, f, f + 1).reshape(b, c, h, h), state);
            }
            a = state[0];
            frames = 1;
        }
        Var z;
        if (kind == Kind.TSM) {
            z = VisionOps.conv(shift(a, b, frames, c, h, h), w2, b2).add(VisionOps.conv(a, skip, null)).relu();
        } else {
            z = VisionOps.conv(a, w2, b2).relu();
        }
        z = VisionOps.maxPool2(z);
        Var pooled = z.mean(3, false).mean(2, false).reshape(b, frames, 2 * c).mean(1, false);
        return pooled.matmul(head);
    }

    /**
     * Multiply-adds per clip of the convolutions and the head.
     *
     * @param t frames per clip
     * @param side the frame side
     * @return the count
     */
    public long macs(int t, int side) {
        int frames = kind == Kind.SINGLE_FRAME ? 1 : kind == Kind.FRAME_DIFFERENCE ? t - 1 : t;
        int h = side / 2;
        long m = frames * VisionOps.convMacs(1, channels, 3, side, side);
        int second = kind == Kind.CONV_LSTM ? 1 : frames;
        m += second * VisionOps.convMacs(channels, 2 * channels, 3, h, h);
        if (kind == Kind.TSM) {
            m += frames * VisionOps.convMacs(channels, 2 * channels, 1, h, h);
        }
        if (kind == Kind.R2PLUS1D) {
            m += (long) frames * channels * channels * 3 * h * h;
        }
        if (kind == Kind.CONV_LSTM) {
            m += frames * VisionOps.convMacs(2 * channels, 4 * channels, 3, h, h);
        }
        return m + 2L * channels * classes;
    }

    /**
     * Trains with cross-entropy.
     *
     * @param clips the clips
     * @param labels their classes
     * @param t frames per clip
     * @param side the frame side
     * @param steps steps
     * @param batch batch size
     * @param lr the learning rate
     * @param seed the seed
     */
    public void train(List<float[]> clips, int[] labels, int t, int side, int steps, int batch, float lr,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            float[][] x = new float[batch][];
            float[] y = new float[batch];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(clips.size());
                x[i] = clips.get(k);
                y[i] = labels[k];
            }
            params.forEach(Var::zeroGrad);
            Functional.crossEntropy(logits(x, t, side), Tensor.of(Shape.of(batch), y)).backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
    }

    /**
     * @param clip one clip
     * @param t frames
     * @param side the frame side
     * @return the predicted class
     */
    public int predict(float[] clip, int t, int side) {
        float[] l = Var.noGrad(() -> logits(new float[][] {clip}, t, side)).value().toArray();
        int best = 0;
        for (int k = 1; k < l.length; k++) {
            if (l[k] > l[best]) {
                best = k;
            }
        }
        return best;
    }
}
