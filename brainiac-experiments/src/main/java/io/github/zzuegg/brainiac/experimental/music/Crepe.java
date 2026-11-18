package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A small CREPE: a 1-D convolutional network that classifies raw, per-frame normalised
 * waveform frames into pitch bins of 25 cents, trained with per-bin binary cross-entropy
 * against Gaussian-blurred one-hot targets (standard deviation 25 cents), so neighbouring bins
 * share credit. Unvoiced frames have all-zero targets; the largest activation is the voicing
 * confidence, and the pitch is the activation-weighted mean of the cents around the
 * strongest bin.
 *
 * <p>Paper: Kim, Salamon, Li, Bello, "CREPE: A Convolutional Representation for Pitch
 * Estimation" (ICASSP 2018), <a href="https://arxiv.org/abs/1802.06182">arXiv:1802.06182</a>.
 */
public final class Crepe {

    /** Samples per input frame. */
    public static final int FRAME = 512;
    /** Bins per semitone (25 cents). */
    public static final int BINS_PER_SEMITONE = 4;

    private final double fmin;
    private final int bins;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var w3;
    private final Var b3;
    private final Var head;
    private final Var headBias;
    private final List<Var> params;

    /**
     * Creates an untrained network.
     *
     * @param fmin the lowest pitch bin in Hz
     * @param fmax the highest pitch in Hz
     * @param channels convolution channels
     * @param seed the seed
     */
    public Crepe(double fmin, double fmax, int channels, long seed) {
        this.fmin = fmin;
        this.bins = (int) Math.ceil(12 * BINS_PER_SEMITONE * Math.log(fmax / fmin) / Math.log(2)) + 1;
        SplittableRandom r = new SplittableRandom(seed);
        w1 = Var.param(Tensor.normal(Shape.of(channels, 1, 32), 0, (float) Math.sqrt(2.0 / 32), r));
        b1 = Var.param(Tensor.zeros(Shape.of(channels)));
        w2 = Var.param(Tensor.normal(Shape.of(channels, channels, 8), 0, (float) Math.sqrt(2.0 / (8 * channels)), r));
        b2 = Var.param(Tensor.zeros(Shape.of(channels)));
        w3 = Var.param(Tensor.normal(Shape.of(channels, channels, 8), 0, (float) Math.sqrt(2.0 / (8 * channels)), r));
        b3 = Var.param(Tensor.zeros(Shape.of(channels)));
        int flat = channels * 16;
        head = Var.param(Tensor.normal(Shape.of(flat, bins), 0, (float) Math.sqrt(1.0 / flat), r));
        headBias = Var.param(Tensor.full(Shape.of(1, bins), -3f));
        params = List.of(w1, b1, w2, b2, w3, b3, head, headBias);
    }

    /** @return the number of pitch bins */
    public int bins() {
        return bins;
    }

    /** @return every parameter */
    public List<Var> parameters() {
        return params;
    }

    /**
     * @param bin a (fractional) bin
     * @return its frequency in Hz
     */
    public double frequency(double bin) {
        return fmin * Math.pow(2, bin / (12.0 * BINS_PER_SEMITONE));
    }

    private static Var conv(Var x, Var w, Var b, int stride) {
        int k = w.shape().dim(2);
        return Functional.conv(x, w, b, new int[] {stride}, new int[] {(k - 1) / 2}, new int[] {k / 2},
                new int[] {1}, 1);
    }

    private static Var pool(Var x, int to) {
        return Functional.adaptivePool(x, Functional.PoolKind.MAX, new int[] {to});
    }

    /**
     * The bin logits of normalised frames.
     *
     * @param frames {@code [batch, FRAME]}
     * @return {@code [batch, bins]}
     */
    public Var logits(Tensor frames) {
        int b = frames.shape().dim(0);
        Var x = Var.of(frames.reshape(b, 1, FRAME));
        Var h = pool(conv(x, w1, b1, 4).relu(), 64);
        h = pool(conv(h, w2, b2, 1).relu(), 32);
        h = pool(conv(h, w3, b3, 1).relu(), 16);
        return h.reshape(b, -1).matmul(head).add(headBias);
    }

    /**
     * Zero-mean, unit-variance frames centred on every hop.
     *
     * @param x the signal
     * @param hop the hop
     * @return {@code [x.length / hop + 1][FRAME]}
     */
    public static float[][] frames(float[] x, int hop) {
        int n = x.length / hop + 1;
        float[][] out = new float[n][];
        for (int t = 0; t < n; t++) {
            float[] f = MusicFeatures.frame(x, t * hop, FRAME);
            double mean = 0;
            for (float v : f) {
                mean += v;
            }
            mean /= FRAME;
            double var = 0;
            for (float v : f) {
                var += (v - mean) * (v - mean);
            }
            double std = Math.sqrt(var / FRAME) + 1e-4;
            for (int j = 0; j < FRAME; j++) {
                f[j] = (float) ((f[j] - mean) / std);
            }
            out[t] = f;
        }
        return out;
    }

    private float[] target(float f0) {
        float[] y = new float[bins];
        if (f0 <= 0) {
            return y;
        }
        double c = 12 * BINS_PER_SEMITONE * Math.log(f0 / fmin) / Math.log(2);
        for (int i = 0; i < bins; i++) {
            double d = i - c;
            y[i] = (float) Math.exp(-d * d / 2);
        }
        return y;
    }

    /**
     * Trains on labelled frames.
     *
     * @param frames normalised frames
     * @param f0 their reference F0 (0 for unvoiced)
     * @param steps steps
     * @param batch batch size
     * @param lr learning rate
     * @param seed the seed
     * @return the last loss
     */
    public float train(List<float[]> frames, List<Float> f0, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        float last = 0;
        for (int s = 0; s < steps; s++) {
            float[][] xb = new float[batch][];
            float[][] yb = new float[batch][];
            for (int i = 0; i < batch; i++) {
                int j = r.nextInt(frames.size());
                xb[i] = frames.get(j);
                yb[i] = target(f0.get(j));
            }
            params.forEach(Var::zeroGrad);
            Var z = logits(MusicFeatures.stack(xb));
            Tensor y = MusicFeatures.stack(yb);
            // binary cross-entropy with logits: softplus(z) - y z, summed over bins as in CREPE
            Var loss = z.softplus().sub(z.mul(y)).sum(1, false).mean();
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
        return last;
    }

    /**
     * Tracks a signal.
     *
     * @param x the signal
     * @param hop the hop
     * @param threshold the voicing confidence threshold
     * @return the track
     */
    public PitchTrackers.Track track(float[] x, int hop, double threshold) {
        float[][] fr = frames(x, hop);
        float[] f0 = new float[fr.length];
        boolean[] voiced = new boolean[fr.length];
        for (int from = 0; from < fr.length; from += 128) {
            int to = Math.min(fr.length, from + 128);
            float[][] chunk = java.util.Arrays.copyOfRange(fr, from, to);
            float[] z = Var.noGrad(() -> logits(MusicFeatures.stack(chunk)).sigmoid()).value().toArray();
            for (int i = 0; i < chunk.length; i++) {
                int best = 0;
                for (int k = 1; k < bins; k++) {
                    if (z[i * bins + k] > z[i * bins + best]) {
                        best = k;
                    }
                }
                double num = 0;
                double den = 0;
                for (int k = Math.max(0, best - 4); k <= Math.min(bins - 1, best + 4); k++) {
                    num += k * z[i * bins + k];
                    den += z[i * bins + k];
                }
                f0[from + i] = (float) frequency(num / den);
                voiced[from + i] = z[i * bins + best] > threshold;
            }
        }
        return new PitchTrackers.Track(f0, voiced);
    }

    /**
     * Collects training frames from labelled signals.
     *
     * @param waves signals
     * @param f0s their per-hop F0
     * @param hop the hop
     * @param stride keep every {@code stride}-th frame
     * @param frames receives frames
     * @param labels receives labels
     */
    public static void collect(List<float[]> waves, List<float[]> f0s, int hop, int stride, List<float[]> frames,
            List<Float> labels) {
        for (int i = 0; i < waves.size(); i++) {
            float[][] fr = frames(waves.get(i), hop);
            for (int t = 0; t < fr.length && t < f0s.get(i).length; t += stride) {
                frames.add(fr[t]);
                labels.add(f0s.get(i)[t]);
            }
        }
    }
}
