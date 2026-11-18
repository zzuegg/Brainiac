package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A small Beat This!-style neural beat tracker: two (dilated) convolutions over time on
 * standardised log-mel frames, one multi-head self-attention layer with a residual
 * connection, and a per-frame beat logit; beats are picked as local maxima above 0.5
 * probability within a 7-frame window, with no dynamic Bayesian network.
 *
 * <p>The {@link Loss#SHIFT_TOLERANT} loss of Foscarin et al. max-pools the predicted logits
 * over {@code ±3} frames before a binary cross-entropy against single-frame annotations, and
 * ignores negatives within {@code ±6} frames of an annotation, so a peak a few frames off
 * costs nothing and the network is free to emit sharp peaks; {@link Loss#PLAIN} is the
 * ablation (frame-exact binary cross-entropy), which punishes near misses and smears the
 * peaks. Both weight positives by the negative-to-positive ratio.
 *
 * <p>Paper: Foscarin, Schlueter, Widmer, "Beat this! Accurate beat tracking without DBN
 * postprocessing" (ISMIR 2024), <a href="https://arxiv.org/abs/2407.21658">arXiv:2407.21658</a>.
 */
public final class NeuralBeatTracker {

    /** Training losses. */
    public enum Loss {
        /** Max-pooled, near-negative-masked binary cross-entropy. */
        SHIFT_TOLERANT,
        /** Plain binary cross-entropy. */
        PLAIN
    }

    private static final int HEADS = 2;
    private final int mels;
    private final int channels;
    private final Var c1;
    private final Var cb1;
    private final Var c2;
    private final Var cb2;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var out;
    private final Var outBias;
    private final List<Var> params;

    /**
     * Creates an untrained tracker.
     *
     * @param mels mel bands of the input
     * @param channels hidden channels (divisible by 2)
     * @param seed the seed
     */
    public NeuralBeatTracker(int mels, int channels, long seed) {
        this.mels = mels;
        this.channels = channels;
        SplittableRandom r = new SplittableRandom(seed);
        c1 = Var.param(Tensor.normal(Shape.of(channels, mels, 5), 0, (float) Math.sqrt(2.0 / (5 * mels)), r));
        cb1 = Var.param(Tensor.zeros(Shape.of(channels)));
        c2 = Var.param(Tensor.normal(Shape.of(channels, channels, 5), 0, (float) Math.sqrt(2.0 / (5 * channels)), r));
        cb2 = Var.param(Tensor.zeros(Shape.of(channels)));
        float s = (float) Math.sqrt(1.0 / channels);
        wq = Var.param(Tensor.normal(Shape.of(channels, channels), 0, s, r));
        wk = Var.param(Tensor.normal(Shape.of(channels, channels), 0, s, r));
        wv = Var.param(Tensor.normal(Shape.of(channels, channels), 0, s, r));
        wo = Var.param(Tensor.normal(Shape.of(channels, channels), 0, s, r));
        out = Var.param(Tensor.normal(Shape.of(channels, 1), 0, s, r));
        outBias = Var.param(Tensor.full(Shape.of(1), -2f));
        params = List.of(c1, cb1, c2, cb2, wq, wk, wv, wo, out, outBias);
    }

    /** @return every parameter */
    public List<Var> parameters() {
        return params;
    }

    private static Var conv(Var x, Var w, Var b, int dilation) {
        int span = dilation * (w.shape().dim(2) - 1);
        return Functional.conv(x, w, b, new int[] {1}, new int[] {span / 2}, new int[] {span - span / 2},
                new int[] {dilation}, 1);
    }

    /**
     * Beat logits.
     *
     * @param features standardised log-mel frames {@code [batch, frames, mels]}
     * @return {@code [batch, frames]}
     */
    public Var logits(Tensor features) {
        int b = features.shape().dim(0);
        int t = features.shape().dim(1);
        Var x = Var.of(features).permute(0, 2, 1);
        Var h = conv(x, c1, cb1, 1).relu();
        h = conv(h, c2, cb2, 2).relu().permute(0, 2, 1);          // [b, t, c]
        int d = channels / HEADS;
        Var q = h.matmul(wq).reshape(b, t, HEADS, d).permute(0, 2, 1, 3);
        Var k = h.matmul(wk).reshape(b, t, HEADS, d).permute(0, 2, 1, 3);
        Var v = h.matmul(wv).reshape(b, t, HEADS, d).permute(0, 2, 1, 3);
        Var a = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(d))).permute(0, 2, 1, 3).reshape(b, t,
                channels);
        h = h.add(a.matmul(wo));
        return h.matmul(out).reshape(b, t).add(outBias);
    }

    /**
     * Standardised log-mel features of a signal.
     *
     * @param x the signal
     * @param rate the sample rate
     * @param hop the hop
     * @param mels mel bands
     * @return {@code [frames][mels]}
     */
    public static float[][] features(float[] x, int rate, int hop, int mels) {
        float[][] m = MusicFeatures.logMel(x, rate, 512, hop, mels);
        double mean = 0;
        double sq = 0;
        int n = 0;
        for (float[] row : m) {
            for (float v : row) {
                mean += v;
                sq += v * v;
                n++;
            }
        }
        mean /= n;
        double std = Math.sqrt(Math.max(1e-9, sq / n - mean * mean));
        for (float[] row : m) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (float) ((row[j] - mean) / std);
            }
        }
        return m;
    }

    private Var loss(Var z, float[][] targets, Loss kind) {
        int b = targets.length;
        int t = targets[0].length;
        float[] y = new float[b * t];
        float[] w = new float[b * t];
        for (int i = 0; i < b; i++) {
            int pos = 0;
            for (int j = 0; j < t; j++) {
                pos += targets[i][j] > 0 ? 1 : 0;
            }
            float posWeight = (t - pos) / (float) Math.max(1, pos);
            for (int j = 0; j < t; j++) {
                y[i * t + j] = targets[i][j];
                boolean near = false;
                if (kind == Loss.SHIFT_TOLERANT && targets[i][j] == 0) {
                    for (int d = -6; d <= 6 && !near; d++) {
                        int jj = j + d;
                        near = jj >= 0 && jj < t && targets[i][jj] > 0;
                    }
                }
                w[i * t + j] = near ? 0 : targets[i][j] > 0 ? posWeight : 1;
            }
        }
        Var logit = z;
        if (kind == Loss.SHIFT_TOLERANT) {
            logit = Functional.pool(z.reshape(b, 1, t), Functional.PoolKind.MAX, new int[] {7}, new int[] {1},
                    new int[] {3}, new int[] {3}, new int[] {1}, false, false, 1f).reshape(b, t);
        }
        Tensor yt = Tensor.of(Shape.of(b, t), y);
        Tensor wt = Tensor.of(Shape.of(b, t), w);
        return logit.softplus().sub(logit.mul(yt)).mul(wt).sum().div((float) (b * t));
    }

    /**
     * Trains on annotated clips.
     *
     * @param features per clip {@code [frames][mels]}, all of one length
     * @param beats per clip its beat times
     * @param fps frames per second
     * @param kind the loss
     * @param steps steps
     * @param batch clips per step
     * @param lr learning rate
     * @param seed the seed
     * @return the last loss
     */
    public float train(List<float[][]> features, List<double[]> beats, double fps, Loss kind, int steps, int batch,
            float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        int t = features.get(0).length;
        float last = 0;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[batch * t * mels];
            float[][] y = new float[batch][t];
            for (int i = 0; i < batch; i++) {
                int c = r.nextInt(features.size());
                float[][] f = features.get(c);
                for (int j = 0; j < t; j++) {
                    System.arraycopy(f[j], 0, x, (i * t + j) * mels, mels);
                }
                for (double bt : beats.get(c)) {
                    int j = (int) Math.round(bt * fps);
                    if (j >= 0 && j < t) {
                        y[i][j] = 1;
                    }
                }
            }
            params.forEach(Var::zeroGrad);
            Var l = loss(logits(Tensor.of(Shape.of(batch, t, mels), x)), y, kind);
            last = l.value().item();
            l.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
        return last;
    }

    /**
     * Beat probabilities of one clip.
     *
     * @param features {@code [frames][mels]}
     * @return per frame the beat probability
     */
    public float[] activation(float[][] features) {
        int t = features.length;
        float[] x = new float[t * mels];
        for (int j = 0; j < t; j++) {
            System.arraycopy(features[j], 0, x, j * mels, mels);
        }
        return Var.noGrad(() -> logits(Tensor.of(Shape.of(1, t, mels), x)).sigmoid()).value().toArray();
    }

    /**
     * Beats of one clip by peak picking.
     *
     * @param features {@code [frames][mels]}
     * @param fps frames per second
     * @param threshold the probability threshold
     * @return beat times in seconds
     */
    public double[] beats(float[][] features, double fps, double threshold) {
        float[] p = activation(features);
        int[] peaks = MusicMetrics.pickPeaks(p, 3, 0, threshold, 4);
        double[] out = new double[peaks.length];
        for (int i = 0; i < peaks.length; i++) {
            out[i] = peaks[i] / fps;
        }
        return out;
    }
}
