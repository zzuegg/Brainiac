package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Self-supervised pitch estimation by transposition equivariance (PESTO), with the
 * shift-consistency frame weighting and pseudo-labelled voicing classifier of Bitra and
 * Beigi.
 *
 * <p>The input is a constant-Q-style frame (3 bins per semitone). Shifting it along frequency
 * by {@code k} bins transposes the sound by {@code k / 3} semitones, so a model whose output
 * distribution over pitch bins shifts by the same {@code k} needs no labels: a stack of 1-D
 * convolutions over frequency (translation-equivariant) feeds a Toeplitz-structured linear
 * layer (equivariant by construction) and a softmax; the convolutions are dilated (1, 4, 12)
 * so that one unit sees more than an octave of harmonics. Training takes two crops of one frame
 * offset by a random {@code k} and minimises
 *
 * <ul>
 *   <li>an equivariance loss: with {@code y(p) = sum_o alpha^o p_o}, the Huber loss of
 *       {@code y(p_2) / y(p_1) - alpha^-k};
 *   <li>a shift cross-entropy (SCE) between {@code p_1} shifted by {@code k} and
 *       {@code p_2}, both ways;
 *   <li>an invariance cross-entropy between a frame and a timbre-augmented copy (random
 *       spectral tilt, gain and noise);
 *   <li>a leak penalty {@code -log sum(shifted p)}: probability shifted past the edge of the
 *       output would otherwise make the shift cross-entropy vanish, a collapse onto the edge
 *       bins that the plain losses allow.
 * </ul>
 *
 * <p>The network then knows pitch up to a constant bin offset, fixed by one synthetic tone
 * of known pitch ({@link #calibrate}) rather than by data labels. Following Bitra and Beigi,
 * {@link #train} alternates an E-step, which scores every training frame by its SCE under a
 * random shift and weights it by {@code exp(-SCE / median SCE)} (noisy and unvoiced frames
 * are inconsistent and get small weights), with an M-step on the weighted losses; the final
 * weights pseudo-label frames as voiced (top 40%) or unvoiced (bottom 30%), and a logistic
 * voicing classifier on the output's peak, entropy, frame energy and SCE learns from them.
 *
 * <p>Papers: Riou, Lattner, Hadjeres, Peeters, "PESTO: Pitch Estimation with Self-supervised
 * Transposition-equivariant Objective" (ISMIR 2023),
 * <a href="https://arxiv.org/abs/2309.02265">arXiv:2309.02265</a>; Bitra, Beigi, "Lightweight
 * Self-Supervised Detection of Fundamental Frequency and Accurate Probability of Voicing in
 * Monophonic Music" (ICPRAM 2026), <a href="https://arxiv.org/abs/2601.11768">arXiv:2601.11768</a>.
 */
public final class EquivariantPitch {

    /** Bins per semitone of the input and output. */
    public static final int BINS_PER_SEMITONE = 3;
    /** The largest training shift in bins (four semitones). */
    public static final int MAX_SHIFT = 12;
    /** The FFT size of the front end. */
    public static final int N_FFT = 2048;

    private final int bins;
    private final int out;
    private final double fmin;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var w3;
    private final Var wd;
    private final Var bd;
    private final Var toeplitz;
    private final Tensor toeplitzIndex;
    private final List<Var> params;
    private double offset;
    private double[] voicing = new double[5];

    /**
     * Creates an untrained model.
     *
     * @param fmin the lowest frequency of the input crop
     * @param octaves the octaves the crop covers
     * @param channels convolution channels
     * @param seed the seed
     */
    public EquivariantPitch(double fmin, int octaves, int channels, long seed) {
        this.fmin = fmin;
        this.bins = 12 * BINS_PER_SEMITONE * octaves;
        this.out = 2 * bins;
        SplittableRandom r = new SplittableRandom(seed);
        w1 = Var.param(Tensor.normal(Shape.of(channels, 1, 9), 0, (float) Math.sqrt(2.0 / 9), r));
        b1 = Var.param(Tensor.zeros(Shape.of(channels)));
        w2 = Var.param(Tensor.normal(Shape.of(channels, channels, 9), 0, (float) Math.sqrt(2.0 / (9 * channels)), r));
        b2 = Var.param(Tensor.zeros(Shape.of(channels)));
        wd = Var.param(Tensor.normal(Shape.of(channels, channels, 9), 0, (float) Math.sqrt(2.0 / (9 * channels)), r));
        bd = Var.param(Tensor.zeros(Shape.of(channels)));
        w3 = Var.param(Tensor.normal(Shape.of(1, channels, 1), 0, (float) Math.sqrt(1.0 / channels), r));
        toeplitz = Var.param(Tensor.normal(Shape.of(bins + out - 1), 0, 0.05f, r));
        float[] idx = new float[bins * out];
        for (int i = 0; i < bins; i++) {
            for (int o = 0; o < out; o++) {
                idx[i * out + o] = o - i + bins - 1;
            }
        }
        toeplitzIndex = Tensor.of(Shape.of(bins * out), idx);
        params = List.of(w1, b1, w2, b2, wd, bd, w3, toeplitz);
    }

    /** @return input and output bins */
    public int bins() {
        return bins;
    }

    /** @return every parameter */
    public List<Var> parameters() {
        return params;
    }

    /**
     * The front end: log-compressed constant-Q frames with {@link #MAX_SHIFT} extra bins on
     * both sides, each frame scaled to a maximum of one, and the raw frame energies.
     *
     * @param x the signal at {@code rate}
     * @param rate the sample rate
     * @param hop the hop
     * @param fmin the lowest frequency of the centre crop
     * @param bins the bins of the centre crop
     * @param energy receives the log energy per frame (may be {@code null})
     * @return {@code [frames][bins + 2 MAX_SHIFT]}
     */
    public static float[][] frontEnd(float[] x, int rate, int hop, double fmin, int bins, float[] energy) {
        float[][] mag = MusicFeatures.magnitude(x, N_FFT, hop);
        double lo = fmin * Math.pow(2, -MAX_SHIFT / (12.0 * BINS_PER_SEMITONE));
        float[][] c = MusicFeatures.logFrequency(mag, rate, lo, 12 * BINS_PER_SEMITONE, bins + 2 * MAX_SHIFT, true);
        for (int t = 0; t < c.length; t++) {
            float max = 1e-3f;
            double sum = 0;
            for (float v : c[t]) {
                max = Math.max(max, v);
                sum += v;
            }
            if (energy != null && t < energy.length) {
                energy[t] = (float) Math.log(1e-4 + sum / c[t].length);
            }
            for (int i = 0; i < c[t].length; i++) {
                c[t][i] /= max;
            }
        }
        return c;
    }

    private Tensor crops(float[][] frames, int[] rows, int shift, float[][] tilt, SplittableRandom noise) {
        float[] v = new float[rows.length * bins];
        for (int i = 0; i < rows.length; i++) {
            float[] f = frames[rows[i]];
            for (int j = 0; j < bins; j++) {
                float a = f[MAX_SHIFT + shift + j];
                if (tilt != null) {
                    a = a * tilt[i][j] + (float) (0.03 * noise.nextGaussian());
                }
                v[i * bins + j] = a;
            }
        }
        return Tensor.of(Shape.of(rows.length, bins), v);
    }

    private static Var conv(Var x, Var w, Var b, int dilation) {
        int k = w.shape().dim(2);
        int span = dilation * (k - 1);
        return Functional.conv(x, w, b, new int[] {1}, new int[] {span / 2}, new int[] {span - span / 2},
                new int[] {dilation}, 1);
    }

    /**
     * The log pitch distribution of crops.
     *
     * @param crops {@code [batch, bins]}
     * @return log-probabilities {@code [batch, 2 bins]}
     */
    public Var logProbs(Tensor crops) {
        int b = crops.shape().dim(0);
        Var x = Var.of(crops.reshape(b, 1, bins));
        Var h = conv(x, w1, b1, 1).relu();
        h = conv(h, w2, b2, 4).relu();
        h = conv(h, wd, bd, 12).relu();
        h = conv(h, w3, null, 1).reshape(b, bins);
        Var t = toeplitz.gather(0, toeplitzIndex).reshape(bins, out);
        return h.matmul(t).logSoftmax(1);
    }

    /** {@code S} with {@code (p S)_o = p_{o + k}}. */
    private Tensor shiftMatrix(int k) {
        float[] s = new float[out * out];
        for (int o = 0; o < out; o++) {
            int src = o + k;
            if (src >= 0 && src < out) {
                s[src * out + o] = 1;
            }
        }
        return Tensor.of(Shape.of(out, out), s);
    }

    private Var perFrameLoss(float[][] frames, int[] rows, int k, SplittableRandom r) {
        double alpha = Math.pow(2, 1.0 / (12 * BINS_PER_SEMITONE));
        Var lp1 = logProbs(crops(frames, rows, 0, null, r));
        Var lp2 = logProbs(crops(frames, rows, k, null, r));
        float[][] tilt = new float[rows.length][bins];
        for (int i = 0; i < rows.length; i++) {
            double slope = (r.nextDouble() - 0.5) * 1.5;
            double gain = 0.6 + 0.8 * r.nextDouble();
            double ripple = r.nextDouble() * 0.3;
            double ph = r.nextDouble() * 6.28;
            for (int j = 0; j < bins; j++) {
                tilt[i][j] = (float) (gain * Math.exp(slope * (j / (double) bins - 0.5))
                        * (1 + ripple * Math.sin(j / 9.0 + ph)));
            }
        }
        Var lpa = logProbs(crops(frames, rows, 0, tilt, r));
        Var p1 = lp1.exp();
        Var p2 = lp2.exp();
        float[] powers = new float[out];
        for (int o = 0; o < out; o++) {
            powers[o] = (float) Math.pow(alpha, o);
        }
        Tensor pw = Tensor.of(Shape.of(out, 1), powers);
        Var y1 = p1.matmul(Var.of(pw));
        Var y2 = p2.matmul(Var.of(pw));
        Var diff = y2.div(y1).sub((float) Math.pow(alpha, -k)).abs();
        Var huber = Var.where(diff.greater(1f), diff.sub(0.5f), diff.square().mul(0.5f)).reshape(rows.length);
        Var p1s = p1.matmul(Var.of(shiftMatrix(k)));
        Var p2s = p2.matmul(Var.of(shiftMatrix(-k)));
        Var sce = p1s.mul(lp2).sum(1, false).neg().add(p2s.mul(lp1).sum(1, false).neg());
        // mass shifted out of the output range would make the shift cross-entropy vanish
        // (a collapse onto the edge bins); penalise it
        Var leak = p1s.sum(1, false).add(1e-4f).log().add(p2s.sum(1, false).add(1e-4f).log()).neg();
        Var inv = p1.mul(lpa).sum(1, false).neg().add(lpa.exp().mul(lp1).sum(1, false).neg());
        return huber.mul(10f).add(sce).add(inv).add(leak.mul(2f));
    }

    /**
     * Shift-consistency scores of frames: the symmetric shift cross-entropy (SCE) under a
     * shift of {@code k} bins. It is low only for frames whose prediction is both sharp and
     * shift-consistent; flat predictions of silence and noise score high.
     *
     * @param frames front-end frames
     * @param rows which frames
     * @param k the shift
     * @return per frame its inconsistency
     */
    public double[] shiftInconsistency(float[][] frames, int[] rows, int k) {
        return Var.noGrad(() -> {
            Var lp1 = logProbs(crops(frames, rows, 0, null, null));
            Var lp2 = logProbs(crops(frames, rows, k, null, null));
            Var p1 = lp1.exp();
            Var p2 = lp2.exp();
            Var p1s = p1.matmul(Var.of(shiftMatrix(k)));
            Var p2s = p2.matmul(Var.of(shiftMatrix(-k)));
            Var kl = p1s.mul(lp2).sum(1, false).add(p2s.mul(lp1).sum(1, false)).neg();
            float[] v = kl.value().toArray();
            double[] out = new double[v.length];
            for (int i = 0; i < v.length; i++) {
                out[i] = Math.max(0, v[i]);
            }
            return out;
        });
    }

    /**
     * Trains without labels: EM rounds of frame weighting by shift consistency and
     * weighted equivariance, SCE and invariance losses.
     *
     * @param frames front-end frames of unlabelled audio
     * @param energy their log energies
     * @param steps total steps
     * @param rounds EM rounds ({@code 1} trains unweighted)
     * @param batch batch size
     * @param lr learning rate
     * @param seed the seed
     * @return the final frame weights
     */
    public double[] train(float[][] frames, float[] energy, int steps, int rounds, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        double[] weights = new double[frames.length];
        Arrays.fill(weights, 1);
        int perRound = Math.max(1, steps / rounds);
        for (int s = 0; s < steps; s++) {
            if (s > 0 && rounds > 1 && s % perRound == 0) {
                weights = frameWeights(frames, r);
            }
            int[] rows = new int[batch];
            float[] w = new float[batch];
            double wsum = 0;
            for (int i = 0; i < batch; i++) {
                rows[i] = r.nextInt(frames.length);
                w[i] = (float) weights[rows[i]];
                wsum += w[i];
            }
            for (int i = 0; i < batch; i++) {
                w[i] = (float) (w[i] / Math.max(1e-6, wsum));
            }
            int k = r.nextInt(2 * MAX_SHIFT + 1) - MAX_SHIFT;
            params.forEach(Var::zeroGrad);
            Var loss = perFrameLoss(frames, rows, k == 0 ? 1 : k, r).mul(Tensor.of(Shape.of(batch), w)).sum();
            loss.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
        double[] finalWeights = frameWeights(frames, r);
        fitVoicing(frames, energy, finalWeights, r);
        return finalWeights;
    }

    private double[] frameWeights(float[][] frames, SplittableRandom r) {
        double[] sce = new double[frames.length];
        for (int from = 0; from < frames.length; from += 256) {
            int to = Math.min(frames.length, from + 256);
            int[] rows = new int[to - from];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = from + i;
            }
            int k = 1 + r.nextInt(MAX_SHIFT);
            double[] v = shiftInconsistency(frames, rows, r.nextBoolean() ? k : -k);
            System.arraycopy(v, 0, sce, from, v.length);
        }
        double[] sorted = sce.clone();
        Arrays.sort(sorted);
        double median = Math.max(1e-6, sorted[sorted.length / 2]);
        double[] w = new double[frames.length];
        for (int i = 0; i < w.length; i++) {
            w[i] = Math.exp(-sce[i] / median);
        }
        return w;
    }

    private double[] features(float[] logp, float energy, double inconsistency) {
        double max = Double.NEGATIVE_INFINITY;
        double ent = 0;
        for (float v : logp) {
            max = Math.max(max, v);
            ent -= Math.exp(v) * v;
        }
        return new double[] {Math.exp(max), ent / Math.log(logp.length), energy / 5.0, -Math.log1p(inconsistency), 1};
    }

    private void fitVoicing(float[][] frames, float[] energy, double[] weights, SplittableRandom r) {
        double[] sorted = weights.clone();
        Arrays.sort(sorted);
        double hi = sorted[(int) (0.6 * (sorted.length - 1))];
        double lo = sorted[(int) (0.3 * (sorted.length - 1))];
        double[][] feats = frameFeatures(frames, energy, r);
        double[] wv = new double[5];
        for (int epoch = 0; epoch < 200; epoch++) {
            double[] g = new double[5];
            int n = 0;
            for (int i = 0; i < frames.length; i++) {
                double label;
                if (weights[i] >= hi) {
                    label = 1;
                } else if (weights[i] <= lo) {
                    label = 0;
                } else {
                    continue;
                }
                double z = 0;
                for (int j = 0; j < 5; j++) {
                    z += wv[j] * feats[i][j];
                }
                double p = 1 / (1 + Math.exp(-z));
                for (int j = 0; j < 5; j++) {
                    g[j] += (p - label) * feats[i][j];
                }
                n++;
            }
            for (int j = 0; j < 5; j++) {
                wv[j] -= 2.0 * g[j] / Math.max(1, n);
            }
        }
        voicing = wv;
    }

    private double[][] frameFeatures(float[][] frames, float[] energy, SplittableRandom r) {
        double[][] result = new double[frames.length][];
        for (int from = 0; from < frames.length; from += 256) {
            int to = Math.min(frames.length, from + 256);
            int[] rows = new int[to - from];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = from + i;
            }
            float[] lp = Var.noGrad(() -> logProbs(crops(frames, rows, 0, null, null))).value().toArray();
            double[] inc = shiftInconsistency(frames, rows, 3);
            for (int i = 0; i < rows.length; i++) {
                result[from + i] = features(Arrays.copyOfRange(lp, i * out, (i + 1) * out), energy[from + i], inc[i]);
            }
        }
        return result;
    }

    private double argmaxBin(float[] lp, int i) {
        int best = 0;
        for (int o = 1; o < out; o++) {
            if (lp[i * out + o] > lp[i * out + best]) {
                best = o;
            }
        }
        double num = 0;
        double den = 0;
        for (int o = Math.max(0, best - 2); o <= Math.min(out - 1, best + 2); o++) {
            double p = Math.exp(lp[i * out + o]);
            num += o * p;
            den += p;
        }
        return num / den;
    }

    /**
     * Fixes the absolute pitch from synthetic harmonic tones of known pitch (the median
     * offset between their predicted bin and their true bin); no data labels are used.
     *
     * @param rate the sample rate
     * @param hop the hop
     * @param references reference pitches in Hz
     */
    public void calibrate(int rate, int hop, double... references) {
        double[] offs = new double[references.length];
        for (int q = 0; q < references.length; q++) {
            double f = references[q];
            float[] x = new float[rate / 2];
            for (int n = 0; n < x.length; n++) {
                double s = 0;
                for (int k = 1; k * f < rate / 2.0 - 100 && k <= 10; k++) {
                    s += Math.sin(2 * Math.PI * k * f * n / rate) / k;
                }
                x[n] = (float) (0.3 * s);
            }
            float[][] fr = frontEnd(x, rate, hop, fmin, bins, null);
            int mid = fr.length / 2;
            float[] lp = Var.noGrad(() -> logProbs(crops(fr, new int[] {mid}, 0, null, null))).value().toArray();
            double truth = 12 * BINS_PER_SEMITONE * Math.log(f / fmin) / Math.log(2);
            offs[q] = truth - argmaxBin(lp, 0);
        }
        Arrays.sort(offs);
        offset = offs[offs.length / 2];
    }

    /**
     * Tracks a signal.
     *
     * @param x the signal
     * @param rate the sample rate
     * @param hop the hop
     * @param learnedVoicing whether to use the pseudo-labelled voicing classifier (else the
     *     output peak above 0.1)
     * @return the track
     */
    public PitchTrackers.Track track(float[] x, int rate, int hop, boolean learnedVoicing) {
        float[] energy = new float[x.length / hop + 1];
        float[][] fr = frontEnd(x, rate, hop, fmin, bins, energy);
        float[] f0 = new float[fr.length];
        boolean[] voiced = new boolean[fr.length];
        double[][] feats = frameFeatures(fr, energy, new SplittableRandom(0));
        for (int from = 0; from < fr.length; from += 256) {
            int to = Math.min(fr.length, from + 256);
            int[] rows = new int[to - from];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = from + i;
            }
            float[] lp = Var.noGrad(() -> logProbs(crops(fr, rows, 0, null, null))).value().toArray();
            for (int i = 0; i < rows.length; i++) {
                double bin = argmaxBin(lp, i) + offset;
                f0[from + i] = (float) (fmin * Math.pow(2, bin / (12.0 * BINS_PER_SEMITONE)));
                double[] ft = feats[from + i];
                if (learnedVoicing) {
                    double z = 0;
                    for (int j = 0; j < 5; j++) {
                        z += voicing[j] * ft[j];
                    }
                    voiced[from + i] = z > 0;
                } else {
                    voiced[from + i] = ft[0] > 0.1;
                }
            }
        }
        return new PitchTrackers.Track(f0, voiced);
    }
}
