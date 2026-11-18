package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Flow-matching text-to-speech with two ways of aligning text to mel frames.
 *
 * <p>{@link Variant#MATCHA} (Mehta et al., ICASSP 2024,
 * <a href="https://arxiv.org/abs/2309.03199">arXiv:2309.03199</a>): a convolutional text encoder
 * predicts a mean mel per symbol; during training {@link MonotonicAlignmentSearch} finds the
 * monotonic alignment of the frames to these means, the prior loss pulls each mean to its frames,
 * and a duration predictor (on the detached encoder states) learns the aligned durations. The
 * decoder generates the mel from the means expanded by duration. {@link Variant#TRUE_DURATIONS}
 * is the same model given the synthesiser's durations instead of MAS (an upper bound).
 *
 * <p>{@link Variant#F5} (Chen et al., 2024, <a href="https://arxiv.org/abs/2410.06885">arXiv:2410.06885</a>):
 * no durations and no alignment. The symbol sequence is padded with filler tokens to the mel
 * length, embedded and convolved, and the decoder infills masked mel frames given the padded
 * text and the unmasked frames (a random prefix kept as an audio prompt during training).
 * Alignment has to emerge in the decoder's attention.
 *
 * <p>Both variants share the decoder: a transformer with rotary positions over frames, fed the
 * noisy mel, the conditioning and a time embedding, trained with the rectified-flow (conditional
 * flow matching) loss and sampled with Euler steps from Gaussian noise. Mels are normalised
 * internally as {@code (logMel + 5) / 3}.
 */
public final class FlowMatchingTts {

    /** How text is aligned to frames. */
    public enum Variant {
        /** Matcha-TTS: monotonic alignment search and a duration predictor. */
        MATCHA,
        /** Matcha-TTS trained on the true durations (no alignment search). */
        TRUE_DURATIONS,
        /** F5-TTS: filler-padded text and masked-mel infilling, no durations. */
        F5
    }

    /**
     * A training utterance.
     *
     * @param text the symbols
     * @param mel the log-mels {@code [frames][mels]}
     * @param durations the true frames per symbol, or {@code null} when unknown
     */
    public record Utterance(int[] text, float[][] mel, int[] durations) {
    }

    private static final int TIME = 32;
    private static final float MEAN = -5f;
    private static final float SCALE = 3f;

    private final Variant variant;
    private final int vocab;
    private final int filler;
    private final int mels;
    private final int width;
    private final SplittableRandom random;
    private final Var embedding;
    private final List<Var[]> textConvs = new ArrayList<>();
    private final Var meanOut;
    private final Var meanBias;
    private final Var dur1;
    private final Var dur1Bias;
    private final Var dur2;
    private final Var dur2Bias;
    private final Var inProj;
    private final Var inBias;
    private final Var time1;
    private final Var time1Bias;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final Var outProj;
    private final Var outBias;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private int[] lastAlignment;

    /**
     * Creates a model.
     *
     * @param variant the alignment scheme
     * @param vocab the text symbols (a filler symbol is added for F5)
     * @param mels the mel bands
     * @param width the model width
     * @param depth the decoder transformer blocks
     * @param seed the seed
     */
    public FlowMatchingTts(Variant variant, int vocab, int mels, int width, int depth, long seed) {
        this.variant = variant;
        this.vocab = vocab;
        this.filler = vocab;
        this.mels = mels;
        this.width = width;
        this.random = new SplittableRandom(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        embedding = Var.param(Tensor.normal(Shape.of(vocab + 1, width), 0, 0.3f, r));
        for (int l = 0; l < 2; l++) {
            textConvs.add(new Var[] {AudioNets.kernel(r, width, width, 3), AudioNets.zeros(width)});
        }
        meanOut = AudioNets.dense(r, width, mels);
        meanBias = AudioNets.zeros(mels);
        dur1 = AudioNets.dense(r, width, width);
        dur1Bias = AudioNets.zeros(width);
        dur2 = AudioNets.dense(r, width, 1);
        dur2Bias = Var.param(Tensor.full(Shape.of(1), 1f));
        int cond = variant == Variant.F5 ? mels + 1 + width : mels;
        inProj = AudioNets.dense(r, mels + cond, width);
        inBias = AudioNets.zeros(width);
        time1 = AudioNets.dense(r, TIME, width);
        time1Bias = AudioNets.zeros(width);
        TransformerBlock.Spec spec = TransformerBlock.of(4).ffnSize(2 * width).rotary(1000f);
        for (int l = 0; l < depth; l++) {
            blocks.add((TransformerBlock) spec.create(Shape.of(16, width), r));
        }
        outProj = Var.param(Tensor.normal(Shape.of(width, mels), 0, (float) (0.1 / Math.sqrt(width)), r));
        outBias = AudioNets.zeros(mels);
    }

    /** @return the variant */
    public Variant variant() {
        return variant;
    }

    /** Text encoder: {@code [S]} symbols to states {@code [S, width]}. */
    private Var encode(int[] text) {
        float[] idx = new float[text.length];
        for (int i = 0; i < text.length; i++) {
            idx[i] = text[i];
        }
        Var h = embedding.gather(0, Tensor.of(Shape.of(text.length), idx));
        for (Var[] c : textConvs) {
            Var x = h.transpose().reshape(1, width, text.length);
            Var y = AudioNets.conv(x, c[0], c[1], 1, 1).relu().reshape(width, text.length).transpose();
            h = h.add(y);
        }
        return h;
    }

    private Var means(Var states) {
        return states.matmul(meanOut).add(meanBias);
    }

    private Var logDurations(Var states) {
        return states.matmul(dur1).add(dur1Bias).relu().matmul(dur2).add(dur2Bias);
    }

    private static Tensor normalised(float[][] mel, int frames) {
        int m = mel[0].length;
        float[] f = new float[frames * m];
        for (int j = 0; j < frames; j++) {
            float[] row = mel[Math.min(j, mel.length - 1)];
            for (int k = 0; k < m; k++) {
                f[j * m + k] = (row[k] - MEAN) / SCALE;
            }
        }
        return Tensor.of(Shape.of(frames, m), f);
    }

    private static float[] pathIndex(int[] path) {
        float[] idx = new float[path.length];
        for (int j = 0; j < path.length; j++) {
            idx[j] = path[j];
        }
        return idx;
    }

    /** The velocity network: {@code [b, F, mels]} noisy mels and {@code [b, F, cond]} to {@code [b, F, mels]}. */
    private Var velocity(Var xt, Var cond, float[] t) {
        int b = xt.shape().dim(0);
        int f = xt.shape().dim(1);
        Var h = Var.concat(2, xt, cond).matmul(inProj).add(inBias);
        Var te = Var.of(AudioNets.sinusoid(t, 1000f, TIME)).matmul(time1).add(time1Bias);
        h = h.add(te.mul(te.sigmoid()).reshape(b, 1, width));
        for (TransformerBlock blk : blocks) {
            h = blk.forward(List.of(h), null, false);
        }
        return h.reshape(b * f, width).matmul(outProj).add(outBias).reshape(b, f, mels);
    }

    /** F5 conditioning per frame: masked mel, the mask, and the convolved filler-padded text. */
    private Var f5Condition(int[] text, Tensor mel, boolean[] known, int frames) {
        int[] padded = new int[frames];
        for (int j = 0; j < frames; j++) {
            padded[j] = j < text.length ? text[j] : filler;
        }
        Var states = encode(padded);
        float[] m = new float[frames * (mels + 1)];
        float[] mv = mel == null ? null : mel.toArray();
        for (int j = 0; j < frames; j++) {
            if (known[j]) {
                System.arraycopy(mv, j * mels, m, j * (mels + 1), mels);
                m[j * (mels + 1) + mels] = 1;
            }
        }
        return Var.concat(1, Var.of(Tensor.of(Shape.of(frames, mels + 1), m)), states);
    }

    /**
     * One training step.
     *
     * @param batch utterances, all with the same number of frames
     * @return the flow-matching loss
     */
    public float step(List<Utterance> batch) {
        int b = batch.size();
        int frames = batch.get(0).mel().length;
        List<Var> conds = new ArrayList<>();
        List<Var> targets = new ArrayList<>();
        List<Var> lossWeights = new ArrayList<>();
        Var aux = null;
        for (Utterance u : batch) {
            Tensor y = normalised(u.mel(), frames);
            targets.add(Var.of(y.reshape(1, frames, mels)));
            if (variant == Variant.F5) {
                boolean[] known = new boolean[frames];
                if (random.nextDouble() < 0.7) {
                    int prefix = (int) (frames * 0.3 * random.nextDouble());
                    for (int j = 0; j < prefix; j++) {
                        known[j] = true;
                    }
                }
                float[] w = new float[frames];
                for (int j = 0; j < frames; j++) {
                    w[j] = known[j] ? 0 : 1;
                }
                conds.add(f5Condition(u.text(), y, known, frames).reshape(1, frames, mels + 1 + width));
                lossWeights.add(Var.of(Tensor.of(Shape.of(1, frames, 1), w)));
                continue;
            }
            Var states = encode(u.text());
            Var mu = means(states);                                                   // [S, mels]
            int[] path;
            if (variant == Variant.TRUE_DURATIONS) {
                path = MonotonicAlignmentSearch.path(u.durations(), frames);
            } else {
                float[] mv = mu.value().toArray();
                float[] yv = y.toArray();
                double[][] ll = new double[u.text().length][frames];
                for (int i = 0; i < ll.length; i++) {
                    for (int j = 0; j < frames; j++) {
                        double s = 0;
                        for (int k = 0; k < mels; k++) {
                            double e = yv[j * mels + k] - mv[i * mels + k];
                            s += e * e;
                        }
                        ll[i][j] = -0.5 * s;
                    }
                }
                path = MonotonicAlignmentSearch.align(ll);
                lastAlignment = path;
            }
            Var aligned = mu.gather(0, Tensor.of(Shape.of(frames), pathIndex(path)));    // [F, mels]
            int[] d = MonotonicAlignmentSearch.durations(path, u.text().length);
            float[] logD = new float[d.length];
            for (int i = 0; i < d.length; i++) {
                logD[i] = (float) Math.log(1 + d[i]);
            }
            Var prior = aligned.sub(Var.of(y)).square().mean();
            Var durLoss = logDurations(Var.of(states.value())).reshape(d.length)
                    .sub(Var.of(Tensor.of(Shape.of(d.length), logD))).square().mean();
            Var term = prior.add(durLoss);
            aux = aux == null ? term : aux.add(term);
            conds.add(aligned.reshape(1, frames, mels));
            lossWeights.add(Var.of(Tensor.full(Shape.of(1, frames, 1), 1f)));
        }
        Var y1 = Var.concat(0, targets.toArray(new Var[0]));
        Var cond = Var.concat(0, conds.toArray(new Var[0]));
        Var weight = Var.concat(0, lossWeights.toArray(new Var[0]));
        Tensor x0 = Tensor.normal(Shape.of(b, frames, mels), 0, 1, random);
        float[] t = new float[b];
        for (int i = 0; i < b; i++) {
            t[i] = (float) random.nextDouble();
        }
        Tensor tt = Tensor.of(Shape.of(b, 1, 1), t);
        Var xt = Var.of(x0).mul(Var.of(Tensor.full(Shape.of(b, 1, 1), 1f).sub(tt))).add(y1.mul(Var.of(tt)));
        Var target = y1.sub(Var.of(x0));
        Var flow = velocity(xt, cond, t).sub(target).square().mul(weight).sum()
                .div(weight.sum().mul((float) mels));
        Var loss = aux == null ? flow : flow.add(aux.mul(1f / b));
        float v = flow.value().item();
        loss.backward();
        optimizer.step(parameters());
        return v;
    }

    /** @return the alignment MAS found for the last utterance of the last training batch, or null */
    public int[] lastAlignment() {
        return lastAlignment;
    }

    /**
     * The alignment MAS finds for an utterance under the current encoder (Matcha only).
     *
     * @param u the utterance
     * @return frames per symbol
     */
    public int[] alignedDurations(Utterance u) {
        int frames = u.mel().length;
        float[] mv = Var.noGrad(() -> means(encode(u.text()))).value().toArray();
        float[] yv = normalised(u.mel(), frames).toArray();
        double[][] ll = new double[u.text().length][frames];
        for (int i = 0; i < ll.length; i++) {
            for (int j = 0; j < frames; j++) {
                double s = 0;
                for (int k = 0; k < mels; k++) {
                    double e = yv[j * mels + k] - mv[i * mels + k];
                    s += e * e;
                }
                ll[i][j] = -0.5 * s;
            }
        }
        return MonotonicAlignmentSearch.durations(MonotonicAlignmentSearch.align(ll), u.text().length);
    }

    /**
     * Predicted frames per symbol (Matcha variants).
     *
     * @param text the symbols
     * @return the durations, at least 1 each
     */
    public int[] predictDurations(int[] text) {
        float[] ld = Var.noGrad(() -> logDurations(encode(text))).value().toArray();
        int[] d = new int[text.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = Math.max(1, (int) Math.round(Math.exp(ld[i]) - 1));
        }
        return d;
    }

    /**
     * Synthesises log-mels.
     *
     * @param text the symbols
     * @param frames the frames to produce (F5 fills exactly this many; Matcha produces its
     *     predicted length, padded with the last symbol or cut to this many)
     * @param odeSteps Euler steps
     * @param seed the noise seed
     * @return {@code [frames][mels]}
     */
    public float[][] synthesize(int[] text, int frames, int odeSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var cond;
        if (variant == Variant.F5) {
            cond = Var.noGrad(() -> f5Condition(text, null, new boolean[frames], frames))
                    .reshape(1, frames, mels + 1 + width);
        } else {
            int[] path = MonotonicAlignmentSearch.path(predictDurations(text), frames);
            cond = Var.noGrad(() -> means(encode(text)).gather(0, Tensor.of(Shape.of(frames), pathIndex(path))))
                    .reshape(1, frames, mels);
        }
        Var c = Var.of(cond.value());
        Tensor x = Tensor.normal(Shape.of(1, frames, mels), 0, 1, r);
        for (int k = 0; k < odeSteps; k++) {
            float[] t = {(float) k / odeSteps};
            Tensor xk = x;
            Tensor v = Var.noGrad(() -> velocity(Var.of(xk), c, t)).value();
            x = x.add(v.mul(Tensor.scalar(1f / odeSteps)));
        }
        float[] f = x.toArray();
        float[][] out = new float[frames][mels];
        for (int j = 0; j < frames; j++) {
            for (int k = 0; k < mels; k++) {
                out[j][k] = f[j * mels + k] * SCALE + MEAN;
            }
        }
        return out;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding));
        for (Var[] c : textConvs) {
            p.addAll(List.of(c));
        }
        if (variant != Variant.F5) {
            p.addAll(List.of(meanOut, meanBias, dur1, dur1Bias, dur2, dur2Bias));
        }
        p.addAll(List.of(inProj, inBias, time1, time1Bias));
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.add(outProj);
        p.add(outBias);
        return p;
    }

    /** @return the number of text symbols (without the filler) */
    public int vocab() {
        return vocab;
    }
}
