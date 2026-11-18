package io.github.zzuegg.brainiac.experiments.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.DynamicTimeWarping;
import io.github.zzuegg.brainiac.experimental.nn.ConformerBlock;
import io.github.zzuegg.brainiac.experimental.nn.CtcLoss;
import io.github.zzuegg.brainiac.experimental.nn.MambaLayer;
import io.github.zzuegg.brainiac.experimental.nn.MaskedSpeechPretraining;
import io.github.zzuegg.brainiac.experimental.nn.Mfcc;
import io.github.zzuegg.brainiac.experimental.nn.TransducerLoss;
import io.github.zzuegg.brainiac.experimental.timeseries.LearnableFrontEnd;
import io.github.zzuegg.brainiac.experimental.timeseries.Pcen;
import io.github.zzuegg.brainiac.experimental.timeseries.RidgeClassifier;
import io.github.zzuegg.brainiac.experimental.timeseries.ScatteringTransform;
import io.github.zzuegg.brainiac.experimental.timeseries.SincFilterbank;
import io.github.zzuegg.brainiac.experiments.generative.TalkingDigits;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Spoken-digit recognition on the formant synthesiser of {@link TalkingDigits}, extended
 * with speakers (a formant scale and a speaking rate per speaker) and additive noise at a
 * chosen SNR. It is the shared harness of the wave 7 audio items:
 *
 * <ul>
 *   <li>{@link #isolatedWords} (7.6.1): one digit per utterance and 1-20 examples per
 *       class; MFCC + DTW templates ({@link Mfcc}, {@link DynamicTimeWarping}) against the
 *       {@code TalkingDigits} frame-stacked MLP + CTC on log band energies and the same MLP
 *       on MFCCs;</li>
 *   <li>{@link #strings} (7.6.2): digit strings with CTC, comparing encoders behind one
 *       interface ({@link Encoder}): the frame-stacked MLP, a {@link ConformerBlock}, a
 *       Mamba stack (Samba-ASR style), a transformer block and a BiGRU, on log-mel
 *       features with or without SpecAugment;</li>
 *   <li>{@link #transducer} (7.6.3): strings from a Markov grammar, the RNN-Transducer
 *       ({@link TransducerLoss}) against CTC on the same encoder;</li>
 *   <li>{@link #pretraining} (7.6.4): {@link MaskedSpeechPretraining} on unlabelled
 *       utterances, then CTC fine-tuning with few labelled strings, against training from
 *       scratch.</li>
 * </ul>
 * Scores are word (or whole-string) accuracy and the digit error rate (edit distance over
 * reference length).
 */
public final class SpokenDigitRecognition {

    /** Samples per second. */
    public static final int RATE = TalkingDigits.RATE;
    /** Samples per frame. */
    public static final int HOP = TalkingDigits.FRAME;
    /** Log-mel bands. */
    public static final int MELS = 16;
    /** MFCCs kept (deltas double this). */
    public static final int CEPSTRA = 12;

    /**
     * A speaker.
     *
     * @param formantScale multiplies every formant frequency (vocal tract length)
     * @param rate speaking rate (durations are divided by it)
     */
    public record Speaker(double formantScale, double rate) {
    }

    private SpokenDigitRecognition() {
    }

    /**
     * Draws a speaker.
     *
     * @param r the random source
     * @return formant scale in [0.88, 1.12] and rate in [0.8, 1.25]
     */
    public static Speaker speaker(SplittableRandom r) {
        return new Speaker(0.88 + 0.24 * r.nextDouble(), 0.8 + 0.45 * r.nextDouble());
    }

    private static double[][] formants(int digit, double scale) {
        double f1 = 300 + 110 * digit;
        double f2 = 2500 - 170 * ((digit * 7) % 10);
        return new double[][] {{f1 * scale, f2 * scale}, {f1 * 1.35 * scale, (f2 * 0.75 + 200 * (digit % 3)) * scale}};
    }

    /**
     * Speaks a digit string as the {@code TalkingDigits} synthesiser does, with a speaker's
     * formant scale and rate, plus white noise at a given SNR.
     *
     * @param digits the digits
     * @param sp the speaker
     * @param snrDb signal-to-noise ratio in dB ({@code Double.POSITIVE_INFINITY} for the
     *     synthesiser's own faint noise only)
     * @param random varies durations, loudness, phases and noise
     * @return the waveform
     */
    public static float[] speak(int[] digits, Speaker sp, double snrDb, SplittableRandom random) {
        List<Float> wave = new ArrayList<>();
        pause(wave, HOP * (1 + random.nextInt(2)));
        for (int d : digits) {
            double amp = 0.4 + 0.4 * random.nextDouble();
            for (double[] seg : formants(d, sp.formantScale())) {
                int n = (int) ((HOP * (2 + random.nextInt(2)) + random.nextInt(HOP)) / sp.rate());
                double p1 = random.nextDouble() * 6.28;
                double p2 = random.nextDouble() * 6.28;
                for (int s = 0; s < n; s++) {
                    double env = Math.min(1, Math.min(s, n - s) / 60.0);
                    double v = Math.sin(2 * Math.PI * seg[0] * s / RATE + p1)
                            + 0.6 * Math.sin(2 * Math.PI * seg[1] * s / RATE + p2);
                    wave.add((float) (amp * env * v * 0.6));
                }
            }
            pause(wave, HOP * (1 + random.nextInt(2)));
        }
        float[] out = new float[wave.size()];
        double power = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = wave.get(i);
            power += out[i] * out[i] / out.length;
        }
        double noise = Double.isInfinite(snrDb) ? 0.02 : Math.sqrt(power / Math.pow(10, snrDb / 10));
        for (int i = 0; i < out.length; i++) {
            out[i] += (float) (noise * random.nextGaussian());
        }
        return out;
    }

    private static void pause(List<Float> wave, int n) {
        for (int i = 0; i < n; i++) {
            wave.add(0f);
        }
    }

    /**
     * Log-mel features (core {@link Signal#logMel}), padded or cut to a frame count.
     *
     * @param wave the waveform
     * @param frames the frames to produce
     * @return {@code [frames][MELS]}, scaled by 1/5
     */
    public static float[][] logMel(float[] wave, int frames) {
        float[] samples = Arrays.copyOf(wave, frames * HOP);
        float[] v = Signal.logMel(Var.of(Tensor.of(Shape.of(samples.length), samples)), RATE, HOP, HOP, MELS)
                .value().toArray();
        float[][] out = new float[frames][MELS];
        for (int t = 0; t < frames; t++) {
            for (int m = 0; m < MELS; m++) {
                out[t][m] = Math.max(-3, v[t * MELS + m] / 5);
            }
        }
        return out;
    }

    /**
     * MFCCs with deltas ({@link Mfcc}), of the utterance's own length (DTW) or padded.
     *
     * @param wave the waveform
     * @param frames the frames to produce, or -1 for the natural length
     * @return {@code [frames][2 * CEPSTRA]}
     */
    public static float[][] mfcc(float[] wave, int frames) {
        float[] samples = frames < 0 ? wave : Arrays.copyOf(wave, frames * HOP);
        float[][] m = Mfcc.compute(samples, RATE, HOP, HOP, 24, CEPSTRA, true);
        float[][] out = frames < 0 ? m : Arrays.copyOf(m, frames);
        for (int t = 0; t < out.length; t++) {
            if (out[t] == null) {
                out[t] = new float[2 * CEPSTRA];
            }
            out[t] = out[t].clone();
            for (int k = 0; k < out[t].length; k++) {
                out[t][k] /= k < CEPSTRA ? 10 : 2;
            }
        }
        return out;
    }

    /** Speech front ends. */
    public enum Features {
        /** {@code TalkingDigits}' log band energies. */
        BANDS,
        /** Log-mel energies. */
        LOG_MEL,
        /** MFCCs with deltas. */
        MFCC
    }

    /**
     * Features of a waveform.
     *
     * @param f the front end
     * @param wave the waveform
     * @param frames the frames to produce
     * @return {@code [frames][width(f)]}
     */
    public static float[][] features(Features f, float[] wave, int frames) {
        return switch (f) {
            case BANDS -> TalkingDigits.features(wave, frames);
            case LOG_MEL -> logMel(wave, frames);
            case MFCC -> mfcc(wave, frames);
        };
    }

    /**
     * @param f a front end
     * @return its features per frame
     */
    public static int width(Features f) {
        return switch (f) {
            case BANDS -> TalkingDigits.BANDS;
            case LOG_MEL -> MELS;
            case MFCC -> 2 * CEPSTRA;
        };
    }

    /** Acoustic encoders, all mapping {@code [batch, frames, features]} to {@code [batch, frames, width]}. */
    public enum Encoder {
        /** {@code TalkingDigits}' recogniser: frames stacked with two neighbours, two ReLU layers of 96. */
        MLP,
        /** An input projection and one Conformer block. */
        CONFORMER,
        /**
         * An input projection and a bidirectional Mamba block (one {@link MambaLayer} forward
         * in time, one backward, summed with a residual; Samba-ASR uses Mamba stacks).
         */
        MAMBA,
        /** An input projection with sinusoidal positions and one transformer block. */
        TRANSFORMER,
        /** A bidirectional GRU. */
        BIGRU
    }

    /** An encoder, a CTC head and the training loop. */
    public static final class Recognizer {
        private final Encoder kind;
        private final int features;
        private final int frames;
        private final int width;
        private final Var inProj;
        private final List<Layer> blocks = new ArrayList<>();
        private final Var head;
        private final Var headBias;
        private final Optimizer optimizer;

        /**
         * Creates an untrained recogniser.
         *
         * @param kind the encoder
         * @param features input features per frame
         * @param frames frames per utterance
         * @param seed the seed
         */
        public Recognizer(Encoder kind, int features, int frames, long seed) {
            this.kind = kind;
            this.features = features;
            this.frames = frames;
            SplittableRandom r = new SplittableRandom(seed);
            this.width = switch (kind) {
                case MLP -> 96;
                case BIGRU -> 48;
                default -> 40;
            };
            int in = kind == Encoder.MLP ? 5 * features : features;
            int projOut = kind == Encoder.MLP ? 96 : width;
            inProj = Var.param(Tensor.normal(Shape.of(in, projOut), 0, (float) Math.sqrt(2.0 / in), r));
            Shape seq = Shape.of(frames, width);
            switch (kind) {
                case MLP -> blocks.add(Dense.spec(96, Activation.RELU, Init.HE).create(Shape.of(frames, 96), r));
                case CONFORMER -> blocks.add(new ConformerBlock(width, 4, 5, frames, r));
                case MAMBA -> {
                    blocks.add(new MambaLayer(width, width, 4, r));
                    blocks.add(new MambaLayer(width, width, 4, r));
                }
                case TRANSFORMER -> blocks.add(TransformerBlock.of(4).ffnSize(2 * width).create(seq, r));
                case BIGRU -> blocks.add(Recurrent.gru(width / 2).returnSequences(true)
                        .bidirectional(Recurrent.Merge.CONCAT).create(seq, r));
            }
            head = Var.param(Tensor.normal(Shape.of(width == 96 || kind == Encoder.MLP ? 96 : width, 11), 0,
                    0.1f, r));
            headBias = Var.param(Tensor.zeros(Shape.of(11)));
            optimizer = Optimizer.adam(3e-3f);
        }

        private Tensor input(List<float[][]> feats) {
            int ctx = kind == Encoder.MLP ? 2 : 0;
            int w = features * (2 * ctx + 1);
            float[] x = new float[feats.size() * frames * w];
            for (int i = 0; i < feats.size(); i++) {
                float[][] f = feats.get(i);
                for (int t = 0; t < frames; t++) {
                    for (int c = -ctx; c <= ctx; c++) {
                        int src = Math.max(0, Math.min(frames - 1, t + c));
                        System.arraycopy(f[src], 0, x, ((i * frames + t) * (2 * ctx + 1) + c + ctx) * features,
                                features);
                    }
                }
            }
            return Tensor.of(Shape.of(feats.size(), frames, w), x);
        }

        /**
         * The encoder's output.
         *
         * @param feats per utterance {@code [frames][features]}
         * @param training whether training (SpecAugment is applied by the caller)
         * @return {@code [batch, frames, width]}
         */
        public Var encode(List<float[][]> feats, boolean training) {
            return encode(Var.of(input(feats)), training);
        }

        /**
         * The encoder's output for prepared input.
         *
         * @param x {@code [batch, frames, inputs]} (frames already stacked for the MLP)
         * @param training whether training
         * @return {@code [batch, frames, width]}
         */
        public Var encode(Var x, boolean training) {
            Var h = x.matmul(inProj);
            if (kind == Encoder.MLP) {
                h = h.relu();
            }
            if (kind == Encoder.TRANSFORMER) {
                h = h.add(Functional.sinusoidal(frames, width, 10000f, h.value().device()));
            }
            if (kind == Encoder.MAMBA) {
                // the recurrent scan: the same function as the layer's quadratic form, far cheaper here
                MambaLayer forward = (MambaLayer) blocks.get(0);
                MambaLayer backward = (MambaLayer) blocks.get(1);
                return h.add(forward.recurrent(h)).add(backward.recurrent(h.flip(1)).flip(1));
            }
            for (Layer l : blocks) {
                h = l.forward(h, training);
                if (kind == Encoder.MLP) {
                    h = h.relu();
                }
            }
            return h;
        }

        /**
         * Per-frame CTC logits.
         *
         * @param encoded the encoder's output
         * @return {@code [batch, frames, 11]}, class 0 the blank
         */
        public Var logits(Var encoded) {
            return encoded.matmul(head).add(headBias);
        }

        /**
         * One CTC training step.
         *
         * @param feats per utterance {@code [frames][features]}
         * @param digits their digit strings
         * @param specAugment whether to mask random bands and spans first
         * @param r the random source of the masks
         * @return the loss
         */
        public float trainCtc(List<float[][]> feats, List<int[]> digits, boolean specAugment, SplittableRandom r) {
            Var x = Var.of(input(feats));
            if (specAugment) {
                x = Signal.specAugment(x, 1, Math.max(1, features / 5), 1, 4, r);
            }
            Var loss = CtcLoss.loss(logits(encode(x, true)), shift(digits));
            float v = loss.value().item();
            loss.backward();
            optimizer.step(parameters());
            return v;
        }

        /**
         * Greedy CTC decoding.
         *
         * @param feats per utterance {@code [frames][features]}
         * @return the digits heard
         */
        public int[][] recognize(List<float[][]> feats) {
            int[][] d = Var.noGrad(() -> CtcLoss.greedyDecode(logits(encode(feats, false)).value()));
            for (int[] s : d) {
                for (int j = 0; j < s.length; j++) {
                    s[j] -= 1;
                }
            }
            return d;
        }

        /** @return the encoder's and head's trainable values */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(encoderParameters());
            p.add(head);
            p.add(headBias);
            return p;
        }

        /** @return the encoder's trainable values */
        public List<Var> encoderParameters() {
            List<Var> p = new ArrayList<>(List.of(inProj));
            blocks.forEach(b -> p.addAll(b.parameters()));
            return p;
        }

        /** @return the encoder's output width */
        public int width() {
            return kind == Encoder.MLP ? 96 : width;
        }

        /** @return the number of trainable scalars */
        public long parameterCount() {
            long n = 0;
            for (Var v : parameters()) {
                n += v.shape().size();
            }
            return n;
        }
    }

    static int[][] shift(List<int[]> digits) {
        int[][] labels = new int[digits.size()][];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = digits.get(i).clone();
            for (int j = 0; j < labels[i].length; j++) {
                labels[i][j] += 1;   // class 0 is the blank
            }
        }
        return labels;
    }

    /**
     * Levenshtein distance between two digit strings.
     *
     * @param a one string
     * @param b another
     * @return insertions, deletions and substitutions
     */
    public static int editDistance(int[] a, int[] b) {
        int[] prev = new int[b.length + 1];
        int[] cur = new int[b.length + 1];
        for (int j = 0; j <= b.length; j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= a.length; i++) {
            cur[0] = i;
            for (int j = 1; j <= b.length; j++) {
                cur[j] = Math.min(Math.min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            }
            int[] t = prev;
            prev = cur;
            cur = t;
        }
        return prev[b.length];
    }

    /**
     * The digit error rate.
     *
     * @param heard decoded strings
     * @param truth reference strings
     * @return total edit distance over total reference length
     */
    public static double digitErrorRate(int[][] heard, List<int[]> truth) {
        double e = 0;
        double n = 0;
        for (int i = 0; i < heard.length; i++) {
            e += editDistance(heard[i], truth.get(i));
            n += truth.get(i).length;
        }
        return e / n;
    }

    /** A corpus of utterances. */
    record Corpus(List<float[]> waves, List<int[]> digits) {
    }

    static Corpus corpus(int n, int minDigits, int maxDigits, double snrDb, boolean markov, SplittableRandom r) {
        List<float[]> waves = new ArrayList<>();
        List<int[]> digits = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] d = new int[minDigits + r.nextInt(maxDigits - minDigits + 1)];
            for (int j = 0; j < d.length; j++) {
                d[j] = markov && j > 0 && r.nextDouble() < 0.85 ? (3 * d[j - 1] + 1) % 10 : r.nextInt(10);
            }
            digits.add(d);
            waves.add(speak(d, speaker(r), snrDb, r));
        }
        return new Corpus(waves, digits);
    }

    static List<float[][]> featuresOf(Features f, List<float[]> waves, int frames) {
        List<float[][]> out = new ArrayList<>();
        for (float[] w : waves) {
            out.add(features(f, w, frames));
        }
        return out;
    }

    // ------------------------------------------------------------------ 7.6.1

    /**
     * Isolated words with few examples per class (7.6.1).
     *
     * @param shots examples per digit
     * @param snrDb the noise level of training and test utterances
     * @param steps CTC training steps for the MLP baselines
     * @param seed the seed
     * @return word accuracy of {DTW on MFCC, MLP + CTC on band energies, MLP + CTC on MFCC}
     */
    public static double[] isolatedWords(int shots, double snrDb, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> train = new ArrayList<>();
        List<int[]> trainDigits = new ArrayList<>();
        for (int d = 0; d < 10; d++) {
            for (int s = 0; s < shots; s++) {
                train.add(speak(new int[] {d}, speaker(r), snrDb, r));
                trainDigits.add(new int[] {d});
            }
        }
        Corpus test = corpus(100, 1, 1, snrDb, false, new SplittableRandom(seed + 50));
        DynamicTimeWarping.TemplateClassifier dtw = new DynamicTimeWarping.TemplateClassifier(4);
        for (int i = 0; i < train.size(); i++) {
            dtw.add(trainDigits.get(i)[0], trimmed(mfcc(train.get(i), -1)));
        }
        int dtwRight = 0;
        for (int i = 0; i < 100; i++) {
            dtwRight += dtw.classify(trimmed(mfcc(test.waves().get(i), -1))) == test.digits().get(i)[0] ? 1 : 0;
        }
        double[] out = new double[3];
        out[0] = dtwRight / 100.0;
        int frames = 16;
        Features[] fronts = {Features.BANDS, Features.MFCC};
        for (int k = 0; k < 2; k++) {
            Recognizer rec = new Recognizer(Encoder.MLP, width(fronts[k]), frames, seed);
            List<float[][]> feats = featuresOf(fronts[k], train, frames);
            SplittableRandom br = new SplittableRandom(seed + 3);
            for (int s = 0; s < steps; s++) {
                List<float[][]> bf = new ArrayList<>();
                List<int[]> bd = new ArrayList<>();
                for (int i = 0; i < 16; i++) {
                    int j = br.nextInt(feats.size());
                    bf.add(feats.get(j));
                    bd.add(trainDigits.get(j));
                }
                rec.trainCtc(bf, bd, false, br);
            }
            int[][] heard = rec.recognize(featuresOf(fronts[k], test.waves(), frames));
            int right = 0;
            for (int i = 0; i < 100; i++) {
                right += heard[i].length == 1 && heard[i][0] == test.digits().get(i)[0] ? 1 : 0;
            }
            out[k + 1] = right / 100.0;
        }
        return out;
    }

    /** Drops leading and trailing low-energy frames, so templates align on speech. */
    private static float[][] trimmed(float[][] m) {
        int a = 0;
        int b = m.length;
        while (a < b - 1 && m[a][0] < m[0][0] + 0.3 && a < m.length / 3) {
            a++;
        }
        while (b > a + 1 && m[b - 1][0] < m[m.length - 1][0] + 0.3 && b > 2 * m.length / 3) {
            b--;
        }
        return Arrays.copyOfRange(m, a, b);
    }

    // ------------------------------------------------------------------ 7.6.2

    /**
     * A string-recognition score.
     *
     * @param encoder the encoder
     * @param features the front end
     * @param specAugment whether SpecAugment was used in training
     * @param digitErrorRate the test digit error rate
     * @param parameters trainable scalars
     * @param millis training time
     */
    public record StringScore(Encoder encoder, Features features, boolean specAugment, double digitErrorRate, long parameters,
            long millis) {
        @Override
        public String toString() {
            return String.format("%s/%s%s DER %.3f (%dk params, %d ms)", encoder, features,
                    specAugment ? "+SpecAug" : "",
                    digitErrorRate, parameters / 1000, millis);
        }
    }

    /**
     * Digit strings of 1-3 digits with CTC (7.6.2), batches of 16 drawn from 480 training utterances.
     *
     * @param encoder the encoder
     * @param f the front end
     * @param specAugment whether to use SpecAugment
     * @param snrDb the noise level
     * @param steps training steps of 16 utterances
     * @param seed the seed
     * @return the score on 60 test strings
     */
    public static StringScore strings(Encoder encoder, Features f, boolean specAugment, double snrDb, int steps,
            long seed) {
        return strings(encoder, f, specAugment, snrDb, steps, 480, seed);
    }

    /**
     * Digit strings of 1-3 digits with CTC, from a training pool of a given size.
     *
     * @param encoder the encoder
     * @param f the front end
     * @param specAugment whether to use SpecAugment
     * @param snrDb the noise level
     * @param steps training steps of 16 utterances
     * @param poolSize training utterances
     * @param seed the seed
     * @return the score on 60 test strings
     */
    public static StringScore strings(Encoder encoder, Features f, boolean specAugment, double snrDb, int steps,
            int poolSize, long seed) {
        int frames = 32;
        Recognizer rec = new Recognizer(encoder, width(f), frames, seed);
        SplittableRandom r = new SplittableRandom(seed);
        Corpus pool = corpus(poolSize, 1, 3, snrDb, false, r);
        List<float[][]> poolFeats = featuresOf(f, pool.waves(), frames);
        long t0 = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            List<float[][]> bf = new ArrayList<>();
            List<int[]> bd = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                int j = r.nextInt(poolFeats.size());
                bf.add(poolFeats.get(j));
                bd.add(pool.digits().get(j));
            }
            rec.trainCtc(bf, bd, specAugment, r);
        }
        long ms = (System.nanoTime() - t0) / 1_000_000;
        Corpus test = corpus(60, 1, 3, snrDb, false, new SplittableRandom(seed + 99));
        int[][] heard = rec.recognize(featuresOf(f, test.waves(), frames));
        return new StringScore(encoder, f, specAugment, digitErrorRate(heard, test.digits()), rec.parameterCount(), ms);
    }

    // ------------------------------------------------------------------ 7.6.3

    /**
     * Transducer against CTC scores.
     *
     * @param ctcDer CTC's digit error rate (greedy)
     * @param rnntGreedyDer the transducer's digit error rate, greedy decoding
     * @param rnntBeamDer the transducer's digit error rate, beam search of width 4
     * @param ctcEmissionFrame CTC's mean frame of each correctly placed label emission, over utterance frames
     * @param rnntEmissionFrame the transducer's, likewise
     */
    public record TransducerScore(double ctcDer, double rnntGreedyDer, double rnntBeamDer, double ctcEmissionFrame,
            double rnntEmissionFrame) {
    }

    /**
     * Strings of three digits from a Markov grammar (each digit follows {@code 3d + 1 mod 10}
     * with probability 0.85) under heavy noise, so label context helps (7.6.3). The same
     * frame-stacked MLP encoder is trained with CTC or with the transducer loss, on batches of
     * 16 drawn from 480 training strings.
     *
     * @param snrDb the noise level
     * @param steps training steps of 16 utterances
     * @param seed the seed
     * @return the scores on 50 test strings
     */
    public static TransducerScore transducer(double snrDb, int steps, long seed) {
        int frames = 32;
        Features f = Features.BANDS;
        Recognizer ctc = new Recognizer(Encoder.MLP, width(f), frames, seed);
        Recognizer enc = new Recognizer(Encoder.MLP, width(f), frames, seed);
        TransducerLoss.Joint joint = new TransducerLoss.Joint(enc.width(), 48, 11, new SplittableRandom(seed + 1));
        List<Var> rnntParams = new ArrayList<>(enc.encoderParameters());
        rnntParams.addAll(joint.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        Corpus pool = corpus(480, 3, 3, snrDb, true, r);
        List<float[][]> poolFeats = featuresOf(f, pool.waves(), frames);
        for (int s = 0; s < steps; s++) {
            List<float[][]> feats = new ArrayList<>();
            List<int[]> digits = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                int j = r.nextInt(poolFeats.size());
                feats.add(poolFeats.get(j));
                digits.add(pool.digits().get(j));
            }
            ctc.trainCtc(feats, digits, false, r);
            Var loss = TransducerLoss.loss(joint.logProbs(enc.encode(feats, true), shift(digits)), shift(digits));
            loss.backward();
            opt.step(rnntParams);
        }
        Corpus test = corpus(50, 3, 3, snrDb, true, new SplittableRandom(seed + 99));
        List<float[][]> feats = featuresOf(f, test.waves(), frames);
        int[][] ctcHeard = ctc.recognize(feats);
        Tensor encoded = Var.noGrad(() -> enc.encode(feats, false)).value();
        Tensor ctcLogits = Var.noGrad(() -> ctc.logits(ctc.encode(feats, false))).value();
        int[][] greedy = new int[50][];
        int[][] beam = new int[50][];
        double rnntFrame = 0;
        int rnntCount = 0;
        for (int i = 0; i < 50; i++) {
            Tensor e = encoded.slice(0, i, i + 1).reshape(frames, enc.width());
            List<Integer> at = new ArrayList<>();
            greedy[i] = unshift(joint.greedy(e, at));
            beam[i] = unshift(joint.beam(e, 4));
            for (int k = 0; k < Math.min(at.size(), test.digits().get(i).length); k++) {
                rnntFrame += at.get(k) / (double) frames;
                rnntCount++;
            }
        }
        double ctcFrame = 0;
        int ctcCount = 0;
        float[] lg = ctcLogits.toArray();
        for (int i = 0; i < 50; i++) {
            int prev = 0;
            int k = 0;
            for (int t = 0; t < frames && k < test.digits().get(i).length; t++) {
                int best = 0;
                for (int c = 1; c < 11; c++) {
                    if (lg[(i * frames + t) * 11 + c] > lg[(i * frames + t) * 11 + best]) {
                        best = c;
                    }
                }
                if (best != 0 && best != prev) {
                    ctcFrame += t / (double) frames;
                    ctcCount++;
                    k++;
                }
                prev = best;
            }
        }
        return new TransducerScore(digitErrorRate(ctcHeard, test.digits()), digitErrorRate(greedy, test.digits()),
                digitErrorRate(beam, test.digits()), ctcFrame / Math.max(1, ctcCount),
                rnntFrame / Math.max(1, rnntCount));
    }

    private static int[] unshift(int[] labels) {
        int[] out = labels.clone();
        for (int j = 0; j < out.length; j++) {
            out[j] -= 1;
        }
        return out;
    }

    // ------------------------------------------------------------------ 7.6.4

    /**
     * Pretraining scores.
     *
     * @param labelled labelled strings in fine-tuning
     * @param scratchDer the digit error rate of the encoder trained from scratch
     * @param pretrainedDer after masked pretraining and fine-tuning
     */
    public record PretrainScore(int labelled, double scratchDer, double pretrainedDer) {
    }

    /**
     * Masked speech pretraining on unlabelled utterances, then CTC fine-tuning on few
     * labelled ones, against the same encoder trained from scratch on those (7.6.4).
     *
     * @param encoder the encoder (not the frame-stacked MLP)
     * @param mode {@code hubert} or {@code wav2vec2}
     * @param unlabelledSteps pretraining steps of 16 unlabelled utterances
     * @param labelled labelled training strings
     * @param fineTuneSteps CTC steps of 16 labelled strings (sampled with replacement)
     * @param seed the seed
     * @return the scores on 60 test strings
     */
    public static PretrainScore pretraining(Encoder encoder, MaskedSpeechPretraining.Mode mode, int unlabelledSteps,
            int labelled, int fineTuneSteps, long seed) {
        int frames = 32;
        Features f = Features.LOG_MEL;
        Corpus labels = corpus(labelled, 1, 3, 20, false, new SplittableRandom(seed + 7));
        List<float[][]> labelledFeats = featuresOf(f, labels.waves(), frames);
        Corpus test = corpus(60, 1, 3, 20, false, new SplittableRandom(seed + 99));
        List<float[][]> testFeats = featuresOf(f, test.waves(), frames);
        double[] der = new double[2];
        for (int pre = 0; pre < 2; pre++) {
            Recognizer rec = new Recognizer(encoder, width(f), frames, seed);
            SplittableRandom r = new SplittableRandom(seed + 3);
            if (pre == 1) {
                MaskedSpeechPretraining ssl = new MaskedSpeechPretraining(mode, x -> rec.encode(x, true),
                        rec.encoderParameters(), width(f), rec.width(), seed);
                Corpus unl = corpus(200, 1, 3, 20, false, new SplittableRandom(seed + 11));
                List<float[][]> unlFeats = featuresOf(f, unl.waves(), frames);
                ssl.fitTargets(unlFeats);
                for (int s = 0; s < unlabelledSteps; s++) {
                    List<float[][]> batch = new ArrayList<>();
                    for (int i = 0; i < 16; i++) {
                        batch.add(unlFeats.get(r.nextInt(unlFeats.size())));
                    }
                    ssl.trainStep(batch, r);
                }
            }
            for (int s = 0; s < fineTuneSteps; s++) {
                List<float[][]> bf = new ArrayList<>();
                List<int[]> bd = new ArrayList<>();
                for (int i = 0; i < 16; i++) {
                    int j = r.nextInt(labelledFeats.size());
                    bf.add(labelledFeats.get(j));
                    bd.add(labels.digits().get(j));
                }
                rec.trainCtc(bf, bd, false, r);
            }
            der[pre] = digitErrorRate(rec.recognize(testFeats), test.digits());
        }
        return new PretrainScore(labelled, der[0], der[1]);
    }

    // ------------------------------------------------------------------ wave 8: 8.2.16 front ends

    /** Front ends for the noise sweep (8.2.16). */
    public enum FrontEnd {
        /** Log-mel energies (core {@link Signal#logMel}). */
        LOG_MEL,
        /** MFCCs with deltas. */
        MFCC,
        /** Core {@code Signal} mel energies through fixed, untrained {@link Pcen} instead of a log. */
        MEL_PCEN,
        /** Mel energies through {@link Pcen} trained with the recogniser. */
        MEL_PCEN_LEARNED,
        /** A {@link SincFilterbank} on the waveform, log-compressed, trained with the recogniser. */
        SINCNET,
        /** {@link LearnableFrontEnd} (Gabor filters, Gaussian pooling, PCEN), trained with the recogniser. */
        LEAF
    }

    /** {mean, standard deviation} of values. */
    private static double[] moments(float[] v) {
        double m = 0;
        for (float x : v) {
            m += x / (double) v.length;
        }
        double s = 0;
        for (float x : v) {
            s += (x - m) * (x - m) / v.length;
        }
        return new double[] {m, Math.sqrt(s) + 1e-6};
    }

    /** Stacks each frame with two neighbours either side, as the MLP encoder's {@code input} does. */
    private static Var stackFrames(Var x) {
        int t = x.shape().dim(1);
        Var[] parts = new Var[5];
        for (int c = -2; c <= 2; c++) {
            if (c == 0) {
                parts[c + 2] = x;
            } else if (c < 0) {
                Var[] cat = new Var[-c + 1];
                for (int k = 0; k < -c; k++) {
                    cat[k] = x.slice(1, 0, 1);
                }
                cat[-c] = x.slice(1, 0, t + c);
                parts[c + 2] = Var.concat(1, cat);
            } else {
                Var[] cat = new Var[c + 1];
                cat[0] = x.slice(1, c, t);
                for (int k = 1; k <= c; k++) {
                    cat[k] = x.slice(1, t - 1, t);
                }
                parts[c + 2] = Var.concat(1, cat);
            }
        }
        return Var.concat(2, parts);
    }

    private static Var melEnergies(Var waves, int frames) {
        Tensor fb = Signal.melFilterbank(RATE, HOP, MELS, 0, RATE / 2.0);
        return Signal.power(Signal.stft(waves, HOP, HOP)).matmul(Var.of(fb)).slice(1, 0, frames);
    }

    private static Tensor waves(List<float[]> list, int samples) {
        float[] x = new float[list.size() * samples];
        for (int i = 0; i < list.size(); i++) {
            float[] w = list.get(i);
            System.arraycopy(w, 0, x, i * samples, Math.min(samples, w.length));
        }
        return Tensor.of(Shape.of(list.size(), samples), x);
    }

    /**
     * The digit error rate of the frame-wise MLP + CTC recogniser on 1-3 digit strings (the
     * {@link #strings} task: 480 training utterances, 60 test strings, training and test at the
     * same SNR) with a given front end; the learned front ends train jointly with the recogniser.
     *
     * @param front the front end
     * @param snrDb the noise level
     * @param steps training steps of 16 utterances
     * @param seed the seed
     * @return the test digit error rate
     */
    public static double frontEndDer(FrontEnd front, double snrDb, int steps, long seed) {
        if (front == FrontEnd.LOG_MEL || front == FrontEnd.MFCC) {
            return strings(Encoder.MLP, front == FrontEnd.LOG_MEL ? Features.LOG_MEL : Features.MFCC, false, snrDb,
                    steps, seed).digitErrorRate();
        }
        int frames = 32;
        int samples = frames * HOP;
        Recognizer rec = new Recognizer(Encoder.MLP, MELS, frames, seed);
        Layer fe = switch (front) {
            case MEL_PCEN, MEL_PCEN_LEARNED -> new Pcen(MELS, frames, 0.15);
            case SINCNET -> new SincFilterbank(MELS, 65, 2, HOP, RATE);
            default -> new LearnableFrontEnd(MELS, 65, 2, HOP, frames, RATE);
        };
        java.util.function.UnaryOperator<Var> raw = switch (front) {
            case MEL_PCEN, MEL_PCEN_LEARNED -> w -> fe.forward(melEnergies(w, frames).mul(1e-2f), true);
            default -> w -> fe.forward(w, true);
        };

        List<Var> params = new ArrayList<>(rec.parameters());
        if (front != FrontEnd.MEL_PCEN) {
            params.addAll(fe.parameters());
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        Corpus pool = corpus(480, 1, 3, snrDb, false, r);
        // a frozen affine map onto the scaled log-mel features' global mean and spread, measured on
        // 64 training utterances at initialisation: without it the CTC recogniser can collapse to
        // blanks at this budget for inputs on another scale (PCEN is about 0-1.5)
        List<float[]> probe = pool.waves().subList(0, 64);
        double[] own = moments(Var.noGrad(() -> raw.apply(Var.of(waves(probe, samples)))).value().toArray());
        List<float[][]> ref = featuresOf(Features.LOG_MEL, probe, frames);
        float[] flat = new float[64 * frames * MELS];
        for (int i = 0; i < 64; i++) {
            for (int t = 0; t < frames; t++) {
                System.arraycopy(ref.get(i)[t], 0, flat, (i * frames + t) * MELS, MELS);
            }
        }
        double[] target = moments(flat);
        float gain = (float) (target[1] / own[1]);
        float shift = (float) (target[0] - own[0] * gain);
        java.util.function.UnaryOperator<Var> features = w -> raw.apply(w).mul(gain).add(shift);
        for (int s = 0; s < steps; s++) {
            List<float[]> bw = new ArrayList<>();
            List<int[]> bd = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                int j = r.nextInt(pool.waves().size());
                bw.add(pool.waves().get(j));
                bd.add(pool.digits().get(j));
            }
            Var x = stackFrames(features.apply(Var.of(waves(bw, samples))));
            CtcLoss.loss(rec.logits(rec.encode(x, true)), shift(bd)).backward();
            opt.step(params);
        }
        Corpus test = corpus(60, 1, 3, snrDb, false, new SplittableRandom(seed + 99));
        int[][] heard = Var.noGrad(() -> CtcLoss.greedyDecode(rec.logits(rec.encode(stackFrames(features.apply(
                Var.of(waves(test.waves(), samples)))), false)).value()));
        for (int[] h : heard) {
            for (int j = 0; j < h.length; j++) {
                h[j] -= 1;
            }
        }
        return digitErrorRate(heard, test.digits());
    }

    /**
     * The noise sweep of 8.2.16: digit error rate per front end and SNR.
     *
     * @param fronts the front ends
     * @param snrs the noise levels in dB
     * @param steps training steps per run
     * @param seed the seed
     * @return front end to DER per SNR
     */
    public static java.util.Map<FrontEnd, double[]> noiseSweep(FrontEnd[] fronts, double[] snrs, int steps,
            long seed) {
        java.util.Map<FrontEnd, double[]> out = new java.util.LinkedHashMap<>();
        for (FrontEnd f : fronts) {
            double[] der = new double[snrs.length];
            for (int i = 0; i < snrs.length; i++) {
                der[i] = frontEndDer(f, snrs[i], steps, seed);
            }
            out.put(f, der);
        }
        return out;
    }

    // ------------------------------------------------------------------ wave 8: 8.2.17 scattering

    /**
     * Isolated words from few examples per class (8.2.17): a second-order wavelet scattering
     * transform of the waveform (Morlet, {@code J = 8}, four wavelets per octave, low-pass
     * subsampled to 8 positions, log-compressed) with a ridge classifier, against MFCCs and
     * log-mel frames with the same ridge classifier and wave 7's DTW template matcher on MFCCs.
     *
     * @param shots training examples per digit
     * @param snrDb the noise level of training and test utterances
     * @param seed the seed
     * @return method to word accuracy on 100 test words
     */
    public static java.util.Map<String, Double> scatteringWords(int shots, double snrDb, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> train = new ArrayList<>();
        int[] labels = new int[10 * shots];
        for (int d = 0; d < 10; d++) {
            for (int s = 0; s < shots; s++) {
                labels[train.size()] = d;
                train.add(speak(new int[] {d}, speaker(r), snrDb, r));
            }
        }
        Corpus test = corpus(100, 1, 1, snrDb, false, new SplittableRandom(seed + 50));
        int[] truth = new int[100];
        for (int i = 0; i < 100; i++) {
            truth[i] = test.digits().get(i)[0];
        }
        java.util.Map<String, Double> out = new java.util.LinkedHashMap<>();
        int size = 2048;
        ScatteringTransform st = ScatteringTransform.oneDimensional(size, 8, 4);
        float[][] trainW = padded(train, size);
        float[][] testW = padded(test.waves(), size);
        double[][] a = ScatteringTransform.logCompress(st.transform(trainW, 0, false), 1e3);
        double[][] b = ScatteringTransform.logCompress(st.transform(testW, 0, false), 1e3);
        out.put("scattering + ridge", ridgeAccuracy(a, labels, b, truth));
        int frames = size / HOP;
        out.put("MFCC + ridge", ridgeAccuracy(flat(train, Features.MFCC, frames), labels,
                flat(test.waves(), Features.MFCC, frames), truth));
        out.put("log-mel + ridge", ridgeAccuracy(flat(train, Features.LOG_MEL, frames), labels,
                flat(test.waves(), Features.LOG_MEL, frames), truth));
        DynamicTimeWarping.TemplateClassifier dtw = new DynamicTimeWarping.TemplateClassifier(4);
        for (int i = 0; i < train.size(); i++) {
            dtw.add(labels[i], trimmed(mfcc(train.get(i), -1)));
        }
        int right = 0;
        for (int i = 0; i < 100; i++) {
            right += dtw.classify(trimmed(mfcc(test.waves().get(i), -1))) == truth[i] ? 1 : 0;
        }
        out.put("DTW on MFCC", right / 100.0);
        return out;
    }

    private static float[][] padded(List<float[]> waves, int size) {
        float[][] out = new float[waves.size()][];
        for (int i = 0; i < out.length; i++) {
            out[i] = Arrays.copyOf(waves.get(i), size);
        }
        return out;
    }

    private static double[][] flat(List<float[]> waves, Features f, int frames) {
        double[][] out = new double[waves.size()][];
        for (int i = 0; i < out.length; i++) {
            float[][] m = features(f, waves.get(i), frames);
            out[i] = new double[frames * m[0].length];
            for (int t = 0; t < frames; t++) {
                for (int k = 0; k < m[t].length; k++) {
                    out[i][t * m[0].length + k] = m[t][k];
                }
            }
        }
        return out;
    }

    private static double ridgeAccuracy(double[][] train, int[] labels, double[][] test, int[] truth) {
        RidgeClassifier c = RidgeClassifier.fit(train, labels, 10, 10.0);
        return RidgeClassifier.accuracy(c.predict(test), truth);
    }
}
