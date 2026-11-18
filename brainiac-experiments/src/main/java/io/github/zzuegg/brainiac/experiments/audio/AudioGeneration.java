package io.github.zzuegg.brainiac.experiments.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.audio.DiffusionVocoder;
import io.github.zzuegg.brainiac.experimental.audio.FiniteScalarQuantizer;
import io.github.zzuegg.brainiac.experimental.audio.FlowMatchingTts;
import io.github.zzuegg.brainiac.experimental.audio.GanVocoder;
import io.github.zzuegg.brainiac.experimental.audio.GriffinLim;
import io.github.zzuegg.brainiac.experimental.audio.LatentAudioDiffusion;
import io.github.zzuegg.brainiac.experimental.audio.MelFeatures;
import io.github.zzuegg.brainiac.experimental.audio.MonotonicAlignmentSearch;
import io.github.zzuegg.brainiac.experimental.nn.ResidualVectorQuantizer;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.VectorQuantizer;
import io.github.zzuegg.brainiac.experiments.generative.TalkingDigits;
import io.github.zzuegg.brainiac.experiments.generative.ToyAudioCodec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Wave 9, WP5 audio generation (9.5.9-9.5.13) on {@link TalkingDigits}' formant speech at 8 kHz
 * with 40-band log-mels ({@link MelFeatures#speech()}). The judge throughout is a
 * {@link TalkingDigits} CTC recogniser trained on real speech and frozen: generated or
 * resynthesised speech is scored by its digit error rate.
 * <ul>
 * <li>{@link #vocoders}: Griffin-Lim, the HiFi-GAN-lite {@link GanVocoder} with its ablations, and
 * the {@link DiffusionVocoder} (DiffWave and WaveGrad conditioning) at several step counts.
 * <li>{@link #tts}: flow-matching TTS with monotonic alignment search (Matcha), with the true
 * durations, and with F5's filler padding; scored on strings as long as in training and longer.
 * <li>{@link #latentGeneration}: class-conditional spoken digits by latent and by mel diffusion.
 * <li>{@link #codecs}: FSQ against VQ and RVQ on {@link ToyAudioCodec} frames at matched code
 * spaces.
 * </ul>
 */
public final class AudioGeneration {

    /** The front end used throughout. */
    public static final MelFeatures MEL = MelFeatures.speech();
    /** The frames every judged utterance is padded to. */
    public static final int JUDGE_FRAMES = 56;
    private static final int HOP = TalkingDigits.FRAME;
    private static final int SIL = 20;
    private static final Map<Long, TalkingDigits> JUDGES = new HashMap<>();

    private AudioGeneration() {
    }

    // ---------------------------------------------------------------- data and judge

    /**
     * An utterance with its phone-level alignment.
     *
     * @param digits the digits
     * @param wave the waveform
     * @param text the phones: silence, then two phones and a silence per digit
     * @param boundaries the end sample of every phone
     */
    public record Aligned(int[] digits, float[] wave, int[] text, int[] boundaries) {

        /**
         * Frames per phone when the waveform is cut or padded to {@code frames} frames (frame
         * {@code j} belongs to the phone that holds sample {@code j * hop}).
         *
         * @param frames the frames
         * @return the durations
         */
        public int[] durations(int frames) {
            int[] d = new int[text.length];
            int p = 0;
            for (int j = 0; j < frames; j++) {
                int s = j * HOP;
                while (p < text.length - 1 && s >= boundaries[p]) {
                    p++;
                }
                d[p]++;
            }
            return d;
        }
    }

    /**
     * Speaks digits exactly as {@link TalkingDigits#speak} does (the same draws in the same
     * order, with its formant formula copied) and also records where each phone ends, which the
     * synthesiser does not report.
     *
     * @param digits the digits
     * @param random the random source
     * @return the aligned utterance
     */
    public static Aligned speakAligned(int[] digits, SplittableRandom random) {
        List<Float> wave = new ArrayList<>();
        List<Integer> ends = new ArrayList<>();
        int[] text = new int[1 + 3 * digits.length];
        text[0] = SIL;
        silence(wave, HOP * (1 + random.nextInt(2)), random);
        ends.add(wave.size());
        int k = 1;
        for (int d : digits) {
            double amp = 0.4 + 0.4 * random.nextDouble();
            double f1 = 300 + 110 * d;
            double f2 = 2500 - 170 * ((d * 7) % 10);
            double[][] formants = {{f1, f2}, {f1 * 1.35, f2 * 0.75 + 200 * (d % 3)}};
            for (int seg = 0; seg < 2; seg++) {
                int n = HOP * (2 + random.nextInt(2)) + random.nextInt(HOP);
                double p1 = random.nextDouble() * 6.28;
                double p2 = random.nextDouble() * 6.28;
                for (int s = 0; s < n; s++) {
                    double env = Math.min(1, Math.min(s, n - s) / 60.0);
                    double v = Math.sin(2 * Math.PI * formants[seg][0] * s / TalkingDigits.RATE + p1)
                            + 0.6 * Math.sin(2 * Math.PI * formants[seg][1] * s / TalkingDigits.RATE + p2);
                    wave.add((float) (amp * env * v * 0.6 + 0.02 * random.nextGaussian()));
                }
                ends.add(wave.size());
                text[k++] = 2 * d + seg;
            }
            silence(wave, HOP * (1 + random.nextInt(2)), random);
            ends.add(wave.size());
            text[k++] = SIL;
        }
        float[] out = new float[wave.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = wave.get(i);
        }
        int[] b = ends.stream().mapToInt(Integer::intValue).toArray();
        b[b.length - 1] = Integer.MAX_VALUE;
        return new Aligned(digits, out, text, b);
    }

    private static void silence(List<Float> wave, int n, SplittableRandom random) {
        for (int i = 0; i < n; i++) {
            wave.add((float) (0.02 * random.nextGaussian()));
        }
    }

    private static List<Aligned> corpus(int n, int minDigits, int maxDigits, SplittableRandom r) {
        List<Aligned> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] d = new int[minDigits + r.nextInt(maxDigits - minDigits + 1)];
            for (int j = 0; j < d.length; j++) {
                d[j] = r.nextInt(10);
            }
            out.add(speakAligned(d, r));
        }
        return out;
    }

    /**
     * The frozen judge: a {@link TalkingDigits} recogniser over {@link #JUDGE_FRAMES} frames,
     * trained on real strings of 1-5 digits (cached per seed).
     *
     * @param seed the seed
     * @return the recogniser
     */
    public static synchronized TalkingDigits judge(long seed) {
        return JUDGES.computeIfAbsent(seed, s -> {
            TalkingDigits td = TalkingDigits.of(JUDGE_FRAMES, s);
            SplittableRandom r = new SplittableRandom(s + 1);
            for (int step = 0; step < 200; step++) {
                List<int[]> digits = new ArrayList<>();
                List<float[]> waves = TalkingDigits.utterances(16, 5, r, digits);
                td.trainStep(waves, digits);
            }
            return td;
        });
    }

    private static double der(TalkingDigits judge, List<float[]> waves, List<int[]> digits) {
        return SpokenDigitRecognition.digitErrorRate(judge.recognize(waves), digits);
    }

    private static float[] fit(float[] wave, int frames) {
        return Arrays.copyOf(wave, frames * HOP);
    }

    private static float[][] melOf(float[] wave, int frames) {
        return MEL.logMel(fit(wave, frames));          // frames + 1 rows
    }

    private static float[][] firstRows(float[][] m, int n) {
        return Arrays.copyOf(m, n);
    }

    private static float[][] withExtraRow(float[][] m) {
        float[][] out = Arrays.copyOf(m, m.length + 1);
        out[m.length] = m[m.length - 1];
        return out;
    }

    // ---------------------------------------------------------------- 9.5.9 / 9.5.10 vocoders

    /**
     * A vocoder's scores on resynthesised test speech.
     *
     * @param vocoder the vocoder
     * @param stftDistance the multi-resolution STFT distance to the real waveform
     * @param digitErrorRate the judge's digit error rate
     * @param millisPerSecond synthesis time per second of audio
     */
    public record VocoderRow(String vocoder, double stftDistance, double digitErrorRate, double millisPerSecond) {
    }

    private static VocoderRow score(String name, List<Aligned> test, int frames, UnaryOperator<float[][]> melToWave,
            TalkingDigits judge) {
        List<float[]> out = new ArrayList<>();
        List<int[]> digits = new ArrayList<>();
        double dist = 0;
        long start = System.nanoTime();
        List<float[]> made = new ArrayList<>();
        for (Aligned a : test) {
            made.add(melToWave.apply(melOf(a.wave(), frames))[0]);
        }
        double ms = (System.nanoTime() - start) / 1e6;
        for (int i = 0; i < test.size(); i++) {
            float[] real = fit(test.get(i).wave(), frames);
            dist += MelFeatures.stftDistance(real, made.get(i)) / test.size();
            out.add(made.get(i));
            digits.add(test.get(i).digits());
        }
        double seconds = test.size() * frames * HOP / (double) TalkingDigits.RATE;
        return new VocoderRow(name, dist, der(judge, out, digits), ms / seconds);
    }

    /**
     * 9.5.9 and 9.5.10: vocoders resynthesising held-out speech from its log-mels.
     *
     * @param ganSteps GAN vocoder steps (each variant)
     * @param diffusionSteps diffusion vocoder steps (each conditioning)
     * @param test test utterances (1-3 digits)
     * @param sampling the diffusion sampling step counts, such as 6, 12 and 50
     * @param seed the seed
     * @return one row per vocoder (and step count)
     */
    public static List<VocoderRow> vocoders(int ganSteps, int diffusionSteps, int test, int[] sampling, long seed) {
        int frames = 32;
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> train = new ArrayList<>();
        for (Aligned a : corpus(200, 1, 3, r)) {
            train.add(a.wave());
        }
        List<Aligned> held = corpus(test, 1, 3, new SplittableRandom(seed + 77));
        TalkingDigits judge = judge(seed);
        List<VocoderRow> rows = new ArrayList<>();
        rows.add(realRow(held, frames, judge));
        for (int it : new int[] {32, 100}) {
            GriffinLim gl = new GriffinLim(MEL.nFft(), MEL.hop(), it, 0.99f, seed);
            rows.add(score("Griffin-Lim " + it, held, frames,
                    m -> new float[][] {gl.vocode(MEL, m, frames * HOP)}, judge));
        }
        String[] names = {"HiFi-GAN-lite (MPD + MSD)", "HiFi-GAN-lite without MPD", "generator, mel loss only"};
        for (int v = 0; v < names.length; v++) {
            GanVocoder g = GanVocoder.of(MEL, 32, seed + 10 + v);   // discriminators 8-16 channels wide
            if (v == 1) {
                g.periods();
            } else if (v == 2) {
                g.adversarial(false);
            }
            for (int s = 0; s < ganSteps; s++) {
                g.step(g.crops(train, 4, 8));
            }
            rows.add(score(names[v], held, frames, m -> new float[][] {g.vocode(m, frames * HOP)}, judge));
        }
        for (DiffusionVocoder.Conditioning c : DiffusionVocoder.Conditioning.values()) {
            DiffusionVocoder dv = new DiffusionVocoder(MEL, 16, 6, 50, c, seed + 20);
            GanVocoder cropper = GanVocoder.of(MEL, 4, seed + 30);
            for (int s = 0; s < diffusionSteps; s++) {
                dv.step(cropper.crops(train, 4, 4));
            }
            String name = c == DiffusionVocoder.Conditioning.DISCRETE ? "DiffWave" : "WaveGrad-style";
            for (int steps : sampling) {
                rows.add(score(name + ", " + steps + " steps", held, frames,
                        m -> new float[][] {dv.vocode(m, frames * HOP, steps, seed)}, judge));
            }
        }
        return rows;
    }

    private static VocoderRow realRow(List<Aligned> held, int frames, TalkingDigits judge) {
        List<float[]> waves = new ArrayList<>();
        List<int[]> digits = new ArrayList<>();
        for (Aligned a : held) {
            waves.add(fit(a.wave(), frames));
            digits.add(a.digits());
        }
        return new VocoderRow("real waveform", 0, der(judge, waves, digits), 0);
    }

    // ---------------------------------------------------------------- 9.5.11 TTS

    /**
     * A TTS model's scores.
     *
     * @param model the model
     * @param derShort digit error rate on 1-3 digit strings (as in training)
     * @param derLong digit error rate on 4-5 digit strings (longer than any in training)
     * @param alignmentError mean phone-boundary error in frames of the training alignment (MAS)
     *     or of the predicted durations against the true ones; NaN when there are none
     * @param odeSteps Euler steps per utterance
     */
    public record TtsRow(String model, double derShort, double derLong, double alignmentError, int odeSteps) {
    }

    /**
     * 9.5.11: flow-matching TTS, vocoded by Griffin-Lim (32 iterations) and judged.
     *
     * @param steps training steps per model (batches of 8)
     * @param test test strings per length bucket
     * @param odeSteps Euler steps
     * @param seed the seed
     * @return one row per model, plus the vocoded ground-truth mels
     */
    public static List<TtsRow> tts(int steps, int test, int odeSteps, long seed) {
        int frames = 34;
        int longFrames = JUDGE_FRAMES;
        SplittableRandom r = new SplittableRandom(seed);
        List<Aligned> train = corpus(400, 1, 3, r);
        List<FlowMatchingTts.Utterance> data = new ArrayList<>();
        for (Aligned a : train) {
            data.add(new FlowMatchingTts.Utterance(a.text(), firstRows(melOf(a.wave(), frames), frames),
                    a.durations(frames)));
        }
        List<Aligned> shortTest = corpus(test, 1, 3, new SplittableRandom(seed + 5));
        List<Aligned> longTest = corpus(test, 4, 5, new SplittableRandom(seed + 6));
        TalkingDigits judge = judge(seed);
        GriffinLim gl = new GriffinLim(MEL.nFft(), MEL.hop(), 32, 0.99f, seed);
        List<TtsRow> rows = new ArrayList<>();
        UnaryOperator<float[][]> vocode = m -> new float[][] {gl.vocode(MEL, withExtraRow(m), m.length * HOP)};
        rows.add(new TtsRow("ground-truth mels + Griffin-Lim",
                judgeTts(shortTest, frames, (a, f) -> firstRows(melOf(a.wave(), f), f), vocode, judge),
                judgeTts(longTest, longFrames, (a, f) -> firstRows(melOf(a.wave(), f), f), vocode, judge), 0, 0));
        String[] names = {"Matcha-style (MAS + durations)", "Matcha-style with true durations", "F5-style filler padding"};
        FlowMatchingTts.Variant[] variants = {FlowMatchingTts.Variant.MATCHA, FlowMatchingTts.Variant.TRUE_DURATIONS,
            FlowMatchingTts.Variant.F5};
        for (int v = 0; v < variants.length; v++) {
            FlowMatchingTts model = new FlowMatchingTts(variants[v], SIL + 1, MEL.mels(), 64, 2, seed + 40 + v);
            for (int s = 0; s < steps; s++) {
                List<FlowMatchingTts.Utterance> b = new ArrayList<>();
                for (int i = 0; i < 8; i++) {
                    b.add(data.get(r.nextInt(data.size())));
                }
                model.step(b);
            }
            double align = Double.NaN;
            if (variants[v] != FlowMatchingTts.Variant.F5) {
                align = 0;
                for (Aligned a : shortTest) {
                    int[] truth = a.durations(frames);
                    int[] found = variants[v] == FlowMatchingTts.Variant.MATCHA
                            ? model.alignedDurations(new FlowMatchingTts.Utterance(a.text(),
                                    firstRows(melOf(a.wave(), frames), frames), null))
                            : model.predictDurations(a.text());
                    align += MonotonicAlignmentSearch.boundaryError(truth, found) / shortTest.size();
                }
            }
            long noise = seed + 3;
            rows.add(new TtsRow(names[v],
                    judgeTts(shortTest, frames, (a, f) -> model.synthesize(a.text(), f, odeSteps, noise), vocode,
                            judge),
                    judgeTts(longTest, longFrames, (a, f) -> model.synthesize(a.text(), f, odeSteps, noise), vocode,
                            judge),
                    align, odeSteps));
        }
        return rows;
    }

    private interface MelSource {
        float[][] mel(Aligned a, int frames);
    }

    private static double judgeTts(List<Aligned> test, int frames, MelSource source, UnaryOperator<float[][]> vocode,
            TalkingDigits judge) {
        List<float[]> waves = new ArrayList<>();
        List<int[]> digits = new ArrayList<>();
        for (Aligned a : test) {
            waves.add(vocode.apply(source.mel(a, frames))[0]);
            digits.add(a.digits());
        }
        return der(judge, waves, digits);
    }

    // ---------------------------------------------------------------- 9.5.12 latent diffusion

    /**
     * A class-conditional generator's scores.
     *
     * @param model the model
     * @param classAccuracy the share of samples the judge hears as exactly the requested digit
     * @param mmd the RBF feature MMD between generated and real mels (an FAD proxy)
     * @param evaluations network evaluations per sample
     * @param millis generation time for all samples
     */
    public record LatentRow(String model, double classAccuracy, double mmd, int evaluations, double millis) {
    }

    /**
     * 9.5.12: class-conditional spoken digits (16 frames) by diffusion in a mel-VAE latent
     * against the same denoiser on mels; vocoded by Griffin-Lim and judged.
     *
     * @param autoencoderSteps VAE steps
     * @param denoiserSteps denoiser steps (each model, batches of 32)
     * @param perClass samples per digit (at most 10)
     * @param sampling Euler steps
     * @param seed the seed
     * @return one row per model, plus real clips through the VAE
     */
    public static List<LatentRow> latentGeneration(int autoencoderSteps, int denoiserSteps, int perClass, int sampling,
            long seed) {
        int frames = 16;
        SplittableRandom r = new SplittableRandom(seed);
        List<float[][]> clips = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int i = 0; i < 600; i++) {
            int d = i % 10;
            clips.add(firstRows(melOf(speakAligned(new int[] {d}, r).wave(), frames), frames));
            labels.add(d);
        }
        TalkingDigits judge = judge(seed);
        GriffinLim gl = new GriffinLim(MEL.nFft(), MEL.hop(), 32, 0.99f, seed);
        int[] want = new int[10 * Math.min(10, perClass)];
        List<float[][]> reference = new ArrayList<>();
        for (int i = 0; i < want.length; i++) {
            want[i] = i % 10;
            reference.add(clips.get(clips.size() - 100 + i));   // held out of training, label i % 10
        }
        List<LatentRow> rows = new ArrayList<>();
        rows.add(new LatentRow("real clips", classAccuracy(reference, want, gl, judge),
                mmd(reference, clips.subList(0, 200)), 0, 0));
        for (LatentAudioDiffusion.Space space : LatentAudioDiffusion.Space.values()) {
            LatentAudioDiffusion model = new LatentAudioDiffusion(space, frames, MEL.mels(), 10, 8, 4, 128, seed + 50);
            if (space == LatentAudioDiffusion.Space.LATENT) {
                for (int s = 0; s < autoencoderSteps; s++) {
                    model.trainAutoencoder(sample(clips, 32, r, null, null));
                }
                List<float[][]> rec = model.reconstruct(reference);
                rows.add(new LatentRow("real clips through the VAE", classAccuracy(rec, want, gl, judge),
                        mmd(rec, clips.subList(0, 200)), 0, 0));
            }
            for (int s = 0; s < denoiserSteps; s++) {
                int[] lab = new int[32];
                List<float[][]> b = sample(clips, 32, r, labels, lab);
                model.trainDenoiser(b, lab);
            }
            long start = System.nanoTime();
            List<float[][]> gen = model.generate(want, sampling, 2f, seed + 9);
            double ms = (System.nanoTime() - start) / 1e6;
            String name = space == LatentAudioDiffusion.Space.LATENT ? "latent diffusion (" + model.dimension() + "-d)"
                    : "mel diffusion (" + model.dimension() + "-d)";
            rows.add(new LatentRow(name, classAccuracy(gen, want, gl, judge), mmd(gen, clips.subList(0, 200)),
                    LatentAudioDiffusion.evaluations(sampling, 2f), ms));
        }
        return rows;
    }

    private static List<float[][]> sample(List<float[][]> clips, int n, SplittableRandom r, List<Integer> labels,
            int[] out) {
        List<float[][]> b = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int k = r.nextInt(clips.size() - 100);
            b.add(clips.get(k));
            if (out != null) {
                out[i] = labels.get(k);
            }
        }
        return b;
    }

    private static double classAccuracy(List<float[][]> mels, int[] want, GriffinLim gl, TalkingDigits judge) {
        List<float[]> waves = new ArrayList<>();
        for (float[][] m : mels) {
            waves.add(gl.vocode(MEL, withExtraRow(m), m.length * HOP));
        }
        int[][] heard = judge.recognize(waves);
        int ok = 0;
        for (int i = 0; i < heard.length; i++) {
            ok += heard[i].length == 1 && heard[i][0] == want[i] ? 1 : 0;
        }
        return ok / (double) heard.length;
    }

    /** The squared RBF-kernel MMD of flattened mels, bandwidth from the median distance. */
    static double mmd(List<float[][]> a, List<float[][]> b) {
        List<float[]> x = flatten(a);
        List<float[]> y = flatten(b);
        List<Double> d = new ArrayList<>();
        for (int i = 0; i < Math.min(50, x.size()); i++) {
            for (int j = 0; j < Math.min(50, y.size()); j++) {
                d.add(dist(x.get(i), y.get(j)));
            }
        }
        d.sort(null);
        double h = Math.max(1e-6, d.get(d.size() / 2));
        return mean(x, x, h, true) + mean(y, y, h, true) - 2 * mean(x, y, h, false);
    }

    private static List<float[]> flatten(List<float[][]> m) {
        List<float[]> out = new ArrayList<>();
        for (float[][] c : m) {
            float[] f = new float[c.length * c[0].length];
            for (int j = 0; j < c.length; j++) {
                for (int k = 0; k < c[j].length; k++) {
                    f[j * c[0].length + k] = Math.max(-8, c[j][k]);
                }
            }
            out.add(f);
        }
        return out;
    }

    private static double dist(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double e = a[i] - b[i];
            s += e * e;
        }
        return s;
    }

    private static double mean(List<float[]> x, List<float[]> y, double h, boolean same) {
        double s = 0;
        int n = 0;
        for (int i = 0; i < x.size(); i++) {
            for (int j = 0; j < y.size(); j++) {
                if (same && i == j) {
                    continue;
                }
                s += Math.exp(-dist(x.get(i), y.get(j)) / h);
                n++;
            }
        }
        return s / Math.max(1, n);
    }

    // ---------------------------------------------------------------- 9.5.13 codecs

    /**
     * A quantizer's scores in the toy codec.
     *
     * @param quantizer the quantizer
     * @param codebook the size of its code space
     * @param snr the round-trip signal-to-noise ratio on held-out frames, dB
     * @param perplexity the perplexity of the codes on held-out frames
     * @param utilisation the share of the code space used on held-out frames
     */
    public record CodecRow(String quantizer, int codebook, double snr, double perplexity, double utilisation) {
    }

    /**
     * 9.5.13: FSQ against VQ (with dead-code restarts) and two-stage RVQ in a
     * {@link ToyAudioCodec}-style frame autoencoder (64 samples per frame), at code spaces of
     * about 2^10, 2^12 and 2^14.
     *
     * @param steps training steps per codec (batches of 64 frames)
     * @param seed the seed
     * @return one row per quantizer and size
     */
    public static List<CodecRow> codecs(int steps, long seed) {
        int frame = 64;
        ToyAudioCodec source = ToyAudioCodec.of(frame, 8, 1, 2, seed);
        SplittableRandom r = new SplittableRandom(seed);
        float[][] test = source.frames(20000, new SplittableRandom(seed + 1));
        int[][] fsqLevels = {{8, 5, 5, 5}, {7, 5, 5, 5, 5}, {8, 8, 8, 6, 5}};
        int[] sizes = {1024, 4096, 16384};
        int[] rvq = {32, 64, 128};
        List<CodecRow> rows = new ArrayList<>();
        for (int k = 0; k < sizes.length; k++) {
            for (int kind = 0; kind < 3; kind++) {
                int dim = kind == 0 ? fsqLevels[k].length : 8;
                SimpleMlp enc = SimpleMlp.of(seed + 3, v -> v.mul(v.sigmoid()), frame, 128, dim);
                SimpleMlp dec = SimpleMlp.of(seed + 4, v -> v.mul(v.sigmoid()), dim, 128, frame);
                FiniteScalarQuantizer fsq = kind == 0 ? new FiniteScalarQuantizer(fsqLevels[k]) : null;
                VectorQuantizer vq = kind == 1 ? VectorQuantizer.of(sizes[k], dim, 0.25f, seed + 5) : null;
                ResidualVectorQuantizer rv = kind == 2 ? ResidualVectorQuantizer.of(2, rvq[k], dim, 0.25f, seed + 6)
                        : null;
                Optimizer opt = Optimizer.adam(2e-3f);
                for (int s = 0; s < steps; s++) {
                    Tensor x = matrix(source.frames(64, r));
                    Var z = enc.forward(Var.of(x));
                    Var q;
                    Var aux = null;
                    if (fsq != null) {
                        q = fsq.quantize(z).output();
                    } else if (vq != null) {
                        VectorQuantizer.Result res = vq.quantize(z);
                        q = res.output();
                        aux = res.loss();
                    } else {
                        ResidualVectorQuantizer.Result res = rv.quantize(z, 2);
                        q = res.output();
                        aux = res.loss();
                    }
                    Var loss = dec.forward(q).sub(Var.of(x)).square().mean();
                    if (aux != null) {
                        loss = loss.add(aux);
                    }
                    loss.backward();
                    List<Var> p = new ArrayList<>(enc.parameters());
                    p.addAll(dec.parameters());
                    if (vq != null) {
                        p.addAll(vq.parameters());
                    }
                    if (rv != null) {
                        p.addAll(rv.parameters());
                    }
                    opt.step(p);
                    if (s % 50 == 49 && s < steps * 3 / 4) {
                        Tensor zs = Var.noGrad(() -> enc.forward(Var.of(x))).value();
                        if (vq != null) {
                            vq.restartDeadCodes(zs);
                        } else if (rv != null) {
                            rv.restartDeadCodes(zs);
                        }
                    }
                }
                Tensor xt = matrix(test);
                Var zt = Var.noGrad(() -> enc.forward(Var.of(xt)));
                int[] codes;
                Var qt;
                String name;
                if (fsq != null) {
                    FiniteScalarQuantizer.Result res = fsq.quantize(zt);
                    codes = res.codes();
                    qt = res.output();
                    name = "FSQ " + Arrays.toString(fsqLevels[k]);
                } else if (vq != null) {
                    VectorQuantizer.Result res = vq.quantize(zt);
                    codes = res.codes();
                    qt = res.output();
                    name = "VQ " + sizes[k];
                } else {
                    ResidualVectorQuantizer.Result res = rv.quantize(zt, 2);
                    codes = new int[test.length];
                    for (int i = 0; i < codes.length; i++) {
                        codes[i] = res.codes()[0][i] * rvq[k] + res.codes()[1][i];
                    }
                    qt = res.output();
                    name = "RVQ 2 x " + rvq[k];
                }
                int space = fsq != null ? fsq.codebookSize() : sizes[k];
                long[] usage = new long[space];
                for (int c : codes) {
                    usage[c]++;
                }
                float[] y = Var.noGrad(() -> dec.forward(Var.of(qt.value()))).value().toArray();
                float[] xv = xt.toArray();
                double sig = 0;
                double noise = 0;
                for (int i = 0; i < xv.length; i++) {
                    sig += xv[i] * (double) xv[i];
                    double e = xv[i] - y[i];
                    noise += e * e;
                }
                rows.add(new CodecRow(name, space, 10 * Math.log10(sig / noise),
                        FiniteScalarQuantizer.perplexity(usage), FiniteScalarQuantizer.utilisation(usage)));
            }
        }
        return rows;
    }

    private static Tensor matrix(float[][] rows) {
        int n = rows.length;
        int d = rows[0].length;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(rows[i], 0, f, i * d, d);
        }
        return Tensor.of(Shape.of(n, d), f);
    }
}
