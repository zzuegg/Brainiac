package io.github.zzuegg.brainiac.experiments.speech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SpeakerVoices;
import io.github.zzuegg.brainiac.experimental.nn.ConvTasNet;
import io.github.zzuegg.brainiac.experimental.nn.PermutationInvariantLoss;
import io.github.zzuegg.brainiac.experimental.speech.Enhancement;
import io.github.zzuegg.brainiac.experimental.speech.SpeechFeatures;
import io.github.zzuegg.brainiac.experimental.speech.VoiceActivity;
import io.github.zzuegg.brainiac.experimental.timeseries.Pcen;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Speech front ends on {@link SpeakerVoices} (wave 10, items 10.6.6 and 10.6.7).
 *
 * <p>{@link #vad}: voice activity detection on conversations with silences in white, pink and
 * babble noise: the energy detector (with and without a zero-crossing veto, and on a PCEN
 * front end, {@link Pcen} with its default parameters), Sohn's likelihood ratio with and without
 * the HMM hangover, and the small CRNN ({@link VoiceActivity.Crnn}) trained on conversations at
 * random SNRs and noise types. Metrics: frame F1 (energy at a 6 dB margin, Sohn at 0.15, the
 * probabilities at 0.5) and AUC, and the CRNN's size.
 *
 * <p>{@link #enhancement}: single-channel enhancement of digit strings in stationary (white,
 * pink) and non-stationary (babble) noise: spectral subtraction, Wiener and MMSE-STSA gains
 * (noise from the Sohn VAD), the frame-MLP magnitude mask of wave 7's separation harness, the
 * CRNN with a magnitude or a complex ratio mask ({@link Enhancement.MaskNet}) and a one-source
 * {@link ConvTasNet}, against the noisy input and the ideal ratio mask (oracle). Metrics: SI-SNR
 * improvement, the log-spectral distance and, optionally, the digit error rate of a
 * {@link SpokenDigitRecognition} recogniser trained on clean speech.
 */
public final class SpeechFrontEnds {

    static final int MELS = 24;
    static final int HOP = SpeakerVoices.FRAME;
    static final int SAMPLES = 8192;

    private SpeechFrontEnds() {
    }

    // ---------------------------------------------------------------- 10.6.6 VAD

    /** VAD methods. */
    public enum Detector {
        /** Frame energy over the adaptive floor. */
        ENERGY,
        /** Energy with a zero-crossing veto. */
        ENERGY_ZCR,
        /** Energy of PCEN-normalised mel bands. */
        PCEN_ENERGY,
        /** Sohn's likelihood ratio. */
        SOHN,
        /** Sohn's likelihood ratio with the HMM hangover. */
        SOHN_HMM,
        /** The CRNN. */
        CRNN
    }

    /**
     * A VAD score.
     *
     * @param method the detector
     * @param f1 frame F1
     * @param auc frame AUC
     */
    public record VadRow(String method, double f1, double auc) {
        @Override
        public String toString() {
            return String.format("%s F1 %.3f AUC %.3f", method, f1, auc);
        }
    }

    static float[][] logMel(float[] wave) {
        float[][] lm = SpeechFeatures.logMel(wave, SpeakerVoices.RATE, HOP, MELS);
        for (float[] f : lm) {
            for (int b = 0; b < f.length; b++) {
                f[b] = Math.max(-3, f[b] / 5);
            }
        }
        return lm;
    }

    private static SpeakerVoices.Conversation conversation(List<SpeakerVoices.Voice> voices, SpeakerVoices.Noise noise,
            double snr, SplittableRandom r) {
        List<SpeakerVoices.Voice> who = List.of(voices.get(r.nextInt(voices.size())),
                voices.get(r.nextInt(voices.size())));
        return SpeakerVoices.conversation(who, 4, 0.1, noise, snr, r);
    }

    /**
     * Trains the CRNN detector on conversations.
     *
     * @param voices the training speakers
     * @param channels its width
     * @param steps training steps (batch 8 of 128-frame chunks)
     * @param seed the seed
     * @return the detector
     */
    public static VoiceActivity.Crnn trainCrnn(List<SpeakerVoices.Voice> voices, int channels, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        VoiceActivity.Crnn net = new VoiceActivity.Crnn(MELS, channels, r);
        Optimizer opt = Optimizer.adam(5e-3f);
        int chunk = 128;
        SpeakerVoices.Noise[] kinds = SpeakerVoices.Noise.values();
        for (int s = 0; s < steps; s++) {
            List<float[][]> feats = new ArrayList<>();
            float[] labels = new float[8 * chunk];
            for (int i = 0; i < 8; i++) {
                SpeakerVoices.Conversation c = conversation(voices, kinds[r.nextInt(kinds.length)], -5 + 25 * r.nextDouble(),
                        r);
                float[][] f = logMel(c.wave());
                boolean[] speech = c.speech();
                int frames = Math.min(f.length, speech.length);
                int start = frames > chunk ? r.nextInt(frames - chunk + 1) : 0;
                feats.add(SpeechFeatures.window(f, start, chunk));
                for (int t = 0; t < chunk; t++) {
                    labels[i * chunk + t] = start + t < frames && speech[start + t] ? 1 : 0;
                }
            }
            Var loss = net.loss(SpeechFeatures.batch(feats), Tensor.of(Shape.of(8, chunk), labels));
            loss.backward();
            opt.step(net.parameters());
        }
        return net;
    }

    /**
     * Voice activity detection per noise type and SNR.
     *
     * @param noises the noise types
     * @param snrs the SNRs in dB
     * @param conversations test conversations per condition
     * @param steps CRNN training steps
     * @param seed the seed
     * @param methods the detectors
     * @return per condition ("WHITE@10" and so on) the rows in method order
     */
    public static Map<String, List<VadRow>> vad(SpeakerVoices.Noise[] noises, double[] snrs, int conversations,
            int steps, long seed, Detector... methods) {
        List<SpeakerVoices.Voice> voices = SpeakerVoices.speakers(60, seed);
        List<SpeakerVoices.Voice> train = voices.subList(0, 40);
        List<SpeakerVoices.Voice> test = voices.subList(40, 60);
        VoiceActivity.Crnn crnn = Arrays.asList(methods).contains(Detector.CRNN) ? trainCrnn(train, 16, steps, seed)
                : null;
        Map<String, List<VadRow>> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed + 5);
        for (SpeakerVoices.Noise noise : noises) {
            for (double snr : snrs) {
                List<SpeakerVoices.Conversation> set = new ArrayList<>();
                for (int i = 0; i < conversations; i++) {
                    set.add(conversation(test, noise, snr, r));
                }
                List<VadRow> rows = new ArrayList<>();
                for (Detector d : methods) {
                    double f1 = 0;
                    double auc = 0;
                    for (SpeakerVoices.Conversation c : set) {
                        boolean[] truth = c.speech();
                        double[] score = score(d, c.wave(), crnn);
                        double threshold = switch (d) {
                            case ENERGY, ENERGY_ZCR, PCEN_ENERGY -> 6;
                            case SOHN -> 0.15;
                            default -> 0.5;
                        };
                        f1 += VoiceActivity.f1(VoiceActivity.decide(score, threshold), truth) / set.size();
                        auc += VoiceActivity.auc(score, truth) / set.size();
                    }
                    rows.add(new VadRow(d.name(), f1, auc));
                }
                out.put(noise + "@" + (int) snr, rows);
            }
        }
        return out;
    }

    static double[] score(Detector d, float[] wave, VoiceActivity.Crnn crnn) {
        return switch (d) {
            case ENERGY -> VoiceActivity.energy(VoiceActivity.logEnergy(wave, HOP), null);
            case ENERGY_ZCR -> VoiceActivity.energy(VoiceActivity.logEnergy(wave, HOP),
                    VoiceActivity.zeroCrossings(wave, HOP));
            case PCEN_ENERGY -> {
                int frames = wave.length / HOP;
                Var mel = Signal.logMel(Var.of(Tensor.of(Shape.of(wave.length), wave)), SpeakerVoices.RATE, 2 * HOP,
                        HOP, MELS).slice(0, 0, frames).exp().reshape(1, frames, MELS);
                float[] p = Var.noGrad(() -> new Pcen(MELS, frames, 0.04).forward(mel)).value().toArray();
                double[] e = new double[frames];
                for (int t = 0; t < frames; t++) {
                    double s = 0;
                    for (int m = 0; m < MELS; m++) {
                        s += p[t * MELS + m];
                    }
                    e[t] = 10 * Math.log10(s / MELS + 1e-6);
                }
                yield VoiceActivity.energy(e, null);
            }
            case SOHN -> VoiceActivity.sohn(wave, HOP);
            case SOHN_HMM -> VoiceActivity.hmmHangover(VoiceActivity.sohn(wave, HOP), 0.95);
            case CRNN -> crnn.probabilities(logMel(wave));
        };
    }

    // ---------------------------------------------------------------- 10.6.7 enhancement

    /** Enhancement methods. */
    public enum Enhancer {
        /** The noisy input. */
        NOISY,
        /** Spectral subtraction (alpha 2, floor 0.05). */
        SPECTRAL_SUBTRACTION,
        /** Decision-directed Wiener gain. */
        WIENER,
        /** Decision-directed MMSE-STSA gain. */
        MMSE_STSA,
        /** Wave 7's frame-MLP magnitude mask. */
        MLP_MASK,
        /** CRNN magnitude mask. */
        CRNN_MAGNITUDE,
        /** CRNN complex ratio mask. */
        CRNN_COMPLEX,
        /** Conv-TasNet with one output. */
        CONV_TASNET,
        /** The ideal ratio mask (oracle). */
        IDEAL_RATIO_MASK
    }

    /**
     * An enhancement score.
     *
     * @param method the method
     * @param siSnrImprovement mean SI-SNR improvement in dB
     * @param logSpectralDistance mean log-spectral distance to the clean speech in dB (lower is better)
     * @param digitError the recogniser's digit error rate (NaN if not measured)
     */
    public record EnhanceRow(String method, double siSnrImprovement, double logSpectralDistance, double digitError) {
        @Override
        public String toString() {
            return String.format("%s SI-SNRi %.2f dB LSD %.2f dB%s", method, siSnrImprovement, logSpectralDistance,
                    Double.isNaN(digitError) ? "" : String.format(" DER %.3f", digitError));
        }
    }

    /** A noisy utterance with its clean reference and digits. */
    record Pair(float[] noisy, float[] clean, int[] digits) {
    }

    static Pair pair(List<SpeakerVoices.Voice> voices, SpeakerVoices.Noise noise, double snr, SplittableRandom r) {
        int[] digits = new int[2 + r.nextInt(2)];
        for (int i = 0; i < digits.length; i++) {
            digits[i] = r.nextInt(10);
        }
        SpeakerVoices.Utterance u = SpeakerVoices.speak(digits, voices.get(r.nextInt(voices.size())), r);
        float[] clean = Arrays.copyOf(u.wave(), SAMPLES);
        boolean[] active = Arrays.copyOf(u.active(), SAMPLES);
        float[] noisy = SpeakerVoices.addNoise(clean, active, noise, snr, r)[0];
        return new Pair(noisy, clean, digits);
    }

    /**
     * Speech enhancement at a set of SNRs, for one noise type. The learned enhancers are
     * trained on all noise types at -5 to 15 dB.
     *
     * @param noise the test noise
     * @param snrs the test SNRs
     * @param testUtterances test utterances per SNR
     * @param steps training steps (batch 8)
     * @param recognizerSteps steps of the digit recogniser (0 to skip the digit error)
     * @param seed the seed
     * @param methods the methods
     * @return per SNR the rows in method order
     */
    public static Map<Double, List<EnhanceRow>> enhancement(SpeakerVoices.Noise noise, double[] snrs, int testUtterances,
            int steps, int recognizerSteps, long seed, Enhancer... methods) {
        List<SpeakerVoices.Voice> voices = SpeakerVoices.speakers(60, seed);
        List<SpeakerVoices.Voice> train = voices.subList(0, 40);
        List<SpeakerVoices.Voice> test = voices.subList(40, 60);
        SplittableRandom tr = new SplittableRandom(seed + 11);
        Map<Double, List<Pair>> sets = new LinkedHashMap<>();
        for (double snr : snrs) {
            List<Pair> set = new ArrayList<>();
            for (int i = 0; i < testUtterances; i++) {
                set.add(pair(test, noise, snr, tr));
            }
            sets.put(snr, set);
        }
        SpokenDigitRecognition.Recognizer rec = recognizerSteps > 0 ? recognizer(train, recognizerSteps, seed) : null;
        Map<Double, List<EnhanceRow>> out = new LinkedHashMap<>();
        for (double snr : snrs) {
            out.put(snr, new ArrayList<>());
        }
        for (Enhancer m : methods) {
            java.util.function.UnaryOperator<float[]> learned = null;
            if (m == Enhancer.MLP_MASK || m == Enhancer.CRNN_MAGNITUDE || m == Enhancer.CRNN_COMPLEX
                    || m == Enhancer.CONV_TASNET) {
                learned = trainEnhancer(m, train, steps, seed + m.ordinal());
            }
            for (double snr : snrs) {
                List<Pair> set = sets.get(snr);
                double imp = 0;
                double lsd = 0;
                List<float[][]> feats = new ArrayList<>();
                List<int[]> digits = new ArrayList<>();
                for (Pair p : set) {
                    float[] est = switch (m) {
                        case NOISY -> p.noisy();
                        case SPECTRAL_SUBTRACTION -> Enhancement.spectralSubtraction(p.noisy(), noiseFrames(p.noisy()), 2,
                                0.05);
                        case WIENER -> Enhancement.statisticalGain(p.noisy(), noiseFrames(p.noisy()), false);
                        case MMSE_STSA -> Enhancement.statisticalGain(p.noisy(), noiseFrames(p.noisy()), true);
                        case IDEAL_RATIO_MASK -> idealRatioMask(p);
                        default -> learned.apply(p.noisy());
                    };
                    imp += (PermutationInvariantLoss.siSnr(est, p.clean()) - PermutationInvariantLoss.siSnr(p.noisy(),
                            p.clean())) / set.size();
                    lsd += logSpectralDistance(est, p.clean()) / set.size();
                    if (rec != null) {
                        feats.add(SpokenDigitRecognition.features(SpokenDigitRecognition.Features.LOG_MEL, est,
                                SAMPLES / HOP));
                        digits.add(p.digits());
                    }
                }
                double der = rec != null ? SpokenDigitRecognition.digitErrorRate(rec.recognize(feats), digits)
                        : Double.NaN;
                out.get(snr).add(new EnhanceRow(m.name(), imp, lsd, der));
            }
        }
        return out;
    }

    private static boolean[] noiseFrames(float[] noisy) {
        double[] post = VoiceActivity.hmmHangover(VoiceActivity.sohn(noisy, HOP), 0.95);
        boolean[] noise = new boolean[post.length];
        for (int t = 0; t < post.length; t++) {
            noise[t] = post[t] < 0.5;
        }
        return noise;
    }

    private static java.util.function.UnaryOperator<float[]> trainEnhancer(Enhancer m, List<SpeakerVoices.Voice> train,
            int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SpeakerVoices.Noise[] kinds = SpeakerVoices.Noise.values();
        Enhancement.MaskNet mask = null;
        ConvTasNet tasnet = null;
        if (m == Enhancer.CONV_TASNET) {
            tasnet = new ConvTasNet(16, 32, 32, 5, 1, r);
        } else {
            mask = new Enhancement.MaskNet(64, m != Enhancer.MLP_MASK, m == Enhancer.CRNN_COMPLEX, r);
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            float[] noisy = new float[8 * SAMPLES];
            float[] clean = new float[8 * SAMPLES];
            for (int i = 0; i < 8; i++) {
                Pair p = pair(train, kinds[r.nextInt(kinds.length)], -5 + 20 * r.nextDouble(), r);
                System.arraycopy(p.noisy(), 0, noisy, i * SAMPLES, SAMPLES);
                System.arraycopy(p.clean(), 0, clean, i * SAMPLES, SAMPLES);
            }
            Tensor x = Tensor.of(Shape.of(8, SAMPLES), noisy);
            Tensor y = Tensor.of(Shape.of(8, SAMPLES), clean);
            Var loss;
            if (mask != null) {
                loss = mask.loss(x, y);
                loss.backward();
                opt.step(mask.parameters());
            } else {
                loss = PermutationInvariantLoss.siSnr(tasnet.forward(Var.of(x), true).reshape(8, SAMPLES), Var.of(y))
                        .mean().neg();
                loss.backward();
                opt.step(tasnet.parameters());
            }
        }
        Enhancement.MaskNet fm = mask;
        ConvTasNet ft = tasnet;
        return w -> fm != null ? fm.enhance(w)
                : Var.noGrad(() -> ft.forward(Var.of(Tensor.of(Shape.of(1, w.length), w)))).value().toArray();
    }

    private static float[] idealRatioMask(Pair p) {
        float[] noise = new float[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            noise[i] = p.noisy()[i] - p.clean()[i];
        }
        return Var.noGrad(() -> {
            Var mix = Signal.stft(Var.of(Tensor.of(Shape.of(SAMPLES), p.noisy())), Enhancement.N_FFT, Enhancement.HOP);
            Var s = Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(SAMPLES), p.clean())), Enhancement.N_FFT,
                    Enhancement.HOP)).sqrt();
            Var n = Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(SAMPLES), noise)), Enhancement.N_FFT,
                    Enhancement.HOP)).sqrt();
            int t = s.shape().dim(0);
            Var mask = s.div(s.add(n).add(1e-8f)).reshape(t, Enhancement.BINS, 1);
            return Signal.istft(mix.mul(mask), Enhancement.N_FFT, Enhancement.HOP, SAMPLES);
        }).value().toArray();
    }

    /**
     * The log-spectral distance: the RMS over bins of the difference of log power spectra
     * (dB), averaged over frames where the clean speech is within 40 dB of its loudest frame.
     *
     * @param estimate the enhanced waveform
     * @param clean the clean waveform
     * @return the distance in dB
     */
    public static double logSpectralDistance(float[] estimate, float[] clean) {
        float[] a = Var.noGrad(() -> Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(estimate.length), estimate)),
                Enhancement.N_FFT, Enhancement.HOP))).value().toArray();
        float[] b = Var.noGrad(() -> Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(clean.length), clean)),
                Enhancement.N_FFT, Enhancement.HOP))).value().toArray();
        int bins = Enhancement.BINS;
        int frames = b.length / bins;
        double[] energy = new double[frames];
        double max = -Double.MAX_VALUE;
        for (int t = 0; t < frames; t++) {
            for (int k = 0; k < bins; k++) {
                energy[t] += b[t * bins + k];
            }
            energy[t] = 10 * Math.log10(energy[t] + 1e-12);
            max = Math.max(max, energy[t]);
        }
        double total = 0;
        int used = 0;
        for (int t = 0; t < frames; t++) {
            if (energy[t] < max - 40) {
                continue;
            }
            double s = 0;
            for (int k = 0; k < bins; k++) {
                double d = 10 * Math.log10(a[t * bins + k] + 1e-8) - 10 * Math.log10(b[t * bins + k] + 1e-8);
                s += d * d;
            }
            total += Math.sqrt(s / bins);
            used++;
        }
        return total / Math.max(1, used);
    }

    private static SpokenDigitRecognition.Recognizer recognizer(List<SpeakerVoices.Voice> voices, int steps, long seed) {
        int frames = SAMPLES / HOP;
        SpokenDigitRecognition.Recognizer rec = new SpokenDigitRecognition.Recognizer(
                SpokenDigitRecognition.Encoder.MLP, SpokenDigitRecognition.width(SpokenDigitRecognition.Features.LOG_MEL),
                frames, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            List<float[][]> feats = new ArrayList<>();
            List<int[]> digits = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                Pair p = pair(voices, SpeakerVoices.Noise.WHITE, 30, r);
                feats.add(SpokenDigitRecognition.features(SpokenDigitRecognition.Features.LOG_MEL, p.clean(), frames));
                digits.add(p.digits());
            }
            rec.trainCtc(feats, digits, true, r);
        }
        return rec;
    }

    /**
     * Prints the long-budget runs.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        System.out.println("VAD: " + vad(SpeakerVoices.Noise.values(), new double[] {20, 10, 0}, 20, 1500, 1,
                Detector.values()));
        for (SpeakerVoices.Noise noise : SpeakerVoices.Noise.values()) {
            System.out.println("enhancement " + noise + ": " + enhancement(noise, new double[] {-5, 0, 5, 15}, 24, 1500,
                    1500, 1, Enhancer.values()));
        }
    }
}
