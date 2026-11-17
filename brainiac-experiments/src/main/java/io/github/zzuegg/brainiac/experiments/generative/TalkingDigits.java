package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.CtcLoss;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * X30, talking digits at toy scale: a synthesizer that "speaks" digit strings as
 * formant-like tone sequences, and a recognizer trained with {@code CtcLoss} that hears
 * them back. Each digit is two vowel-like segments, each a mix of two formant tones whose
 * frequencies depend on the digit; durations, loudness, pauses and noise vary. Audio is
 * cut into frames, turned into log band energies with the core STFT, stacked with
 * neighbouring frames for context and classified per frame into blank or a digit; CTC
 * trains this without any frame-level alignment, and best-path decoding reads the string.
 */
public final class TalkingDigits {
    /** Samples per second. */
    public static final int RATE = 8000;
    /** Samples per frame (16 ms). */
    public static final int FRAME = 128;
    /** Log-energy bands per frame. */
    public static final int BANDS = 16;
    private static final int CONTEXT = 2;
    private final int frames;
    private final Model model;
    private final Optimizer optimizer = Optimizer.adam(3e-3f);

    private TalkingDigits(int frames, long seed) {
        this.frames = frames;
        model = Model.sequential().input(frames, BANDS * (2 * CONTEXT + 1)).dense(96, Activation.RELU)
                .dense(96, Activation.RELU).dense(11).seed(seed).build();
    }

    /**
     * Creates an untrained recognizer.
     *
     * @param frames the frames per utterance (audio is padded or cut to this)
     * @param seed the seed
     * @return the recognizer
     */
    public static TalkingDigits of(int frames, long seed) {
        return new TalkingDigits(frames, seed);
    }

    /** The two formant pairs (Hz) of a digit's two segments. */
    static double[][] formants(int digit) {
        double f1 = 300 + 110 * digit;
        double f2 = 2500 - 170 * ((digit * 7) % 10);
        return new double[][] {{f1, f2}, {f1 * 1.35, f2 * 0.75 + 200 * (digit % 3)}};
    }

    /**
     * Speaks a digit string.
     *
     * @param digits the digits 0..9
     * @param random varies durations, loudness, pauses and noise
     * @return the waveform
     */
    public static float[] speak(int[] digits, SplittableRandom random) {
        List<Float> wave = new ArrayList<>();
        silence(wave, FRAME * (1 + random.nextInt(2)), random);
        for (int d : digits) {
            double amp = 0.4 + 0.4 * random.nextDouble();
            for (double[] seg : formants(d)) {
                int n = FRAME * (2 + random.nextInt(2)) + random.nextInt(FRAME);
                double p1 = random.nextDouble() * 6.28;
                double p2 = random.nextDouble() * 6.28;
                for (int s = 0; s < n; s++) {
                    double env = Math.min(1, Math.min(s, n - s) / 60.0);
                    double v = Math.sin(2 * Math.PI * seg[0] * s / RATE + p1)
                            + 0.6 * Math.sin(2 * Math.PI * seg[1] * s / RATE + p2);
                    wave.add((float) (amp * env * v * 0.6 + 0.02 * random.nextGaussian()));
                }
            }
            silence(wave, FRAME * (1 + random.nextInt(2)), random);
        }
        float[] out = new float[wave.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = wave.get(i);
        }
        return out;
    }

    private static void silence(List<Float> wave, int n, SplittableRandom random) {
        for (int i = 0; i < n; i++) {
            wave.add((float) (0.02 * random.nextGaussian()));
        }
    }

    /**
     * Log band energies per frame, from the power of the core short-time Fourier transform
     * ({@code Signal.stft}: Hann window, one frame every {@code FRAME} samples, centred on it).
     *
     * @param wave the waveform
     * @param frames the number of frames to produce (silence pads, the rest is cut)
     * @return {@code [frames][BANDS]}
     */
    public static float[][] features(float[] wave, int frames) {
        float[] samples = java.util.Arrays.copyOf(wave, frames * FRAME);
        float[] power = Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(samples.length), samples)), FRAME, FRAME))
                .value().toArray();   // [frames + 1, FRAME / 2 + 1]
        int bins = FRAME / 2 + 1;
        float[][] out = new float[frames][BANDS];
        for (int f = 0; f < frames; f++) {
            for (int k = 1; k < bins; k++) {
                out[f][Math.min(BANDS - 1, (k - 1) * BANDS / (bins - 1))] += power[f * bins + k];
            }
            for (int b = 0; b < BANDS; b++) {
                out[f][b] = (float) (Math.log(out[f][b] + 1e-3) / 5);
            }
        }
        return out;
    }

    /** Stacks each frame with its neighbours, giving the model some context. */
    private Tensor input(List<float[]> waves) {
        int width = BANDS * (2 * CONTEXT + 1);
        float[] x = new float[waves.size() * frames * width];
        for (int i = 0; i < waves.size(); i++) {
            float[][] f = features(waves.get(i), frames);
            for (int t = 0; t < frames; t++) {
                for (int c = -CONTEXT; c <= CONTEXT; c++) {
                    int src = Math.max(0, Math.min(frames - 1, t + c));
                    System.arraycopy(f[src], 0, x, ((i * frames + t) * (2 * CONTEXT + 1) + c + CONTEXT) * BANDS, BANDS);
                }
            }
        }
        return Tensor.of(Shape.of(waves.size(), frames, width), x);
    }

    /**
     * Draws random utterances.
     *
     * @param n how many
     * @param maxDigits the longest digit string
     * @param random the random source
     * @param digits receives the digit strings
     * @return the waveforms
     */
    public static List<float[]> utterances(int n, int maxDigits, SplittableRandom random, List<int[]> digits) {
        List<float[]> waves = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] d = new int[1 + random.nextInt(maxDigits)];
            for (int j = 0; j < d.length; j++) {
                d[j] = random.nextInt(10);
            }
            digits.add(d);
            waves.add(speak(d, random));
        }
        return waves;
    }

    /**
     * One CTC training step.
     *
     * @param waves waveforms
     * @param digits their digit strings
     * @return the loss
     */
    public float trainStep(List<float[]> waves, List<int[]> digits) {
        int[][] labels = new int[digits.size()][];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = digits.get(i).clone();
            for (int j = 0; j < labels[i].length; j++) {
                labels[i][j] += 1;   // class 0 is the blank
            }
        }
        Var loss = CtcLoss.loss(model.forward(Var.of(input(waves)), true), labels);
        float value = loss.value().item();
        loss.backward();
        optimizer.step(model.parameters());
        return value;
    }

    /**
     * Recognizes digit strings.
     *
     * @param waves waveforms
     * @return the decoded digits per waveform
     */
    public int[][] recognize(List<float[]> waves) {
        int[][] decoded = CtcLoss.greedyDecode(model.forward(Var.of(input(waves))).value());
        for (int[] d : decoded) {
            for (int j = 0; j < d.length; j++) {
                d[j] -= 1;
            }
        }
        return decoded;
    }
}
