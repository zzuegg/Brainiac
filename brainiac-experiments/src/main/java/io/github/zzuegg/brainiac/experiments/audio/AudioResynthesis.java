package io.github.zzuegg.brainiac.experiments.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.HarmonicNoiseSynth;
import io.github.zzuegg.brainiac.experimental.nn.MultiScaleSpectralLoss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * DDSP resynthesis of simple melodies (7.6.5). Two synthetic instruments play random
 * melodies on a pentatonic scale: instrument A has every harmonic at {@code 1/k} and a
 * little breath noise, instrument B only the odd harmonics and more noise. From each clip
 * the f0 track ({@link HarmonicNoiseSynth#estimateF0}) and loudness are extracted; a
 * decoder maps them, frame by frame with one frame of context either side, to audio, and
 * is trained with the {@link MultiScaleSpectralLoss} on instrument A only:
 *
 * <ul>
 *   <li>DDSP: an MLP predicts harmonic amplitudes and noise magnitudes for the
 *       {@link HarmonicNoiseSynth};</li>
 *   <li>baseline: an MLP of about the same size predicting the frame's raw samples (a
 *       sample-level decoder without the synthesiser's bias).</li>
 * </ul>
 * Metrics on new melodies: the spectral distance to the target, the f0 error (cents) of
 * the output's pitch track against the target's, and timbre transfer: B's melodies
 * rendered by the A-trained decoder, scored against the same notes played by A.
 */
public final class AudioResynthesis {

    /** Samples per second. */
    public static final int RATE = 8000;
    /** Samples per control frame. */
    public static final int HOP = 64;
    /** Frames per clip. */
    public static final int FRAMES = 32;
    /** Harmonics of the synthesiser. */
    public static final int HARMONICS = 16;
    /** Noise bands of the synthesiser. */
    public static final int BANDS = 8;
    private static final int[] SCALES = {64, 128, 256};
    private static final double[] PITCHES = {220, 247, 277, 330, 370, 440, 494};
    private static final HarmonicNoiseSynth SYNTH = new HarmonicNoiseSynth(RATE, HOP, FRAMES, HARMONICS, BANDS, 5);

    private AudioResynthesis() {
    }

    /** One note of a melody: its pitch, first frame, length in frames and level. */
    record Note(double f0, int start, int length, double level) {
    }

    static List<Note> melody(SplittableRandom r) {
        List<Note> notes = new ArrayList<>();
        int t = r.nextInt(2);
        while (t < FRAMES - 2) {
            int len = Math.min(FRAMES - t, 4 + r.nextInt(5));
            notes.add(new Note(PITCHES[r.nextInt(PITCHES.length)], t, len, 0.3 + 0.5 * r.nextDouble()));
            t += len + (r.nextDouble() < 0.3 ? 1 : 0);
        }
        return notes;
    }

    /**
     * Plays a melody.
     *
     * @param notes the notes
     * @param oddOnly instrument B (odd harmonics, more noise) or A
     * @param r the noise and phase source
     * @return {@code FRAMES * HOP} samples
     */
    static float[] play(List<Note> notes, boolean oddOnly, SplittableRandom r) {
        float[] out = new float[FRAMES * HOP];
        for (Note n : notes) {
            int len = n.length() * HOP;
            double phase = r.nextDouble() * 6.28;
            for (int s = 0; s < len; s++) {
                int i = n.start() * HOP + s;
                if (i >= out.length) {
                    break;
                }
                double env = n.level() * Math.min(1, s / 48.0) * Math.exp(-2.0 * s / len) * Math.min(1, (len - s) / 32.0);
                phase += 2 * Math.PI * n.f0() / RATE;
                double v = 0;
                for (int k = 1; k * n.f0() < RATE / 2.0; k++) {
                    if (oddOnly && k % 2 == 0) {
                        continue;
                    }
                    v += Math.sin(k * phase) / k;
                }
                out[i] += (float) (env * (0.5 * v + (oddOnly ? 0.08 : 0.02) * r.nextGaussian()));
            }
        }
        return out;
    }

    /** Per-clip conditioning: f0 track, decoder inputs {@code [FRAMES][6]} and the oscillator bank. */
    record Controls(float[] f0, float[][] inputs, float[] oscillators) {
    }

    static Controls controls(float[] wave) {
        float[] f0 = HarmonicNoiseSynth.estimateF0(wave, RATE, HOP, FRAMES, 150, 600);
        float[] loud = HarmonicNoiseSynth.loudness(wave, HOP, FRAMES);
        float[][] in = new float[FRAMES][6];
        for (int t = 0; t < FRAMES; t++) {
            for (int c = -1; c <= 1; c++) {
                int s = Math.max(0, Math.min(FRAMES - 1, t + c));
                in[t][2 * (c + 1)] = f0[s] / 500f;
                in[t][2 * (c + 1) + 1] = (loud[s] + 40) / 40f;
            }
        }
        return new Controls(f0, in, SYNTH.oscillators(new float[][] {f0}).toArray());
    }

    /** A decoder from controls to audio. */
    interface Decoder {
        Var render(List<Controls> controls);

        List<Var> parameters();
    }

    private static Var[] mlp(SplittableRandom r, int... sizes) {
        Var[] w = new Var[2 * (sizes.length - 1)];
        for (int i = 0; i + 1 < sizes.length; i++) {
            w[2 * i] = Var.param(Tensor.normal(Shape.of(sizes[i], sizes[i + 1]), 0,
                    (float) Math.sqrt(2.0 / sizes[i]), r));
            w[2 * i + 1] = Var.param(Tensor.zeros(Shape.of(sizes[i + 1])));
        }
        return w;
    }

    private static Var apply(Var[] w, Var x) {
        Var h = x;
        for (int i = 0; i < w.length; i += 2) {
            h = h.matmul(w[i]).add(w[i + 1]);
            if (i + 2 < w.length) {
                h = h.relu();
            }
        }
        return h;
    }

    private static Var inputs(List<Controls> controls) {
        float[] x = new float[controls.size() * FRAMES * 6];
        for (int i = 0; i < controls.size(); i++) {
            for (int t = 0; t < FRAMES; t++) {
                System.arraycopy(controls.get(i).inputs()[t], 0, x, (i * FRAMES + t) * 6, 6);
            }
        }
        return Var.of(Tensor.of(Shape.of(controls.size(), FRAMES, 6), x));
    }

    /** The DDSP decoder: MLP to synthesiser controls. */
    static final class Ddsp implements Decoder {
        private final HarmonicNoiseSynth synth = SYNTH;
        private final Var[] w;

        Ddsp(long seed) {
            w = mlp(new SplittableRandom(seed), 6, 64, 64, HARMONICS + BANDS);
        }

        @Override
        public Var render(List<Controls> controls) {
            Var z = apply(w, inputs(controls));
            int per = HARMONICS * FRAMES * HOP;
            float[] osc = new float[controls.size() * per];
            for (int i = 0; i < controls.size(); i++) {
                System.arraycopy(controls.get(i).oscillators(), 0, osc, i * per, per);
            }
            Var amps = z.slice(2, 0, HARMONICS).sub(3f).softplus().mul(0.3f);
            Var noise = z.slice(2, HARMONICS, HARMONICS + BANDS).sub(3f).softplus().mul(0.05f);
            return synth.render(Tensor.of(Shape.of(controls.size(), HARMONICS, FRAMES * HOP), osc), amps, noise);
        }

        @Override
        public List<Var> parameters() {
            return List.of(w);
        }
    }

    /** The baseline: MLP straight to each frame's samples. */
    static final class SampleMlp implements Decoder {
        private final Var[] w;

        SampleMlp(long seed) {
            w = mlp(new SplittableRandom(seed), 6, 52, 52, HOP);
        }

        @Override
        public Var render(List<Controls> controls) {
            return apply(w, inputs(controls)).reshape(controls.size(), FRAMES * HOP);
        }

        @Override
        public List<Var> parameters() {
            return List.of(w);
        }
    }

    /**
     * Scores of one decoder.
     *
     * @param decoder its name
     * @param parameters trainable scalars
     * @param spectralDistance mean multi-scale spectral distance on instrument A test melodies
     * @param f0ErrorCents mean absolute f0 error, in cents, on frames voiced in both
     * @param transferDistance spectral distance of B's melodies rendered by the A decoder to A playing them
     */
    public record Score(String decoder, long parameters, double spectralDistance, double f0ErrorCents,
            double transferDistance) {
        @Override
        public String toString() {
            return String.format("%s (%d params): spectral %.3f, f0 error %.0f cents, transfer %.3f", decoder,
                    parameters, spectralDistance, f0ErrorCents, transferDistance);
        }
    }

    /**
     * Trains DDSP and the sample-level baseline on instrument A and scores both.
     *
     * @param steps training steps of 8 clips
     * @param seed the seed
     * @return {DDSP, sample MLP, and a reference row whose spectral distance is instrument B's
     *     own audio against A playing the same notes (no transfer)}
     */
    public static Score[] run(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> train = new ArrayList<>();
        List<Controls> trainCtl = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            float[] w = play(melody(r), false, r);
            train.add(w);
            trainCtl.add(controls(w));
        }
        List<float[]> testA = new ArrayList<>();
        List<Controls> testCtl = new ArrayList<>();
        List<Controls> transferCtl = new ArrayList<>();
        List<float[]> transferB = new ArrayList<>();
        SplittableRandom tr = new SplittableRandom(seed + 77);
        for (int i = 0; i < 16; i++) {
            List<Note> m = melody(tr);
            float[] a = play(m, false, tr);
            float[] b = play(m, true, tr);
            testA.add(a);
            testCtl.add(controls(a));
            transferB.add(b);
            transferCtl.add(controls(b));
        }
        Decoder[] decoders = {new Ddsp(seed), new SampleMlp(seed)};
        String[] names = {"DDSP", "sample MLP"};
        Score[] out = new Score[3];
        for (int d = 0; d < 2; d++) {
            Decoder dec = decoders[d];
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom br = new SplittableRandom(seed + 3);
            for (int s = 0; s < steps; s++) {
                List<Controls> bc = new ArrayList<>();
                float[] target = new float[8 * FRAMES * HOP];
                for (int i = 0; i < 8; i++) {
                    int j = br.nextInt(train.size());
                    bc.add(trainCtl.get(j));
                    System.arraycopy(train.get(j), 0, target, i * FRAMES * HOP, FRAMES * HOP);
                }
                Var loss = MultiScaleSpectralLoss.loss(dec.render(bc),
                        Var.of(Tensor.of(Shape.of(8, FRAMES * HOP), target)), SCALES);
                loss.backward();
                opt.step(dec.parameters());
            }
            float[] rendered = Var.noGrad(() -> dec.render(testCtl)).value().toArray();
            float[] transfer = Var.noGrad(() -> dec.render(transferCtl)).value().toArray();
            double spec = 0;
            double trans = 0;
            double cents = 0;
            int voiced = 0;
            for (int i = 0; i < testA.size(); i++) {
                float[] y = slice(rendered, i);
                spec += MultiScaleSpectralLoss.distance(y, testA.get(i), SCALES) / testA.size();
                trans += MultiScaleSpectralLoss.distance(slice(transfer, i), testA.get(i), SCALES) / testA.size();
                float[] fy = HarmonicNoiseSynth.estimateF0(y, RATE, HOP, FRAMES, 150, 600);
                float[] ft = testCtl.get(i).f0();
                for (int t = 0; t < FRAMES; t++) {
                    if (ft[t] > 0) {
                        // an unvoiced output frame counts as a full octave off
                        cents += fy[t] > 0 ? Math.min(1200, Math.abs(1200 * Math.log(fy[t] / ft[t]) / Math.log(2)))
                                : 1200;
                        voiced++;
                    }
                }
            }
            long params = 0;
            for (Var v : dec.parameters()) {
                params += v.shape().size();
            }
            out[d] = new Score(names[d], params, spec, cents / Math.max(1, voiced), trans);
        }
        double raw = 0;
        for (int i = 0; i < testA.size(); i++) {
            raw += MultiScaleSpectralLoss.distance(transferB.get(i), testA.get(i), SCALES) / testA.size();
        }
        out[2] = new Score("B audio itself", 0, Double.NaN, Double.NaN, raw);
        return out;
    }

    private static float[] slice(float[] all, int i) {
        float[] y = new float[FRAMES * HOP];
        System.arraycopy(all, i * FRAMES * HOP, y, 0, y.length);
        return y;
    }
}
