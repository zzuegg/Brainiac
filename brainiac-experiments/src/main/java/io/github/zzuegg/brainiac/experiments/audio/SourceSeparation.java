package io.github.zzuegg.brainiac.experiments.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.experimental.nn.ConvTasNet;
import io.github.zzuegg.brainiac.experimental.nn.PermutationInvariantLoss;
import io.github.zzuegg.brainiac.experimental.timeseries.DualPathRnn;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Two-speaker separation of spoken digits (7.6.6). Each source is one digit of
 * {@link SpokenDigitRecognition#speak}, from a low-formant speaker (scale 0.88-0.96) or a
 * high-formant one (1.04-1.12), placed at a random offset in a 1024-sample clip (cut if longer); the second
 * is scaled to a given signal-to-interference ratio (0 or 5 dB) and added. Methods:
 *
 * <ul>
 *   <li>{@link ConvTasNet} trained with uPIT negative SI-SNR ({@link PermutationInvariantLoss});</li>
 *   <li>an STFT-mask MLP (log magnitudes of three frames in, one sigmoid mask per source
 *       out, applied to the mixture's STFT and inverted with core {@link Signal#istft}),
 *       trained with the same loss at about the same parameter count;</li>
 *   <li>FastICA ({@link ComponentAnalysis#fastIca}) on a two-microphone instantaneous mixture
 *       of the same sources (it needs as many channels as sources);</li>
 *   <li>the ideal ratio mask (oracle) {@code |S_i| / sum_j |S_j|} on the mixture's STFT.</li>
 * </ul>
 * Metric: SI-SNR improvement over the (first-microphone) mixture, in dB.
 */
public final class SourceSeparation {

    /** Samples per clip. */
    public static final int SAMPLES = 1024;
    private static final int N_FFT = 128;
    private static final int HOP = 32;
    private static final int BINS = N_FFT / 2 + 1;
    private static final float LEARNING_RATE = 5e-3f;

    private SourceSeparation() {
    }

    static float[] source(boolean low, SplittableRandom r) {
        SpokenDigitRecognition.Speaker sp = new SpokenDigitRecognition.Speaker(
                (low ? 0.88 : 1.04) + 0.08 * r.nextDouble(), 0.9 + 0.3 * r.nextDouble());
        float[] w = SpokenDigitRecognition.speak(new int[] {r.nextInt(10)}, sp, 40, r);
        float[] out = new float[SAMPLES];
        int offset = w.length >= SAMPLES ? 0 : r.nextInt(SAMPLES - w.length + 1);
        System.arraycopy(w, 0, out, offset, Math.min(w.length, SAMPLES));
        return out;
    }

    private static double power(float[] x) {
        double p = 1e-12;
        for (float v : x) {
            p += v * v;
        }
        return p / x.length;
    }

    /** A mixture: two references (the second already scaled) and their sum. */
    record Mixture(float[][] sources, float[] mixture) {
    }

    static Mixture mix(float[] a, float[] b, double sirDb) {
        double g = Math.sqrt(power(a) / power(b) / Math.pow(10, sirDb / 10));
        float[] s2 = new float[SAMPLES];
        float[] m = new float[SAMPLES];
        for (int i = 0; i < SAMPLES; i++) {
            s2[i] = (float) (b[i] * g);
            m[i] = a[i] + s2[i];
        }
        return new Mixture(new float[][] {a, s2}, m);
    }

    /** A single-channel separator trained with uPIT. */
    interface Separator {
        /** @return {@code [b, 2, SAMPLES]} */
        Var separate(Var mixtures);

        List<Var> parameters();
    }

    /** The STFT-mask MLP. */
    static final class StftMask implements Separator {
        private final Var w1;
        private final Var b1;
        private final Var w2;
        private final Var b2;

        StftMask(int hidden, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            w1 = Var.param(Tensor.normal(Shape.of(3 * BINS, hidden), 0, (float) Math.sqrt(2.0 / (3 * BINS)), r));
            b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
            w2 = Var.param(Tensor.normal(Shape.of(hidden, 2 * BINS), 0, (float) Math.sqrt(1.0 / hidden), r));
            b2 = Var.param(Tensor.zeros(Shape.of(2 * BINS)));
        }

        @Override
        public Var separate(Var mixtures) {
            int b = mixtures.shape().dim(0);
            Var spec = Signal.stft(mixtures, N_FFT, HOP);                       // [b, T, F, 2]
            int t = spec.shape().dim(1);
            Var logMag = Signal.power(spec).add(1e-6f).log().mul(0.1f);         // [b, T, F]
            Var prev = Var.concat(1, logMag.slice(1, 0, 1), logMag.slice(1, 0, t - 1));
            Var next = Var.concat(1, logMag.slice(1, 1, t), logMag.slice(1, t - 1, t));
            Var in = Var.concat(2, prev, logMag, next);
            Var masks = in.matmul(w1).add(b1).relu().matmul(w2).add(b2).sigmoid();
            Var[] outs = new Var[2];
            for (int i = 0; i < 2; i++) {
                Var m = masks.slice(2, i * BINS, (i + 1) * BINS).reshape(b, t, BINS, 1);
                outs[i] = Signal.istft(spec.mul(m), N_FFT, HOP, SAMPLES).reshape(b, 1, SAMPLES);
            }
            return Var.concat(1, outs);
        }

        @Override
        public List<Var> parameters() {
            return List.of(w1, b1, w2, b2);
        }
    }

    /** DPRNN as a {@link Separator}: window 16, 32 filters, width 32, chunks of 16 frames, two blocks. */
    static final class Dprnn implements Separator {
        private final DualPathRnn net;

        Dprnn(long seed) {
            net = new DualPathRnn(16, 32, 32, 16, 2, 2, SAMPLES, new SplittableRandom(seed));
        }

        @Override
        public Var separate(Var mixtures) {
            return net.forward(mixtures, true);
        }

        @Override
        public List<Var> parameters() {
            return net.parameters();
        }
    }

    /** Conv-TasNet as a {@link Separator}. */
    static final class TasNet implements Separator {
        private final ConvTasNet net;

        TasNet(long seed) {
            net = new ConvTasNet(16, 32, 32, 5, 2, new SplittableRandom(seed));
        }

        @Override
        public Var separate(Var mixtures) {
            return net.forward(mixtures, true);
        }

        @Override
        public List<Var> parameters() {
            return net.parameters();
        }
    }

    /**
     * One method's scores.
     *
     * @param method its name
     * @param parameters trainable scalars (0 for the untrained ones)
     * @param improvement0 mean SI-SNR improvement at 0 dB, in dB
     * @param improvement5 at 5 dB
     */
    public record Score(String method, long parameters, double improvement0, double improvement5) {
        @Override
        public String toString() {
            return String.format("%s (%d params): SI-SNRi %.1f dB at 0 dB, %.1f dB at 5 dB", method, parameters,
                    improvement0, improvement5);
        }
    }

    static Tensor stack(List<Mixture> batch, boolean references) {
        int n = batch.size();
        float[] x = new float[n * (references ? 2 : 1) * SAMPLES];
        for (int i = 0; i < n; i++) {
            if (references) {
                System.arraycopy(batch.get(i).sources()[0], 0, x, (2 * i) * SAMPLES, SAMPLES);
                System.arraycopy(batch.get(i).sources()[1], 0, x, (2 * i + 1) * SAMPLES, SAMPLES);
            } else {
                System.arraycopy(batch.get(i).mixture(), 0, x, i * SAMPLES, SAMPLES);
            }
        }
        return references ? Tensor.of(Shape.of(n, 2, SAMPLES), x) : Tensor.of(Shape.of(n, SAMPLES), x);
    }

    /** The separation methods. */
    public enum Method {
        /** Conv-TasNet with uPIT. */
        CONV_TASNET("Conv-TasNet"),
        /** The STFT-mask MLP with uPIT. */
        STFT_MASK("STFT-mask MLP"),
        /** FastICA on two microphones. */
        FASTICA("FastICA (2 mics)"),
        /** The ideal ratio mask (oracle). */
        IDEAL_RATIO_MASK("ideal ratio mask"),
        /** Dual-path RNN ({@link DualPathRnn}) on the Conv-TasNet encoder and decoder (wave 8, 8.2.18). */
        DPRNN("DPRNN");

        private final String label;

        Method(String label) {
            this.label = label;
        }

        /** @return a readable name */
        public String label() {
            return label;
        }
    }

    /**
     * Trains the learned separators on 0-5 dB mixtures and scores the methods on 24 test
     * mixtures at 0 dB and 24 at 5 dB.
     *
     * @param steps training steps of 8 mixtures
     * @param seed the seed
     * @param methods the methods to score
     * @return their scores, in the given order
     */
    public static Score[] run(int steps, long seed, Method... methods) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> low = new ArrayList<>();
        List<float[]> high = new ArrayList<>();
        for (int i = 0; i < 160; i++) {
            low.add(source(true, r));
            high.add(source(false, r));
        }
        List<List<Mixture>> tests = new ArrayList<>();
        List<List<float[][]>> micPairs = new ArrayList<>();
        SplittableRandom tr = new SplittableRandom(seed + 99);
        for (double sir : new double[] {0, 5}) {
            List<Mixture> set = new ArrayList<>();
            List<float[][]> mics = new ArrayList<>();
            for (int i = 0; i < 24; i++) {
                Mixture m = tr.nextBoolean() ? mix(source(true, tr), source(false, tr), sir)
                        : mix(source(false, tr), source(true, tr), sir);
                set.add(m);
                double a = 0.3 + 0.5 * tr.nextDouble();
                double c = 0.3 + 0.5 * tr.nextDouble();
                float[][] pair = new float[2][SAMPLES];
                for (int k = 0; k < SAMPLES; k++) {
                    pair[0][k] = (float) (m.sources()[0][k] + a * m.sources()[1][k]);
                    pair[1][k] = (float) (c * m.sources()[0][k] + m.sources()[1][k]);
                }
                mics.add(pair);
            }
            tests.add(set);
            micPairs.add(mics);
        }
        Score[] out = new Score[methods.length];
        for (int mi = 0; mi < methods.length; mi++) {
            Method method = methods[mi];
            if (method == Method.FASTICA || method == Method.IDEAL_RATIO_MASK) {
                out[mi] = classical(method, tests, micPairs, seed);
                continue;
            }
            Separator sep = switch (method) {
                case CONV_TASNET -> new TasNet(seed);
                case DPRNN -> new Dprnn(seed);
                default -> new StftMask(96, seed);
            };
            Optimizer opt = Optimizer.adam(LEARNING_RATE);
            SplittableRandom br = new SplittableRandom(seed + 5);
            for (int step = 0; step < steps; step++) {
                List<Mixture> batch = new ArrayList<>();
                for (int i = 0; i < 8; i++) {
                    float[] a = low.get(br.nextInt(low.size()));
                    float[] b = high.get(br.nextInt(high.size()));
                    batch.add(br.nextBoolean() ? mix(a, b, 5 * br.nextDouble()) : mix(b, a, 5 * br.nextDouble()));
                }
                Var loss = PermutationInvariantLoss.loss(sep.separate(Var.of(stack(batch, false))),
                        Var.of(stack(batch, true)));
                loss.backward();
                opt.step(sep.parameters());
            }
            double[] imp = new double[2];
            for (int k = 0; k < 2; k++) {
                List<Mixture> set = tests.get(k);
                float[] est = Var.noGrad(() -> sep.separate(Var.of(stack(set, false)))).value().toArray();
                for (int i = 0; i < set.size(); i++) {
                    float[][] e = new float[2][SAMPLES];
                    System.arraycopy(est, (2 * i) * SAMPLES, e[0], 0, SAMPLES);
                    System.arraycopy(est, (2 * i + 1) * SAMPLES, e[1], 0, SAMPLES);
                    imp[k] += PermutationInvariantLoss.improvement(e, set.get(i).sources(), set.get(i).mixture())
                            / set.size();
                }
            }
            long params = 0;
            for (Var v : sep.parameters()) {
                params += v.shape().size();
            }
            out[mi] = new Score(method.label(), params, imp[0], imp[1]);
        }
        return out;
    }

    private static Score classical(Method method, List<List<Mixture>> tests, List<List<float[][]>> micPairs,
            long seed) {
        double[] ica = new double[2];
        double[] irm = new double[2];
        for (int k = 0; k < 2; k++) {
            List<Mixture> set = tests.get(k);
            for (int i = 0; i < set.size(); i++) {
                float[][] mics = micPairs.get(k).get(i);
                if (method == Method.FASTICA) {
                    double[][] x = new double[SAMPLES][2];
                    for (int n = 0; n < SAMPLES; n++) {
                        x[n][0] = mics[0][n];
                        x[n][1] = mics[1][n];
                    }
                    double[][] y = ComponentAnalysis.fastIca(x, 2, 200, seed + i).transform(x);
                    float[][] e = new float[2][SAMPLES];
                    for (int n = 0; n < SAMPLES; n++) {
                        e[0][n] = (float) y[n][0];
                        e[1][n] = (float) y[n][1];
                    }
                    ica[k] += PermutationInvariantLoss.improvement(e, set.get(i).sources(), mics[0]) / set.size();
                } else {
                    irm[k] += PermutationInvariantLoss.improvement(idealRatioMask(set.get(i)), set.get(i).sources(),
                            set.get(i).mixture()) / set.size();
                }
            }
        }
        double[] v = method == Method.FASTICA ? ica : irm;
        return new Score(method.label(), 0, v[0], v[1]);
    }

    static float[][] idealRatioMask(Mixture m) {
        return Var.noGrad(() -> {
            Var mixSpec = Signal.stft(Var.of(Tensor.of(Shape.of(SAMPLES), m.mixture())), N_FFT, HOP);
            Var a = Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(SAMPLES), m.sources()[0])), N_FFT, HOP))
                    .sqrt();
            Var b = Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(SAMPLES), m.sources()[1])), N_FFT, HOP))
                    .sqrt();
            Var total = a.add(b).add(1e-8f);
            int t = a.shape().dim(0);
            float[][] out = new float[2][];
            Var[] mags = {a, b};
            for (int i = 0; i < 2; i++) {
                Var mask = mags[i].div(total).reshape(t, BINS, 1);
                out[i] = Signal.istft(mixSpec.mul(mask), N_FFT, HOP, SAMPLES).value().toArray();
            }
            return out;
        });
    }

    /**
     * The learning curve of 8.2.18: each trained separator is retrained from scratch at every
     * budget (the {@link #run} task and test sets), so the wave 7 comparison at 150 steps is
     * repeated at larger budgets.
     *
     * @param budgets training steps, for example {150, 600, 1500}
     * @param seed the seed
     * @param methods the trained separators
     * @return per budget, the scores in the order of {@code methods}
     */
    public static java.util.Map<Integer, Score[]> learningCurve(int[] budgets, long seed, Method... methods) {
        java.util.Map<Integer, Score[]> out = new java.util.LinkedHashMap<>();
        for (int steps : budgets) {
            out.put(steps, run(steps, seed, methods));
        }
        return out;
    }
}
