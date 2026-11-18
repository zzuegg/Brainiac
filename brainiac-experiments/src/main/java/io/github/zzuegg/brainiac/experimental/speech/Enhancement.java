package io.github.zzuegg.brainiac.experimental.speech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PermutationInvariantLoss;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Single-channel speech enhancement on the STFT (core {@link Signal#stft} /
 * {@link Signal#istft}).
 *
 * <ul>
 *   <li>{@link #spectralSubtraction} (Boll, 1979): subtract {@code alpha} times the noise
 *       magnitude spectrum, estimated from noise-only frames (from a VAD, or the quietest tenth
 *       of frames), floored at {@code beta} times the noisy magnitude; the noisy phase is kept;</li>
 *   <li>{@link #statisticalGain}: the decision-directed a-priori SNR
 *       {@code xi = a |S_prev|^2 / N + (1 - a) max(gamma - 1, 0)} (Ephraim and Malah, 1984) with
 *       either the Wiener gain {@code xi / (1 + xi)} or the MMSE short-time spectral amplitude
 *       gain {@code sqrt(pi v) / (2 gamma) exp(-v / 2) ((1 + v) I0(v / 2) + v I1(v / 2))},
 *       {@code v = xi gamma / (1 + xi)};</li>
 *   <li>{@link MaskNet}: a learned time-frequency mask from log magnitudes: a frame MLP with
 *       one frame of context on each side, optionally followed by a GRU (the CRNN), predicting a
 *       magnitude mask (sigmoid, noisy phase) or a complex ratio mask (Williamson et al., TASLP
 *       2016: real and imaginary parts, bounded by {@code 2 tanh}, which also corrects the
 *       phase), trained on the waveform with negative SI-SNR.</li>
 * </ul>
 * Papers: <a href="https://doi.org/10.1109/TASSP.1979.1163209">doi:10.1109/TASSP.1979.1163209</a>,
 * <a href="https://doi.org/10.1109/TASSP.1984.1164453">doi:10.1109/TASSP.1984.1164453</a>,
 * <a href="https://doi.org/10.1109/TASLP.2015.2512042">doi:10.1109/TASLP.2015.2512042</a>.
 */
public final class Enhancement {

    /** STFT size of the classical methods and the mask networks. */
    public static final int N_FFT = 256;
    /** STFT hop. */
    public static final int HOP = 64;
    /** Frequency bins. */
    public static final int BINS = N_FFT / 2 + 1;

    private Enhancement() {
    }

    private static float[][] spectrum(float[] wave) {
        Tensor s = Var.noGrad(() -> Signal.stft(Var.of(Tensor.of(Shape.of(wave.length), wave)), N_FFT, HOP)).value();
        float[] v = s.toArray();
        int frames = s.shape().dim(0);
        float[][] out = new float[frames][];
        for (int t = 0; t < frames; t++) {
            out[t] = Arrays.copyOfRange(v, t * BINS * 2, (t + 1) * BINS * 2);
        }
        return out;
    }

    private static float[] invert(float[][] spec, int samples) {
        int frames = spec.length;
        float[] v = new float[frames * BINS * 2];
        for (int t = 0; t < frames; t++) {
            System.arraycopy(spec[t], 0, v, t * BINS * 2, BINS * 2);
        }
        return Var.noGrad(() -> Signal.istft(Var.of(Tensor.of(Shape.of(frames, BINS, 2), v)), N_FFT, HOP, samples))
                .value().toArray();
    }

    private static double[] noisePsd(float[][] spec, boolean[] noiseFrames) {
        int frames = spec.length;
        double[] energy = new double[frames];
        for (int t = 0; t < frames; t++) {
            for (int k = 0; k < BINS; k++) {
                energy[t] += spec[t][2 * k] * spec[t][2 * k] + spec[t][2 * k + 1] * spec[t][2 * k + 1];
            }
        }
        boolean[] use = new boolean[frames];
        int count = 0;
        if (noiseFrames != null) {
            for (int t = 0; t < frames; t++) {
                // noiseFrames is at the VAD's frame rate; map by time
                int f = Math.min(noiseFrames.length - 1, (int) ((long) t * noiseFrames.length / frames));
                use[t] = noiseFrames[f];
                count += use[t] ? 1 : 0;
            }
        }
        if (count < 3) {
            double[] sorted = energy.clone();
            Arrays.sort(sorted);
            double cut = sorted[Math.max(0, frames / 10)];
            for (int t = 0; t < frames; t++) {
                use[t] = energy[t] <= cut;
            }
        }
        double[] psd = new double[BINS];
        int n = 0;
        for (int t = 0; t < frames; t++) {
            if (use[t]) {
                n++;
                for (int k = 0; k < BINS; k++) {
                    psd[k] += spec[t][2 * k] * spec[t][2 * k] + spec[t][2 * k + 1] * spec[t][2 * k + 1];
                }
            }
        }
        for (int k = 0; k < BINS; k++) {
            psd[k] = psd[k] / Math.max(1, n) + 1e-10;
        }
        return psd;
    }

    /**
     * Magnitude spectral subtraction.
     *
     * @param noisy the noisy waveform
     * @param noiseFrames noise-only frames from a VAD (any frame rate), or {@code null}
     * @param alpha the over-subtraction factor (such as 2)
     * @param beta the spectral floor (such as 0.05)
     * @return the enhanced waveform
     */
    public static float[] spectralSubtraction(float[] noisy, boolean[] noiseFrames, double alpha, double beta) {
        float[][] spec = spectrum(noisy);
        double[] psd = noisePsd(spec, noiseFrames);
        for (float[] f : spec) {
            for (int k = 0; k < BINS; k++) {
                double re = f[2 * k];
                double im = f[2 * k + 1];
                double mag = Math.sqrt(re * re + im * im) + 1e-12;
                double clean = Math.max(mag - alpha * Math.sqrt(psd[k]), beta * mag);
                double g = clean / mag;
                f[2 * k] = (float) (re * g);
                f[2 * k + 1] = (float) (im * g);
            }
        }
        return invert(spec, noisy.length);
    }

    /**
     * Decision-directed statistical enhancement.
     *
     * @param noisy the noisy waveform
     * @param noiseFrames noise-only frames from a VAD, or {@code null}
     * @param mmse the MMSE-STSA gain instead of Wiener
     * @return the enhanced waveform
     */
    public static float[] statisticalGain(float[] noisy, boolean[] noiseFrames, boolean mmse) {
        float[][] spec = spectrum(noisy);
        double[] psd = noisePsd(spec, noiseFrames);
        double a = 0.98;
        double[] prev = new double[BINS];
        for (float[] f : spec) {
            for (int k = 0; k < BINS; k++) {
                double p = f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
                double gamma = Math.min(p / psd[k], 1e4);
                double xi = Math.max(a * prev[k] / psd[k] + (1 - a) * Math.max(gamma - 1, 0), 10e-3);
                double g;
                if (mmse) {
                    double v = xi * gamma / (1 + xi);
                    g = Math.sqrt(Math.PI * v) / (2 * Math.max(gamma, 1e-6))
                            * ((1 + v) * besselI0e(v / 2) + v * besselI1e(v / 2));
                    g = Math.min(g, 1.5);
                } else {
                    g = xi / (1 + xi);
                }
                prev[k] = g * g * p;
                f[2 * k] *= (float) g;
                f[2 * k + 1] *= (float) g;
            }
        }
        return invert(spec, noisy.length);
    }

    /** {@code exp(-x) I0(x)} (Abramowitz and Stegun 9.8.1-9.8.2). */
    static double besselI0e(double x) {
        double ax = Math.abs(x);
        if (ax < 3.75) {
            double y = (x / 3.75) * (x / 3.75);
            return Math.exp(-ax) * (1 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732
                    + y * (0.0360768 + y * 0.0045813))))));
        }
        double y = 3.75 / ax;
        return (1 / Math.sqrt(ax)) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565
                + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
    }

    /** {@code exp(-x) I1(x)} (Abramowitz and Stegun 9.8.3-9.8.4). */
    static double besselI1e(double x) {
        double ax = Math.abs(x);
        if (ax < 3.75) {
            double y = (x / 3.75) * (x / 3.75);
            return Math.exp(-ax) * ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.02658733
                    + y * (0.00301532 + y * 0.00032411))))));
        }
        double y = 3.75 / ax;
        return (1 / Math.sqrt(ax)) * (0.39894228 + y * (-0.03988024 + y * (-0.00362018 + y * (0.00163801
                + y * (-0.01031555 + y * (0.02282967 + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059))))))));
    }

    /**
     * A learned mask enhancer: waveforms {@code [b, samples]} to enhanced waveforms.
     */
    public static final class MaskNet implements Layer {
        private final boolean recurrent;
        private final boolean complex;
        private final int hidden;
        private final Var w1;
        private final Var b1;
        private final Layer gru;
        private final Var w2;
        private final Var b2;

        /**
         * Creates the network.
         *
         * @param hidden the hidden width
         * @param recurrent whether a GRU follows the frame MLP (the CRNN)
         * @param complex a complex ratio mask instead of a magnitude mask
         * @param random the source of initial weights
         */
        public MaskNet(int hidden, boolean recurrent, boolean complex, SplittableRandom random) {
            this.hidden = hidden;
            this.recurrent = recurrent;
            this.complex = complex;
            w1 = Var.param(Tensor.normal(Shape.of(3 * BINS, hidden), 0, (float) Math.sqrt(2.0 / (3 * BINS)), random));
            b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
            gru = recurrent ? Recurrent.gru(hidden).returnSequences(true).create(Shape.of(8, hidden), random) : null;
            int outs = complex ? 2 * BINS : BINS;
            w2 = Var.param(Tensor.normal(Shape.of(hidden, outs), 0, (float) Math.sqrt(1.0 / hidden), random));
            b2 = Var.param(Tensor.full(Shape.of(outs), complex ? 0f : 1f));
        }

        @Override
        public Var forward(Var waves) {
            int b = waves.shape().dim(0);
            int samples = waves.shape().dim(1);
            Var spec = Signal.stft(waves, N_FFT, HOP);                              // [b, T, F, 2]
            int t = spec.shape().dim(1);
            Var logMag = Signal.power(spec).add(1e-6f).log().mul(0.1f);
            Var prev = Var.concat(1, logMag.slice(1, 0, 1), logMag.slice(1, 0, t - 1));
            Var next = Var.concat(1, logMag.slice(1, 1, t), logMag.slice(1, t - 1, t));
            Var h = Var.concat(2, prev, logMag, next).matmul(w1).add(b1).relu();
            if (gru != null) {
                h = gru.forward(h).add(h);
            }
            Var m = h.matmul(w2).add(b2);
            Var out;
            if (complex) {
                Var mr = m.slice(2, 0, BINS).tanh().mul(2f).reshape(b, t, BINS, 1);
                Var mi = m.slice(2, BINS, 2 * BINS).tanh().mul(2f).reshape(b, t, BINS, 1);
                Var yr = spec.slice(3, 0, 1);
                Var yi = spec.slice(3, 1, 2);
                out = Var.concat(3, yr.mul(mr).sub(yi.mul(mi)), yr.mul(mi).add(yi.mul(mr)));
            } else {
                out = spec.mul(m.sigmoid().reshape(b, t, BINS, 1));
            }
            return Signal.istft(out, N_FFT, HOP, samples);
        }

        /**
         * The negative SI-SNR loss.
         *
         * @param noisy {@code [b, samples]}
         * @param clean {@code [b, samples]}
         * @return the mean loss
         */
        public Var loss(Tensor noisy, Tensor clean) {
            return PermutationInvariantLoss.siSnr(forward(Var.of(noisy)), Var.of(clean)).mean().neg();
        }

        /**
         * Enhances one waveform.
         *
         * @param noisy the samples
         * @return the enhanced samples
         */
        public float[] enhance(float[] noisy) {
            return Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, noisy.length), noisy)))).value().toArray();
        }

        @Override
        public Shape outputShape(Shape input) {
            return input;
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(w1, b1));
            if (gru != null) {
                p.addAll(gru.parameters());
            }
            p.add(w2);
            p.add(b2);
            return p;
        }

        @Override
        public String toString() {
            return (recurrent ? "CRNN " : "MLP ") + (complex ? "complex ratio mask" : "magnitude mask") + " (" + hidden
                    + ")";
        }
    }
}
