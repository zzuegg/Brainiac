package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * The DDSP harmonic-plus-noise synthesiser: a differentiable renderer from per-frame,
 * interpretable controls to audio. A fundamental frequency track (given, as DDSP takes
 * it from a pitch tracker) fixes the phase of every harmonic; the network's per-frame
 * harmonic amplitudes {@code [b, frames, harmonics]} are linearly interpolated to the
 * sample rate and weight {@code sin(k phi(n))} (harmonics above Nyquist are silenced);
 * per-frame noise magnitudes {@code [b, frames, bands]} weight a fixed bank of band-limited
 * white noises, a time-varying filtered noise. Gradients reach the amplitudes and noise
 * magnitudes; the oscillator bank and the noise bank are constants.
 *
 * <p>{@link #estimateF0} (normalised autocorrelation with parabolic refinement) and
 * {@link #loudness} extract the conditioning from audio.
 *
 * <p>Paper: Engel, Hantrakul, Gu, Roberts, "DDSP: Differentiable Digital Signal Processing"
 * (ICLR 2020), <a href="https://arxiv.org/abs/2001.04643">arXiv:2001.04643</a>.
 */
public final class HarmonicNoiseSynth {

    private final int rate;
    private final int hop;
    private final int frames;
    private final int harmonics;
    private final int bands;
    private final Tensor interpolation;
    private final float[][] noiseBank;
    private final Tensor noiseTensor;

    /**
     * Creates a renderer for clips of a fixed length.
     *
     * @param rate samples per second
     * @param hop samples per control frame
     * @param frames control frames per clip (the clip has {@code frames * hop} samples)
     * @param harmonics harmonic oscillators
     * @param bands noise bands, linearly spaced up to Nyquist
     * @param seed draws the noise bank
     */
    public HarmonicNoiseSynth(int rate, int hop, int frames, int harmonics, int bands, long seed) {
        this.rate = rate;
        this.hop = hop;
        this.frames = frames;
        this.harmonics = harmonics;
        this.bands = bands;
        int samples = frames * hop;
        float[] m = new float[frames * samples];
        for (int n = 0; n < samples; n++) {
            double pos = (n + 0.5) / hop - 0.5;
            int lo = (int) Math.floor(pos);
            double frac = pos - lo;
            int a = Math.max(0, Math.min(frames - 1, lo));
            int b = Math.max(0, Math.min(frames - 1, lo + 1));
            m[a * samples + n] += (float) (1 - frac);
            m[b * samples + n] += (float) frac;
        }
        interpolation = Tensor.of(Shape.of(frames, samples), m);
        noiseBank = new float[bands][samples];
        SplittableRandom r = new SplittableRandom(seed);
        double[] white = new double[samples];
        for (int n = 0; n < samples; n++) {
            white[n] = r.nextGaussian();
        }
        int bins = samples / 2 + 1;
        double[] re = new double[bins];
        double[] im = new double[bins];
        for (int k = 0; k < bins; k++) {
            for (int n = 0; n < samples; n++) {
                double ang = -2 * Math.PI * k * n / samples;
                re[k] += white[n] * Math.cos(ang);
                im[k] += white[n] * Math.sin(ang);
            }
        }
        for (int band = 0; band < bands; band++) {
            int k0 = band * (bins - 1) / bands;
            int k1 = (band + 1) * (bins - 1) / bands;
            for (int n = 0; n < samples; n++) {
                double v = 0;
                for (int k = Math.max(1, k0); k < k1; k++) {
                    double ang = 2 * Math.PI * k * n / samples;
                    v += 2 * (re[k] * Math.cos(ang) - im[k] * Math.sin(ang));
                }
                noiseBank[band][n] = (float) (v / samples * Math.sqrt((double) bands));
            }
        }
        float[] bank = new float[bands * samples];
        for (int k = 0; k < bands; k++) {
            System.arraycopy(noiseBank[k], 0, bank, k * samples, samples);
        }
        noiseTensor = Tensor.of(Shape.of(1, bands, samples), bank);
    }

    /** @return samples per clip */
    public int samples() {
        return frames * hop;
    }

    /**
     * The oscillator bank for given f0 tracks: {@code sin(k phi(n))} with the phase the
     * running sum of the interpolated f0, zero where {@code k f0} reaches Nyquist.
     *
     * @param f0 per clip its f0 per frame in Hz ({@code 0} for silence)
     * @return {@code [b, harmonics, samples]}
     */
    public Tensor oscillators(float[][] f0) {
        int samples = samples();
        float[] interp = interpolation.toArray();
        float[] out = new float[f0.length * harmonics * samples];
        for (int i = 0; i < f0.length; i++) {
            double phase = 0;
            for (int n = 0; n < samples; n++) {
                double f = 0;
                for (int t = 0; t < frames; t++) {
                    f += interp[t * samples + n] * f0[i][t];
                }
                phase += 2 * Math.PI * f / rate;
                for (int k = 1; k <= harmonics; k++) {
                    if (f > 0 && k * f < rate / 2.0) {
                        out[(i * harmonics + k - 1) * samples + n] = (float) Math.sin(k * phase);
                    }
                }
            }
        }
        return Tensor.of(Shape.of(f0.length, harmonics, samples), out);
    }

    /**
     * Renders audio.
     *
     * @param oscillators {@link #oscillators} of the clips' f0 tracks
     * @param amplitudes {@code [b, frames, harmonics]}, non-negative
     * @param noise {@code [b, frames, bands]}, non-negative
     * @return {@code [b, samples]}
     */
    public Var render(Tensor oscillators, Var amplitudes, Var noise) {
        int b = amplitudes.shape().dim(0);
        Var interp = Var.of(interpolation);
        Var amp = amplitudes.permute(0, 2, 1).matmul(interp);                 // [b, H, S]
        Var harmonic = amp.mul(oscillators).sum(1, false);
        Var mags = noise.permute(0, 2, 1).matmul(interp);                     // [b, N, S]
        Var filtered = mags.mul(noiseTensor).sum(1, false);
        return harmonic.add(filtered).reshape(b, samples());
    }

    /**
     * A pitch track by normalised autocorrelation over a window of {@code 2 hop} samples
     * centred on each frame, with parabolic peak refinement; unvoiced (correlation below
     * 0.5 or silent) frames get 0.
     *
     * @param wave the samples
     * @param rate samples per second
     * @param hop samples per frame
     * @param frames frames to produce
     * @param fmin lowest f0
     * @param fmax highest f0
     * @return f0 per frame in Hz
     */
    public static float[] estimateF0(float[] wave, int rate, int hop, int frames, double fmin, double fmax) {
        float[] f0 = new float[frames];
        int win = 2 * hop;
        int minLag = (int) Math.floor(rate / fmax);
        int maxLag = (int) Math.ceil(rate / fmin);
        for (int t = 0; t < frames; t++) {
            int start = t * hop + hop / 2 - win / 2;
            double energy = 0;
            for (int n = 0; n < win; n++) {
                double v = at(wave, start + n);
                energy += v * v;
            }
            if (energy < 1e-4 * win) {
                continue;
            }
            double[] corr = new double[maxLag + 2];
            for (int lag = minLag - 1; lag <= maxLag + 1; lag++) {
                double s = 0;
                double e2 = 0;
                for (int n = 0; n < win; n++) {
                    double b = at(wave, start + n + lag);
                    s += at(wave, start + n) * b;
                    e2 += b * b;
                }
                corr[Math.min(maxLag + 1, lag)] = s / Math.sqrt(energy * e2 + 1e-12);
            }
            int best = minLag;
            for (int lag = minLag; lag <= maxLag; lag++) {
                if (corr[lag] > corr[best]) {
                    best = lag;
                }
            }
            if (corr[best] < 0.5) {
                continue;
            }
            double a = corr[best - 1];
            double b = corr[best];
            double c = corr[best + 1];
            double denom = a - 2 * b + c;
            double shift = Math.abs(denom) < 1e-12 ? 0 : 0.5 * (a - c) / denom;
            f0[t] = (float) (rate / (best + Math.max(-0.5, Math.min(0.5, shift))));
        }
        return f0;
    }

    private static double at(float[] w, int i) {
        return i < 0 || i >= w.length ? 0 : w[i];
    }

    /**
     * Frame loudness: RMS in dB over each frame, floored at -80 dB.
     *
     * @param wave the samples
     * @param hop samples per frame
     * @param frames frames to produce
     * @return dB per frame
     */
    public static float[] loudness(float[] wave, int hop, int frames) {
        float[] out = new float[frames];
        for (int t = 0; t < frames; t++) {
            double e = 0;
            for (int n = 0; n < hop; n++) {
                double v = at(wave, t * hop + n);
                e += v * v;
            }
            out[t] = (float) Math.max(-80, 10 * Math.log10(e / hop + 1e-12));
        }
        return out;
    }
}
