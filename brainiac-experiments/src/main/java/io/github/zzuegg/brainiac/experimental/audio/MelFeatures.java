package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The log-mel front end shared by the audio generators: {@code log(power(STFT) x filterbank +
 * 1e-5)} on the core {@link Signal} STFT (Hann window, centred frames, frame {@code t} at sample
 * {@code t * hop}). It is differentiable, so it doubles as the mel-reconstruction loss of a
 * vocoder, and it has an approximate inverse to linear magnitudes for {@link GriffinLim}. Also
 * carries the multi-resolution STFT distance used to score vocoders.
 */
public final class MelFeatures {

    private final int rate;
    private final int nFft;
    private final int hop;
    private final int mels;
    private final Tensor filterbank;

    /**
     * Creates the front end.
     *
     * @param rate the sample rate
     * @param nFft the STFT frame length
     * @param hop the step between frames
     * @param mels the mel bands
     */
    public MelFeatures(int rate, int nFft, int hop, int mels) {
        this.rate = rate;
        this.nFft = nFft;
        this.hop = hop;
        this.mels = mels;
        this.filterbank = Signal.melFilterbank(rate, nFft, mels, 0, rate / 2.0);
    }

    /** @return the 8 kHz speech front end: 256-sample frames, hop 128, 40 mel bands */
    public static MelFeatures speech() {
        return new MelFeatures(8000, 256, 128, 40);
    }

    /** @return the sample rate */
    public int rate() {
        return rate;
    }

    /** @return the STFT frame length */
    public int nFft() {
        return nFft;
    }

    /** @return the hop */
    public int hop() {
        return hop;
    }

    /** @return the mel bands */
    public int mels() {
        return mels;
    }

    /** @return the linear frequency bins, {@code nFft / 2 + 1} */
    public int bins() {
        return nFft / 2 + 1;
    }

    /**
     * @param samples a signal length
     * @return the frames its STFT has
     */
    public int frames(int samples) {
        return samples / hop + 1;
    }

    /**
     * Differentiable log-mels.
     *
     * @param waves {@code [batch, samples]}
     * @return {@code [batch, frames, mels]}
     */
    public Var logMel(Var waves) {
        return Signal.power(Signal.stft(waves, nFft, hop)).matmul(Var.of(filterbank)).add(1e-5f).log();
    }

    /**
     * Log-mels of one waveform.
     *
     * @param wave the samples
     * @return {@code [frames][mels]}
     */
    public float[][] logMel(float[] wave) {
        Tensor t = Var.noGrad(() -> logMel(Var.of(Tensor.of(Shape.of(1, wave.length), wave)))).value();
        return AudioNets.rows(t.reshape(frames(wave.length), mels));
    }

    /**
     * Approximate linear magnitudes from log-mels: each bin's power is the filter-weighted mean
     * of the mel powers that cover it (the normalised transpose of the filterbank).
     *
     * @param logMel {@code [frames][mels]}
     * @return {@code [frames][bins]}
     */
    public float[][] magnitude(float[][] logMel) {
        float[] fb = filterbank.toArray();
        int bins = bins();
        float[][] out = new float[logMel.length][bins];
        for (int f = 0; f < logMel.length; f++) {
            for (int k = 0; k < bins; k++) {
                double num = 0;
                double den = 0;
                for (int m = 0; m < mels; m++) {
                    double w = fb[k * mels + m];
                    num += w * Math.max(0, Math.exp(logMel[f][m]) - 1e-5);
                    den += w;
                }
                out[f][k] = den > 1e-9 ? (float) Math.sqrt(num / den) : 0;
            }
        }
        return out;
    }

    private static float[] magnitudes(float[] x, int n, int hop) {
        Tensor spec = Var.noGrad(() -> Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(x.length), x)), n, hop)))
                .value();
        float[] p = spec.toArray();
        for (int i = 0; i < p.length; i++) {
            p[i] = (float) Math.sqrt(p[i]);
        }
        return p;
    }

    /**
     * The multi-resolution STFT distance (Yamamoto et al., Parallel WaveGAN): spectral
     * convergence plus mean absolute log-magnitude difference, averaged over frame lengths 64,
     * 128 and 256 (hop a quarter of each). Phase-blind, so it scores vocoders fairly.
     *
     * @param reference the target
     * @param estimate the estimate (cut or zero-padded to the reference length)
     * @return the distance, 0 for identical magnitudes
     */
    public static double stftDistance(float[] reference, float[] estimate) {
        float[] est = java.util.Arrays.copyOf(estimate, reference.length);
        double total = 0;
        int[] sizes = {64, 128, 256};
        for (int n : sizes) {
            float[] a = magnitudes(reference, n, n / 4);
            float[] b = magnitudes(est, n, n / 4);
            double diff = 0;
            double norm = 0;
            double log = 0;
            for (int i = 0; i < a.length; i++) {
                double e = a[i] - b[i];
                diff += e * e;
                norm += a[i] * (double) a[i];
                log += Math.abs(Math.log(a[i] + 1e-5) - Math.log(b[i] + 1e-5));
            }
            total += Math.sqrt(diff / Math.max(norm, 1e-12)) + log / a.length;
        }
        return total / sizes.length;
    }
}
