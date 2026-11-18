package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Griffin-Lim phase reconstruction (Griffin and Lim, 1984), the classical vocoder baseline:
 * start from random phases, then repeat {@code x = iSTFT(|S| e^{i phi})}, {@code phi =
 * angle(STFT(x))}, so the phases become consistent with the target magnitudes. With
 * {@code momentum > 0} it is the "fast Griffin-Lim" of Perraudin et al. (2013), as torchaudio
 * implements it. Runs on the core {@link Signal} STFT and iSTFT.
 */
public final class GriffinLim {

    private final int nFft;
    private final int hop;
    private final int iterations;
    private final float momentum;
    private final SplittableRandom random;

    /**
     * Creates the reconstructor.
     *
     * @param nFft the STFT frame length
     * @param hop the step between frames
     * @param iterations the iterations
     * @param momentum the fast Griffin-Lim momentum (0 for the classical algorithm, 0.99 usual)
     * @param seed the seed of the initial phases
     */
    public GriffinLim(int nFft, int hop, int iterations, float momentum, long seed) {
        this.nFft = nFft;
        this.hop = hop;
        this.iterations = iterations;
        this.momentum = momentum;
        this.random = new SplittableRandom(seed);
    }

    private Tensor stft(float[] x) {
        return Var.noGrad(() -> Signal.stft(Var.of(Tensor.of(Shape.of(x.length), x)), nFft, hop)).value();
    }

    private float[] istft(float[] spec, int frames, int samples) {
        int bins = nFft / 2 + 1;
        return Var.noGrad(() -> Signal.istft(Var.of(Tensor.of(Shape.of(frames, bins, 2), spec)), nFft, hop, samples))
                .value().toArray();
    }

    /**
     * Reconstructs a waveform from linear magnitudes.
     *
     * @param magnitude {@code [frames][nFft / 2 + 1]}; frames should be {@code samples / hop + 1}
     * @param samples the length of the waveform
     * @return the waveform
     */
    public float[] invert(float[][] magnitude, int samples) {
        int frames = samples / hop + 1;
        int bins = nFft / 2 + 1;
        float[] mag = new float[frames * bins];
        for (int f = 0; f < frames; f++) {
            float[] row = magnitude[Math.min(f, magnitude.length - 1)];
            System.arraycopy(row, 0, mag, f * bins, bins);
        }
        float[] spec = new float[frames * bins * 2];
        for (int i = 0; i < mag.length; i++) {
            double a = random.nextDouble() * 2 * Math.PI;
            spec[2 * i] = (float) (mag[i] * Math.cos(a));
            spec[2 * i + 1] = (float) (mag[i] * Math.sin(a));
        }
        float[] previous = new float[spec.length];
        float[] x = istft(spec, frames, samples);
        float alpha = momentum / (1 + momentum);
        for (int it = 0; it < iterations; it++) {
            float[] rebuilt = stft(x).toArray();
            for (int i = 0; i < mag.length; i++) {
                double re = rebuilt[2 * i] - alpha * previous[2 * i];
                double im = rebuilt[2 * i + 1] - alpha * previous[2 * i + 1];
                double norm = Math.hypot(re, im);
                if (norm < 1e-12) {
                    re = 1;
                    im = 0;
                    norm = 1;
                }
                spec[2 * i] = (float) (mag[i] * re / norm);
                spec[2 * i + 1] = (float) (mag[i] * im / norm);
            }
            previous = rebuilt;
            x = istft(spec, frames, samples);
        }
        return x;
    }

    /**
     * Vocodes log-mels: approximate magnitudes via {@link MelFeatures#magnitude}, then
     * {@link #invert}.
     *
     * @param mel the front end that made the log-mels (its {@code nFft} and hop must match)
     * @param logMel {@code [frames][mels]}
     * @param samples the length of the waveform
     * @return the waveform
     */
    public float[] vocode(MelFeatures mel, float[][] logMel, int samples) {
        return invert(mel.magnitude(logMel), samples);
    }
}
