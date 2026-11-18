package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Mel-frequency cepstral coefficients: the log-mel energies of core {@link Signal#logMel},
 * decorrelated by an orthonormal DCT-II, keeping the first coefficients (the smooth
 * spectral envelope, which carries the formants, without the pitch harmonics), optionally
 * with their deltas (a regression over {@code +-2} frames) for the dynamics. The classic
 * speech front end for template matching and GMM-HMM recognisers.
 *
 * <p>Paper: Davis and Mermelstein, "Comparison of parametric representations for
 * monosyllabic word recognition in continuously spoken sentences" (IEEE TASSP 1980),
 * <a href="https://ieeexplore.ieee.org/document/1163420">doi</a>.
 */
public final class Mfcc {

    private Mfcc() {
    }

    /**
     * The orthonormal DCT-II matrix that maps {@code mels} log energies to {@code coefficients}
     * cepstra: {@code C[m][c] = s_c cos(pi c (m + 1/2) / mels)}.
     *
     * @param mels input bands
     * @param coefficients outputs kept
     * @return {@code [mels, coefficients]}
     */
    public static Tensor dctMatrix(int mels, int coefficients) {
        float[] d = new float[mels * coefficients];
        for (int m = 0; m < mels; m++) {
            for (int c = 0; c < coefficients; c++) {
                double s = c == 0 ? Math.sqrt(1.0 / mels) : Math.sqrt(2.0 / mels);
                d[m * coefficients + c] = (float) (s * Math.cos(Math.PI * c * (m + 0.5) / mels));
            }
        }
        return Tensor.of(Shape.of(mels, coefficients), d);
    }

    /**
     * Differentiable MFCCs of log-mel spectrograms.
     *
     * @param logMel {@code [..., frames, mels]}
     * @param coefficients cepstra kept, such as 13
     * @return {@code [..., frames, coefficients]}
     */
    public static Var of(Var logMel, int coefficients) {
        return logMel.matmul(Var.of(dctMatrix(logMel.shape().dim(-1), coefficients)));
    }

    /**
     * MFCCs (and deltas) of one waveform.
     *
     * @param wave the samples
     * @param sampleRate their rate
     * @param nFft the frame length
     * @param hop the step between frames
     * @param mels mel bands
     * @param coefficients cepstra kept
     * @param deltas whether to append the deltas
     * @return {@code [frames][coefficients]} or {@code [frames][2 * coefficients]}
     */
    public static float[][] compute(float[] wave, int sampleRate, int nFft, int hop, int mels, int coefficients,
            boolean deltas) {
        Var lm = Signal.logMel(Var.of(Tensor.of(Shape.of(wave.length), wave)), sampleRate, nFft, hop, mels);
        float[] c = of(lm, coefficients).value().toArray();
        int frames = c.length / coefficients;
        float[][] out = new float[frames][deltas ? 2 * coefficients : coefficients];
        for (int t = 0; t < frames; t++) {
            System.arraycopy(c, t * coefficients, out[t], 0, coefficients);
        }
        if (deltas) {
            for (int t = 0; t < frames; t++) {
                for (int k = 0; k < coefficients; k++) {
                    double num = 0;
                    for (int n = 1; n <= 2; n++) {
                        int a = Math.min(frames - 1, t + n);
                        int b = Math.max(0, t - n);
                        num += n * (out[a][k] - out[b][k]);
                    }
                    out[t][coefficients + k] = (float) (num / 10);   // 2 * (1 + 4)
                }
            }
        }
        return out;
    }
}
