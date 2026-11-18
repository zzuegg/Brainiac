package io.github.zzuegg.brainiac.experimental.music;

/**
 * Chroma features (pitch-class profiles): the spectrum read on a constant-Q-style grid of
 * three bins per semitone from C2 (65.4 Hz) over five octaves, each semitone the weighted sum
 * of its three bins, folded onto the 12 pitch classes, scaled to a maximum of one per frame and
 * log-compressed as {@code log(1 + gamma c) / log(1 + gamma)}. Octave and most of timbre are
 * gone; harmony stays.
 *
 * <p>Paper: Fujishima, "Realtime Chord Recognition of Musical Sound: a System Using Common
 * Lisp Music", ICMC 1999.
 */
public final class Chroma {

    /** The FFT size. */
    public static final int N_FFT = 2048;
    /** The lowest frequency (C2). */
    public static final double FMIN = 65.406;

    private Chroma() {
    }

    /**
     * Chroma and frame energy.
     *
     * @param x the signal
     * @param rate the sample rate
     * @param hop the hop
     * @param gamma the log compression ({@code 0} for none)
     * @param energy receives the per-frame total (may be {@code null})
     * @return {@code [x.length / hop + 1][12]}, pitch class 0 = C
     */
    public static float[][] compute(float[] x, int rate, int hop, double gamma, float[] energy) {
        float[][] mag = MusicFeatures.magnitude(x, N_FFT, hop);
        float[][] lf = MusicFeatures.logFrequency(mag, rate, FMIN * Math.pow(2, -1 / 36.0), 36, 36 * 5, false);
        float[][] out = new float[lf.length][12];
        for (int t = 0; t < lf.length; t++) {
            double total = 0;
            for (int s = 0; s < 60; s++) {
                double v = 0.5 * lf[t][3 * s] + lf[t][3 * s + 1] + 0.5 * lf[t][3 * s + 2];
                out[t][s % 12] += (float) v;
                total += v;
            }
            if (energy != null && t < energy.length) {
                energy[t] = (float) total;
            }
            float max = 1e-6f;
            for (int c = 0; c < 12; c++) {
                max = Math.max(max, out[t][c]);
            }
            for (int c = 0; c < 12; c++) {
                out[t][c] /= max;
                if (gamma > 0) {
                    out[t][c] = (float) (Math.log1p(gamma * out[t][c]) / Math.log1p(gamma));
                }
            }
        }
        return out;
    }

    /**
     * Rotates a chroma vector so that pitch class {@code root} becomes C.
     *
     * @param chroma a 12-vector
     * @param root the root
     * @return the rotated copy
     */
    public static float[] rotate(float[] chroma, int root) {
        float[] r = new float[12];
        for (int c = 0; c < 12; c++) {
            r[c] = chroma[(c + root) % 12];
        }
        return r;
    }
}
