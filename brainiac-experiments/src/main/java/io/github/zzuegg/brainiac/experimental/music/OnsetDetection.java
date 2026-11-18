package io.github.zzuegg.brainiac.experimental.music;

/**
 * Onset detection functions and adaptive peak picking.
 *
 * <ul>
 *   <li>{@link Function#ENERGY}: the half-wave rectified difference of log frame energy (the
 *       baseline, deaf to onsets that change the spectrum but not the loudness);
 *   <li>{@link Function#SPECTRAL_FLUX}: the sum over bins of the positive differences of the
 *       log-compressed magnitude spectrum between consecutive frames (Bello et al.);
 *   <li>{@link Function#SUPERFLUX}: spectral flux on a log-frequency spectrum (24 bins per
 *       octave) against a frequency-maximum-filtered frame two hops back, which suppresses
 *       false onsets from vibrato and slow pitch glides (Boeck and Widmer).
 * </ul>
 *
 * <p>{@link #onsets} normalises the function to zero mean and unit variance and picks local
 * maxima above the moving mean plus a threshold ({@link MusicMetrics#pickPeaks}).
 *
 * <p>Papers: Bello, Daudet, Abdallah, Duxbury, Davies, Sandler, "A Tutorial on Onset Detection
 * in Music Signals", IEEE TSAP 13(5), 2005,
 * <a href="https://doi.org/10.1109/TSA.2005.851998">doi:10.1109/TSA.2005.851998</a>; Boeck,
 * Widmer, "Maximum Filter Vibrato Suppression for Onset Detection", DAFx 2013.
 */
public final class OnsetDetection {

    /** Detection functions. */
    public enum Function {
        /** Log-energy difference. */
        ENERGY,
        /** Log-magnitude spectral flux. */
        SPECTRAL_FLUX,
        /** Maximum-filtered log-frequency flux. */
        SUPERFLUX
    }

    /** The STFT size. */
    public static final int N_FFT = 512;

    private OnsetDetection() {
    }

    /**
     * A detection function.
     *
     * @param f the function
     * @param x the signal
     * @param rate the sample rate
     * @param hop the hop
     * @return one value per frame ({@code x.length / hop + 1})
     */
    public static float[] function(Function f, float[] x, int rate, int hop) {
        int frames = x.length / hop + 1;
        float[] out = new float[frames];
        switch (f) {
            case ENERGY -> {
                float[] e = MusicFeatures.energy(x, N_FFT, hop);
                for (int t = 1; t < frames; t++) {
                    out[t] = (float) Math.max(0, Math.log(1e-4 + e[t]) - Math.log(1e-4 + e[t - 1]));
                }
            }
            case SPECTRAL_FLUX -> {
                float[][] m = MusicFeatures.magnitude(x, N_FFT, hop);
                for (int t = 1; t < frames; t++) {
                    double s = 0;
                    for (int k = 0; k < m[t].length; k++) {
                        s += Math.max(0, Math.log1p(10 * m[t][k]) - Math.log1p(10 * m[t - 1][k]));
                    }
                    out[t] = (float) s;
                }
            }
            case SUPERFLUX -> {
                float[][] m = MusicFeatures.logFrequency(MusicFeatures.magnitude(x, 2 * N_FFT, hop), rate, 50, 24,
                        24 * 6, true);
                for (int t = 2; t < frames; t++) {
                    double s = 0;
                    for (int k = 0; k < m[t].length; k++) {
                        float ref = m[t - 2][k];
                        if (k > 0) {
                            ref = Math.max(ref, m[t - 2][k - 1]);
                        }
                        if (k + 1 < m[t].length) {
                            ref = Math.max(ref, m[t - 2][k + 1]);
                        }
                        s += Math.max(0, m[t][k] - ref);
                    }
                    out[t] = (float) s;
                }
            }
            default -> throw new IllegalArgumentException(f.toString());
        }
        return out;
    }

    /**
     * Standardises a detection function to zero mean and unit variance.
     *
     * @param f the function
     * @return a standardised copy
     */
    public static float[] standardise(float[] f) {
        double mean = 0;
        for (float v : f) {
            mean += v;
        }
        mean /= f.length;
        double var = 0;
        for (float v : f) {
            var += (v - mean) * (v - mean);
        }
        double std = Math.sqrt(var / f.length) + 1e-9;
        float[] out = new float[f.length];
        for (int i = 0; i < f.length; i++) {
            out[i] = (float) ((f[i] - mean) / std);
        }
        return out;
    }

    /**
     * Onset times.
     *
     * @param f the function
     * @param x the signal
     * @param rate the sample rate
     * @param hop the hop
     * @param delta the threshold above the moving mean, in standard deviations
     * @return onset times in seconds
     */
    public static double[] onsets(Function f, float[] x, int rate, int hop, double delta) {
        float[] d = standardise(function(f, x, rate, hop));
        int[] peaks = MusicMetrics.pickPeaks(d, 3, 10, delta, 3);
        double[] t = new double[peaks.length];
        for (int i = 0; i < peaks.length; i++) {
            t[i] = peaks[i] * hop / (double) rate;
        }
        return t;
    }
}
