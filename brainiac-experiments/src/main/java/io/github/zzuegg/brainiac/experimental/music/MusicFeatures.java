package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The time-frequency front ends that the music blocks share, on plain arrays: the core
 * {@link Signal#stft} magnitude (centred frames, so frame {@code t} is centred on sample
 * {@code t hop}), a constant-Q-style log-frequency spectrum (the magnitude spectrum read by
 * interpolation on a geometric frequency grid with a fixed number of bins per octave), the
 * core log-mel spectrogram, and per-frame energy.
 */
public final class MusicFeatures {

    private MusicFeatures() {
    }

    /**
     * The STFT magnitude.
     *
     * @param x the signal
     * @param nFft the frame length (a power of two)
     * @param hop the hop
     * @return {@code [x.length / hop + 1][nFft / 2 + 1]}
     */
    public static float[][] magnitude(float[] x, int nFft, int hop) {
        Var spec = Var.noGrad(() -> Signal.stft(Var.of(Tensor.of(Shape.of(x.length), x)), nFft, hop));
        int frames = spec.shape().dim(0);
        int bins = spec.shape().dim(1);
        float[] v = spec.value().toArray();
        float[][] out = new float[frames][bins];
        for (int t = 0; t < frames; t++) {
            for (int k = 0; k < bins; k++) {
                float re = v[(t * bins + k) * 2];
                float im = v[(t * bins + k) * 2 + 1];
                out[t][k] = (float) Math.sqrt(re * re + im * im);
            }
        }
        return out;
    }

    /**
     * The centre frequencies of a geometric grid.
     *
     * @param fmin the lowest frequency
     * @param binsPerOctave bins per octave
     * @param bins how many
     * @return the frequencies in Hz
     */
    public static double[] logGrid(double fmin, int binsPerOctave, int bins) {
        double[] f = new double[bins];
        for (int i = 0; i < bins; i++) {
            f[i] = fmin * Math.pow(2, i / (double) binsPerOctave);
        }
        return f;
    }

    /**
     * A constant-Q-style spectrum: each STFT frame read by linear interpolation at the
     * frequencies of a geometric grid (a peak-preserving maximum over the linear bins a log
     * bin covers above the crossover where log bins become wider than linear ones).
     *
     * @param mag the STFT magnitude {@code [frames][nFft / 2 + 1]}
     * @param rate the sample rate
     * @param fmin the lowest frequency
     * @param binsPerOctave bins per octave
     * @param bins how many bins
     * @param logCompress whether to return {@code log(1 + 10 m)}
     * @return {@code [frames][bins]}
     */
    public static float[][] logFrequency(float[][] mag, int rate, double fmin, int binsPerOctave, int bins,
            boolean logCompress) {
        int nBins = mag[0].length;
        double binHz = rate / 2.0 / (nBins - 1);
        double[] grid = logGrid(fmin, binsPerOctave, bins);
        float[][] out = new float[mag.length][bins];
        double ratio = Math.pow(2, 0.5 / binsPerOctave);
        for (int t = 0; t < mag.length; t++) {
            for (int i = 0; i < bins; i++) {
                double lo = grid[i] / ratio / binHz;
                double hi = grid[i] * ratio / binHz;
                double v;
                if (hi - lo <= 1) {
                    double p = grid[i] / binHz;
                    int a = (int) Math.floor(p);
                    double f = p - a;
                    v = a + 1 < nBins ? mag[t][a] * (1 - f) + mag[t][a + 1] * f : 0;
                } else {
                    v = 0;
                    for (int k = (int) Math.ceil(lo); k <= Math.min(nBins - 1, (int) Math.floor(hi)); k++) {
                        v = Math.max(v, mag[t][k]);
                    }
                }
                out[t][i] = (float) (logCompress ? Math.log1p(10 * v) : v);
            }
        }
        return out;
    }

    /**
     * The core log-mel spectrogram.
     *
     * @param x the signal
     * @param rate the sample rate
     * @param nFft the frame length
     * @param hop the hop
     * @param mels mel bands
     * @return {@code [frames][mels]}
     */
    public static float[][] logMel(float[] x, int rate, int nFft, int hop, int mels) {
        Var m = Var.noGrad(() -> Signal.logMel(Var.of(Tensor.of(Shape.of(x.length), x)), rate, nFft, hop, mels));
        int frames = m.shape().dim(0);
        float[] v = m.value().toArray();
        float[][] out = new float[frames][mels];
        for (int t = 0; t < frames; t++) {
            System.arraycopy(v, t * mels, out[t], 0, mels);
        }
        return out;
    }

    /**
     * Root-mean-square energy of centred frames.
     *
     * @param x the signal
     * @param frameLength the frame length
     * @param hop the hop
     * @return {@code [x.length / hop + 1]}
     */
    public static float[] energy(float[] x, int frameLength, int hop) {
        int frames = x.length / hop + 1;
        float[] e = new float[frames];
        for (int t = 0; t < frames; t++) {
            double s = 0;
            int c = t * hop;
            for (int j = -frameLength / 2; j < frameLength / 2; j++) {
                int n = c + j;
                if (n >= 0 && n < x.length) {
                    s += x[n] * (double) x[n];
                }
            }
            e[t] = (float) Math.sqrt(s / frameLength);
        }
        return e;
    }

    /**
     * A centred frame, zero outside the signal.
     *
     * @param x the signal
     * @param centre the centre sample
     * @param length the frame length
     * @return the frame
     */
    public static float[] frame(float[] x, int centre, int length) {
        float[] f = new float[length];
        int start = centre - length / 2;
        for (int j = 0; j < length; j++) {
            int n = start + j;
            if (n >= 0 && n < x.length) {
                f[j] = x[n];
            }
        }
        return f;
    }

    /**
     * Stacks rows into a tensor.
     *
     * @param rows equal-length rows
     * @return {@code [rows, cols]}
     */
    public static Tensor stack(float[][] rows) {
        int c = rows[0].length;
        float[] v = new float[rows.length * c];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, v, i * c, c);
        }
        return Tensor.of(Shape.of(rows.length, c), v);
    }
}
