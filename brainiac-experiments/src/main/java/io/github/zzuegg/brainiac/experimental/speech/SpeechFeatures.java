package io.github.zzuegg.brainiac.experimental.speech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Frame features shared by the speech blocks, on core {@link Signal}: log-mel spectrograms
 * with one frame per {@code hop} samples, per-utterance mean normalisation, and batching of
 * feature matrices.
 */
public final class SpeechFeatures {

    private SpeechFeatures() {
    }

    /**
     * Log-mel features of a waveform, one frame per {@code hop} samples (the STFT's centred
     * frame count minus one, so that {@code samples / hop} frames remain).
     *
     * @param wave the samples
     * @param rate the sample rate
     * @param hop the hop (also half the FFT size)
     * @param mels mel bands
     * @return {@code [frames][mels]}
     */
    public static float[][] logMel(float[] wave, int rate, int hop, int mels) {
        float[] v = Signal.logMel(Var.of(Tensor.of(Shape.of(wave.length), wave)), rate, 2 * hop, hop, mels)
                .value().toArray();
        int frames = wave.length / hop;
        float[][] out = new float[frames][mels];
        for (int t = 0; t < frames; t++) {
            System.arraycopy(v, t * mels, out[t], 0, mels);
        }
        return out;
    }

    /**
     * Subtracts the per-band mean over time (cepstral mean normalisation) and scales by 1/4.
     *
     * @param features {@code [frames][bands]}
     * @return a normalised copy
     */
    public static float[][] normalise(float[][] features) {
        int bands = features[0].length;
        double[] mean = new double[bands];
        for (float[] f : features) {
            for (int b = 0; b < bands; b++) {
                mean[b] += f[b] / (double) features.length;
            }
        }
        float[][] out = new float[features.length][bands];
        for (int t = 0; t < features.length; t++) {
            for (int b = 0; b < bands; b++) {
                out[t][b] = (float) ((features[t][b] - mean[b]) / 4);
            }
        }
        return out;
    }

    /**
     * A window of frames, padded by repeating the last frame when too short.
     *
     * @param features {@code [frames][bands]}
     * @param start the first frame
     * @param length the window length
     * @return {@code [length][bands]}
     */
    public static float[][] window(float[][] features, int start, int length) {
        float[][] out = new float[length][];
        for (int t = 0; t < length; t++) {
            int s = Math.min(features.length - 1, Math.max(0, start + t));
            out[t] = features[s];
        }
        return out;
    }

    /**
     * Stacks equal-length feature matrices into a batch.
     *
     * @param items the matrices, all {@code [frames][bands]}
     * @return {@code [n, frames, bands]}
     */
    public static Tensor batch(List<float[][]> items) {
        int t = items.get(0).length;
        int b = items.get(0)[0].length;
        float[] x = new float[items.size() * t * b];
        for (int i = 0; i < items.size(); i++) {
            for (int f = 0; f < t; f++) {
                System.arraycopy(items.get(i)[f], 0, x, (i * t + f) * b, b);
            }
        }
        return Tensor.of(Shape.of(items.size(), t, b), x);
    }

    /**
     * The mean feature vector of an utterance.
     *
     * @param features {@code [frames][bands]}
     * @return the mean over frames
     */
    public static double[] mean(float[][] features) {
        double[] m = new double[features[0].length];
        for (float[] f : features) {
            for (int b = 0; b < m.length; b++) {
                m[b] += f[b] / (double) features.length;
            }
        }
        return m;
    }

    /**
     * Cosine similarity.
     *
     * @param a one vector
     * @param b another
     * @return the cosine of their angle
     */
    public static double cosine(double[] a, double[] b) {
        double d = 0;
        double na = 1e-12;
        double nb = 1e-12;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return d / Math.sqrt(na * nb);
    }

    /**
     * Copies a float vector to doubles.
     *
     * @param v the vector
     * @return its double copy
     */
    public static double[] toDouble(float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i];
        }
        return d;
    }

    /**
     * Copies a matrix.
     *
     * @param m the matrix
     * @return a deep copy
     */
    static float[][] copy(float[][] m) {
        float[][] c = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            c[i] = Arrays.copyOf(m[i], m[i].length);
        }
        return c;
    }
}
