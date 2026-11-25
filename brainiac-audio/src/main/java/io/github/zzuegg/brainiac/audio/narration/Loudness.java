package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;

/**
 * Loudness as ITU-R BS.1770 measures it (the LUFS of EBU R128 and the streaming services):
 * K-weighting, 400 ms blocks overlapping by 75%, an absolute gate at -70 LUFS and a relative
 * gate 10 LU below. A narrator's lines all normalized to the same loudness sound equally
 * loud, which matching peaks does not achieve.
 */
public final class Loudness {

    private Loudness() {
    }

    /**
     * The integrated loudness of mono audio.
     *
     * @param audio the audio
     * @return LUFS, or {@link Double#NEGATIVE_INFINITY} for silence
     */
    public static double lufs(Audio audio) {
        float[] k = kWeighted(audio.samples(), audio.sampleRate());
        int rate = audio.sampleRate();
        int block = (int) Math.round(0.4 * rate);
        int step = (int) Math.round(0.1 * rate);
        int count = k.length < block ? 1 : 1 + (k.length - block) / step;
        double[] z = new double[count];
        for (int b = 0; b < count; b++) {
            int from = b * step;
            int to = Math.min(k.length, from + block);
            double sum = 0;
            for (int i = from; i < to; i++) {
                sum += (double) k[i] * k[i];
            }
            z[b] = to > from ? sum / (to - from) : 0;
        }
        double absGate = Math.pow(10, (-70 + 0.691) / 10);
        double mean = gatedMean(z, absGate);
        if (mean <= 0) {
            return Double.NEGATIVE_INFINITY;
        }
        double relGate = mean * Math.pow(10, -10 / 10.0);
        double gated = gatedMean(z, Math.max(absGate, relGate));
        return -0.691 + 10 * Math.log10(gated);
    }

    private static double gatedMean(double[] z, double gate) {
        double sum = 0;
        int n = 0;
        for (double v : z) {
            if (v > gate) {
                sum += v;
                n++;
            }
        }
        return n == 0 ? 0 : sum / n;
    }

    /**
     * Scales audio to a loudness, then lowers it if a peak would pass {@code ceiling}.
     *
     * @param audio the audio
     * @param targetLufs the loudness, such as -16
     * @param ceilingDb the highest peak, in dB of full scale, such as -1
     * @return scaled audio (the same audio if it is silent)
     */
    public static Audio normalize(Audio audio, double targetLufs, double ceilingDb) {
        double now = lufs(audio);
        if (Double.isInfinite(now)) {
            return audio;
        }
        return gain(audio, targetLufs - now, ceilingDb);
    }

    /**
     * Scales audio by a gain in dB, then lowers it if a peak would pass {@code ceilingDb}.
     *
     * @param audio the audio
     * @param db the gain
     * @param ceilingDb the highest peak, in dB of full scale
     * @return scaled audio
     */
    public static Audio gain(Audio audio, double db, double ceilingDb) {
        float[] x = audio.samples();
        double g = Math.pow(10, db / 20);
        double peak = 0;
        for (float v : x) {
            peak = Math.max(peak, Math.abs(v));
        }
        double ceiling = Math.pow(10, ceilingDb / 20);
        if (peak * g > ceiling && peak > 0) {
            g = ceiling / peak;
        }
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) (x[i] * g);
        }
        return new Audio(out, audio.sampleRate());
    }

    /** The two K-weighting filters (a high shelf, then a high pass), designed for the sample rate. */
    static float[] kWeighted(float[] x, int rate) {
        double a = Math.pow(10, 3.999843853973347 / 40);
        double w = 2 * Math.PI * 1681.974450955533 / rate;
        double alpha = Math.sin(w) / (2 * 0.7071752369554196);
        double cos = Math.cos(w);
        double sq = 2 * Math.sqrt(a) * alpha;
        double[] shelf = biquad(a * ((a + 1) + (a - 1) * cos + sq), -2 * a * ((a - 1) + (a + 1) * cos),
                a * ((a + 1) + (a - 1) * cos - sq), (a + 1) - (a - 1) * cos + sq, 2 * ((a - 1) - (a + 1) * cos),
                (a + 1) - (a - 1) * cos - sq);
        w = 2 * Math.PI * 38.13547087602444 / rate;
        alpha = Math.sin(w) / (2 * 0.5003270373238773);
        cos = Math.cos(w);
        double[] pass = biquad((1 + cos) / 2, -(1 + cos), (1 + cos) / 2, 1 + alpha, -2 * cos, 1 - alpha);
        return filter(filter(x, shelf), pass);
    }

    private static double[] biquad(double b0, double b1, double b2, double a0, double a1, double a2) {
        return new double[] {b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0};
    }

    private static float[] filter(float[] x, double[] c) {
        float[] y = new float[x.length];
        double x1 = 0;
        double x2 = 0;
        double y1 = 0;
        double y2 = 0;
        for (int i = 0; i < x.length; i++) {
            double v = c[0] * x[i] + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2;
            x2 = x1;
            x1 = x[i];
            y2 = y1;
            y1 = v;
            y[i] = (float) v;
        }
        return y;
    }
}
