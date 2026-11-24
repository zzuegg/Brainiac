package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * The signal work behind cutting speech at pauses: a loudness envelope, silent stretches,
 * trimming and fades, comparing transcribed text, and a rough voice print.
 */
final class AudioTools {

    /** Envelope frames per second (10 ms hop). */
    static final int FRAME_RATE = 100;
    /** The level below which audio counts as silence, in dB below the loudest frame. */
    static final double SILENCE_DB = 40;

    private AudioTools() {
    }

    /**
     * The level of each 10 ms frame (RMS over 25 ms), in dB of full scale.
     *
     * @param x the samples
     * @param rate the sample rate
     * @return one level per frame
     */
    static double[] envelope(float[] x, int rate) {
        int hop = rate / FRAME_RATE;
        int win = (int) (rate * 0.025);
        int frames = Math.max(1, (x.length + hop - 1) / hop);
        double[] out = new double[frames];
        for (int f = 0; f < frames; f++) {
            int from = Math.max(0, f * hop + hop / 2 - win / 2);
            int to = Math.min(x.length, from + win);
            double sum = 0;
            for (int i = from; i < to; i++) {
                sum += (double) x[i] * x[i];
            }
            double rms = to > from ? Math.sqrt(sum / (to - from)) : 0;
            out[f] = 20 * Math.log10(rms + 1e-9);
        }
        return out;
    }

    /**
     * Where the speech starts and ends: the first and last frames within {@link #SILENCE_DB}
     * of the loudest.
     *
     * @param env the envelope
     * @return {@code {first, last}} frames, or {@code {0, frames - 1}} if all is quiet
     */
    static int[] speech(double[] env) {
        double peak = Arrays.stream(env).max().orElse(-180);
        double t = peak - SILENCE_DB;
        int a = 0;
        while (a < env.length && env[a] < t) {
            a++;
        }
        int b = env.length - 1;
        while (b > a && env[b] < t) {
            b--;
        }
        return a >= env.length ? new int[] {0, env.length - 1} : new int[] {a, b};
    }

    /**
     * The silent stretches between two frames: runs of at least {@code minFrames} frames more
     * than {@link #SILENCE_DB} - 5 dB below the loudest.
     *
     * @param env the envelope
     * @param from the first frame to look at
     * @param to the last frame to look at
     * @param minFrames the shortest run
     * @return {@code {first, after last}} of each run, in time order
     */
    static List<int[]> silences(double[] env, int from, int to, int minFrames) {
        double peak = Arrays.stream(env).max().orElse(-180);
        double t = peak - (SILENCE_DB - 5);
        List<int[]> out = new ArrayList<>();
        int i = Math.max(0, from);
        int end = Math.min(env.length - 1, to);
        while (i <= end) {
            if (env[i] >= t) {
                i++;
                continue;
            }
            int j = i;
            while (j <= end && env[j] < t) {
                j++;
            }
            if (j - i >= minFrames) {
                out.add(new int[] {i, j});
            }
            i = j;
        }
        return out;
    }

    /**
     * Part of audio, between two frames of its envelope.
     *
     * @param a the samples
     * @param rate the sample rate
     * @param fromFrame the first frame
     * @param toFrame the frame after the last
     * @return the part
     */
    static float[] slice(float[] a, int rate, int fromFrame, int toFrame) {
        int hop = rate / FRAME_RATE;
        int from = Math.max(0, Math.min(a.length, fromFrame * hop));
        int to = Math.max(from, Math.min(a.length, toFrame * hop));
        return Arrays.copyOfRange(a, from, to);
    }

    /**
     * Trims silence off both ends, keeping a margin, and fades the ends in and out.
     *
     * @param x the samples
     * @param rate the sample rate
     * @param marginSeconds silence to keep at each end
     * @return the trimmed samples
     */
    static float[] trim(float[] x, int rate, double marginSeconds) {
        if (x.length == 0) {
            return x;
        }
        int[] sp = speech(envelope(x, rate));
        int hop = rate / FRAME_RATE;
        int margin = (int) (marginSeconds * rate);
        int from = Math.max(0, sp[0] * hop - margin);
        int to = Math.min(x.length, (sp[1] + 1) * hop + margin);
        float[] out = Arrays.copyOfRange(x, from, to);
        fade(out, rate, 0.005, 0.005);
        return out;
    }

    /**
     * Fades the ends of samples in place (raised cosine).
     *
     * @param x the samples
     * @param rate the sample rate
     * @param in fade-in seconds
     * @param out fade-out seconds
     */
    static void fade(float[] x, int rate, double in, double out) {
        int a = Math.min(x.length / 2, (int) (in * rate));
        for (int i = 0; i < a; i++) {
            x[i] *= (float) (0.5 - 0.5 * Math.cos(Math.PI * i / a));
        }
        int b = Math.min(x.length / 2, (int) (out * rate));
        for (int i = 0; i < b; i++) {
            x[x.length - 1 - i] *= (float) (0.5 - 0.5 * Math.cos(Math.PI * i / b));
        }
    }

    /**
     * Changes the tempo of speech without changing its pitch (WSOLA: overlapping 30 ms frames
     * taken at a different hop than they are laid down, each shifted by up to 10 ms to line up
     * with the waveform already written).
     *
     * @param x the samples
     * @param rate the sample rate
     * @param speed how much faster, such as 1.2; 1 returns a copy
     * @return the samples, about {@code x.length / speed} long
     */
    static float[] stretch(float[] x, int rate, double speed) {
        if (Math.abs(speed - 1) < 1e-3 || x.length == 0) {
            return x.clone();
        }
        int n = Math.max(16, (int) (rate * 0.03));
        int hs = n / 2;
        int tol = (int) (rate * 0.01);
        int outLen = (int) Math.round(x.length / speed);
        double[] out = new double[outLen + n];
        double[] norm = new double[outLen + n];
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
        }
        int prev = 0;
        for (int k = 0; k * hs < outLen; k++) {
            int pos;
            if (k == 0) {
                pos = 0;
            } else {
                int target = (int) Math.round(k * hs * speed);
                int natural = prev + hs;
                double best = Double.NEGATIVE_INFINITY;
                pos = Math.max(0, Math.min(x.length - 1, target));
                for (int d = -tol; d <= tol; d++) {
                    int p = target + d;
                    if (p < 0 || p >= x.length) {
                        continue;
                    }
                    double c = 0;
                    for (int i = 0; i < n; i += 2) {
                        int a = p + i;
                        int b = natural + i;
                        if (a >= x.length || b >= x.length) {
                            break;
                        }
                        c += (double) x[a] * x[b];
                    }
                    if (c > best) {
                        best = c;
                        pos = p;
                    }
                }
            }
            int at = k * hs;
            for (int i = 0; i < n && pos + i < x.length; i++) {
                out[at + i] += x[pos + i] * w[i];
                norm[at + i] += w[i];
            }
            prev = pos;
        }
        float[] y = new float[outLen];
        for (int i = 0; i < outLen; i++) {
            y[i] = (float) (norm[i] > 1e-3 ? out[i] / norm[i] : out[i]);
        }
        return y;
    }

    private static float[][] melBank(int rate, int nFft, int bands, double lo, double hi) {
        double mLo = 2595 * Math.log10(1 + lo / 700);
        double mHi = 2595 * Math.log10(1 + hi / 700);
        double[] hz = new double[bands + 2];
        for (int i = 0; i < hz.length; i++) {
            double m = mLo + (mHi - mLo) * i / (bands + 1);
            hz[i] = 700 * (Math.pow(10, m / 2595) - 1);
        }
        float[][] fb = new float[bands][nFft / 2 + 1];
        for (int b = 0; b < bands; b++) {
            for (int k = 0; k <= nFft / 2; k++) {
                double f = (double) k * rate / nFft;
                double up = (f - hz[b]) / (hz[b + 1] - hz[b]);
                double down = (hz[b + 2] - f) / (hz[b + 2] - hz[b + 1]);
                fb[b][k] = (float) Math.max(0, Math.min(up, down));
            }
        }
        return fb;
    }

    /** In-place radix-2 FFT. */
    static void fft(double[] re, double[] im) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                double t = re[i];
                re[i] = re[j];
                re[j] = t;
                t = im[i];
                im[i] = im[j];
                im[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2 * Math.PI / len;
            double wr = Math.cos(ang);
            double wi = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double cr = 1;
                double ci = 0;
                for (int k = 0; k < len / 2; k++) {
                    int a = i + k;
                    int b = a + len / 2;
                    double xr = re[b] * cr - im[b] * ci;
                    double xi = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                    double nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
            }
        }
    }

    /**
     * Text reduced for comparing what was said: numbers spelled, lower case, no punctuation, no spaces or hyphens (German writes "zweiundvierzig" and
     * Whisper may write "42").
     *
     * @param text the text
     * @param speller the language's number words
     * @return the reduced text
     */
    static String normalize(String text, NumberSpeller speller) {
        String s = speller.spell(text).toLowerCase(Locale.ROOT);
        s = Normalizer.normalize(s, Normalizer.Form.NFC);
        StringBuilder b = new StringBuilder();
        s.codePoints().filter(Character::isLetterOrDigit).forEach(b::appendCodePoint);
        return b.toString().replace("ß", "ss");
    }

    /**
     * How alike two texts are: 1 minus their edit distance over the longer length.
     *
     * @param a one text, normalized
     * @param b another, normalized
     * @return 0 to 1
     */
    static double similarity(String a, String b) {
        if (a.isEmpty() && b.isEmpty()) {
            return 1;
        }
        int[] prev = new int[b.length() + 1];
        int[] cur = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= a.length(); i++) {
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int sub = prev[j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1);
                cur[j] = Math.min(sub, Math.min(prev[j] + 1, cur[j - 1] + 1));
            }
            int[] t = prev;
            prev = cur;
            cur = t;
        }
        return 1 - (double) prev[b.length()] / Math.max(a.length(), b.length());
    }

    /**
     * A rough voice print: the mean and spread of each log-mel band over the voiced frames,
     * before mean removal. Used to rank takes when no speaker encoder is given.
     *
     * @param audio the audio
     * @return the print
     */
    static float[] voicePrint(Audio audio) {
        float[] x = audio.samples();
        int rate = audio.sampleRate();
        double[] env = envelope(x, rate);
        double peak = Arrays.stream(env).max().orElse(-180);
        int hop = rate / FRAME_RATE;
        int win = (int) (rate * 0.025);
        int nFft = Integer.highestOneBit(win - 1) << 1;
        float[][] fb = melBank(rate, nFft, 40, 60, Math.min(8000, rate / 2.0));
        double[] sum = new double[40];
        double[] sq = new double[40];
        int count = 0;
        double[] re = new double[nFft];
        double[] im = new double[nFft];
        for (int f = 0; f < env.length; f++) {
            if (env[f] < peak - 25) {
                continue;
            }
            Arrays.fill(re, 0);
            Arrays.fill(im, 0);
            int from = f * hop + hop / 2 - win / 2;
            for (int i = 0; i < win; i++) {
                int at = from + i;
                if (at >= 0 && at < x.length) {
                    re[i] = x[at] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win - 1)));
                }
            }
            fft(re, im);
            double total = 0;
            double[] band = new double[40];
            for (int m = 0; m < 40; m++) {
                double e = 0;
                for (int k = 0; k <= nFft / 2; k++) {
                    e += fb[m][k] * (re[k] * re[k] + im[k] * im[k]);
                }
                band[m] = Math.log(e + 1e-8);
                total += band[m];
            }
            for (int m = 0; m < 40; m++) {
                double v = band[m] - total / 40;   // level-independent spectral shape
                sum[m] += v;
                sq[m] += v * v;
            }
            count++;
        }
        float[] out = new float[80];
        for (int m = 0; m < 40 && count > 0; m++) {
            double mean = sum[m] / count;
            out[m] = (float) mean;
            out[40 + m] = (float) Math.sqrt(Math.max(0, sq[m] / count - mean * mean));
        }
        return out;
    }

    /**
     * The cosine of the angle between two vectors.
     *
     * @param a one vector
     * @param b another of the same length
     * @return -1 to 1
     */
    static double cosine(float[] a, float[] b) {
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int i = 0; i < a.length; i++) {
            ab += (double) a[i] * b[i];
            aa += (double) a[i] * a[i];
            bb += (double) b[i] * b[i];
        }
        return aa == 0 || bb == 0 ? 0 : ab / Math.sqrt(aa * bb);
    }
}
