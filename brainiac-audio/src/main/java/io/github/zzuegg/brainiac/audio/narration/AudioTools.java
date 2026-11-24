package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * The signal work behind cutting and splicing speech: a loudness envelope, the quiet dips
 * between words where a cut is least audible, crossfaded joins, log-mel frames, dynamic time
 * warping between two takes, and comparing transcribed text.
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
     * A place where a cut is least audible.
     *
     * @param frame the envelope frame
     * @param level its level, dB of full scale
     * @param depth how far below the louder sides it lies, in dB (more is a better cut)
     */
    record Valley(int frame, double level, double depth) {
    }

    /**
     * The dips of an envelope: frames lower than their neighbours, with how deep each is
     * against the loudest level within 80 ms on either side. Silent stretches give one valley
     * in their middle, very deep.
     *
     * @param env the envelope
     * @return the valleys, in time order
     */
    static List<Valley> valleys(double[] env) {
        double peak = Arrays.stream(env).max().orElse(-180);
        double silent = peak - SILENCE_DB;
        double[] s = smooth(env);
        List<Valley> out = new ArrayList<>();
        int n = s.length;
        int i = 1;
        while (i < n - 1) {
            if (s[i] < silent) {
                int j = i;
                while (j < n - 1 && s[j] < silent) {
                    j++;
                }
                int mid = (i + j) / 2;
                out.add(new Valley(mid, s[mid], SILENCE_DB + (j - i)));
                i = j + 1;
                continue;
            }
            if (s[i] <= s[i - 1] && s[i] < s[i + 1]) {
                double left = -180;
                double right = -180;
                for (int k = Math.max(0, i - 8); k < i; k++) {
                    left = Math.max(left, s[k]);
                }
                for (int k = i + 1; k <= Math.min(n - 1, i + 8); k++) {
                    right = Math.max(right, s[k]);
                }
                double depth = Math.min(left, right) - s[i];
                if (depth >= 1.5) {
                    out.add(new Valley(i, s[i], depth));
                }
            }
            i++;
        }
        return out;
    }

    private static double[] smooth(double[] env) {
        double[] s = new double[env.length];
        for (int i = 0; i < env.length; i++) {
            double sum = 0;
            int c = 0;
            for (int k = Math.max(0, i - 1); k <= Math.min(env.length - 1, i + 1); k++) {
                sum += env[k];
                c++;
            }
            s[i] = sum / c;
        }
        return s;
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
     * Joins pieces of audio, overlapping each join by an equal-power crossfade.
     *
     * @param parts the pieces, in order
     * @param rate the sample rate
     * @param fadeSeconds the overlap of each join
     * @return the joined audio
     */
    static float[] join(List<float[]> parts, int rate, double fadeSeconds) {
        int fade = (int) (fadeSeconds * rate);
        int total = 0;
        for (float[] p : parts) {
            total += p.length;
        }
        float[] out = new float[total];
        int end = 0;
        for (float[] p : parts) {
            int f = Math.min(fade, Math.min(end, p.length / 2));
            int start = end - f;
            for (int i = 0; i < p.length; i++) {
                double v = p[i];
                if (i < f) {
                    double t = (i + 0.5) / f;
                    v = out[start + i] * Math.cos(t * Math.PI / 2) + v * Math.sin(t * Math.PI / 2);
                }
                out[start + i] = (float) v;
            }
            end = start + p.length;
        }
        return Arrays.copyOf(out, end);
    }

    /**
     * Log-mel frames (10 ms hop, 25 ms window, 40 bands up to 8 kHz), with each band's mean
     * removed, for comparing takes of the same words.
     *
     * @param x the samples
     * @param rate the sample rate
     * @return {@code [frames][40]}
     */
    static float[][] logMel(float[] x, int rate) {
        int hop = rate / FRAME_RATE;
        int win = (int) (rate * 0.025);
        int nFft = Integer.highestOneBit(win - 1) << 1;
        int bands = 40;
        int frames = Math.max(1, (x.length + hop - 1) / hop);
        float[][] fb = melBank(rate, nFft, bands, 60, Math.min(8000, rate / 2.0));
        float[][] out = new float[frames][bands];
        double[] re = new double[nFft];
        double[] im = new double[nFft];
        for (int f = 0; f < frames; f++) {
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
            for (int m = 0; m < bands; m++) {
                double e = 0;
                for (int k = 0; k <= nFft / 2; k++) {
                    if (fb[m][k] != 0) {
                        e += fb[m][k] * (re[k] * re[k] + im[k] * im[k]);
                    }
                }
                out[f][m] = (float) Math.log(e + 1e-8);
            }
        }
        for (int m = 0; m < bands; m++) {
            double mean = 0;
            for (float[] fr : out) {
                mean += fr[m];
            }
            mean /= frames;
            for (float[] fr : out) {
                fr[m] -= (float) mean;
            }
        }
        return out;
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
     * Dynamic time warping between two takes: for each frame of {@code a}, the frame of
     * {@code b} the best path matches it to (the middle one if several).
     *
     * @param a the first take's frames
     * @param b the second take's frames
     * @return {@code map[i]}, a frame of {@code b} for frame {@code i} of {@code a}
     */
    static int[] warp(float[][] a, float[][] b) {
        int n = a.length;
        int m = b.length;
        double[][] cost = new double[n + 1][m + 1];
        for (double[] row : cost) {
            Arrays.fill(row, Double.POSITIVE_INFINITY);
        }
        cost[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                double d = 0;
                float[] x = a[i - 1];
                float[] y = b[j - 1];
                for (int k = 0; k < x.length; k++) {
                    double t = x[k] - y[k];
                    d += t * t;
                }
                d = Math.sqrt(d);
                cost[i][j] = d + Math.min(cost[i - 1][j - 1], Math.min(cost[i - 1][j], cost[i][j - 1]));
            }
        }
        int[] lo = new int[n];
        int[] hi = new int[n];
        Arrays.fill(lo, Integer.MAX_VALUE);
        int i = n;
        int j = m;
        while (i > 0 && j > 0) {
            lo[i - 1] = Math.min(lo[i - 1], j - 1);
            hi[i - 1] = Math.max(hi[i - 1], j - 1);
            double diag = cost[i - 1][j - 1];
            double up = cost[i - 1][j];
            double left = cost[i][j - 1];
            if (diag <= up && diag <= left) {
                i--;
                j--;
            } else if (up <= left) {
                i--;
            } else {
                j--;
            }
        }
        int[] map = new int[n];
        for (int k = 0; k < n; k++) {
            map[k] = lo[k] == Integer.MAX_VALUE ? (k > 0 ? map[k - 1] : 0) : (lo[k] + hi[k]) / 2;
        }
        return map;
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
