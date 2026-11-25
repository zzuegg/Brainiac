package io.github.zzuegg.brainiac.audio.narration;

import java.util.Arrays;

/**
 * The melody of speech: tracking its pitch, and changing its pitch and tempo over time
 * without changing its sound, so pieces said on their own can follow the melody of one
 * sentence.
 */
final class Prosody {

    /** The lowest pitch looked for, in Hz. */
    static final double MIN_HZ = 65;
    /** The highest pitch looked for, in Hz. */
    static final double MAX_HZ = 450;

    private Prosody() {
    }

    /**
     * The pitch of each 10 ms frame (YIN), 0 where the frame is unvoiced or quiet.
     *
     * @param x the samples
     * @param rate the sample rate
     * @return Hz per frame of {@link AudioTools#envelope}
     */
    static double[] pitch(float[] x, int rate) {
        double[] env = AudioTools.envelope(x, rate);
        double peak = Arrays.stream(env).max().orElse(-180);
        int hop = rate / AudioTools.FRAME_RATE;
        int maxLag = (int) (rate / MIN_HZ);
        int minLag = (int) (rate / MAX_HZ);
        int win = maxLag;
        double[] out = new double[env.length];
        double[] d = new double[maxLag + 1];
        for (int f = 0; f < env.length; f++) {
            if (env[f] < peak - 30) {
                continue;
            }
            int from = f * hop + hop / 2 - (win + maxLag) / 2;
            if (from < 0 || from + win + maxLag >= x.length) {
                continue;
            }
            // difference function, cumulative mean normalized
            double run = 0;
            d[0] = 1;
            for (int lag = 1; lag <= maxLag; lag++) {
                double s = 0;
                for (int i = 0; i < win; i += 2) {
                    double v = x[from + i] - x[from + i + lag];
                    s += v * v;
                }
                run += s;
                d[lag] = run > 0 ? s * lag / run : 1;
            }
            int best = -1;
            for (int lag = minLag; lag < maxLag; lag++) {
                if (d[lag] < 0.2 && d[lag] <= d[lag + 1]) {
                    best = lag;
                    break;
                }
            }
            if (best < 0) {
                continue;
            }
            // parabolic interpolation around the dip
            double a = d[best - 1];
            double b = d[best];
            double c = d[best + 1];
            double den = a - 2 * b + c;
            double lag = best + (Math.abs(den) > 1e-12 ? 0.5 * (a - c) / den : 0);
            out[f] = rate / lag;
        }
        return out;
    }

    /**
     * Pitch in semitones relative to a reference, smoothed over 50 ms, with unvoiced frames
     * filled in from their neighbours.
     *
     * @param hz pitch per frame, 0 where unvoiced
     * @param refHz the reference
     * @return semitones per frame, or {@code null} if no frame is voiced
     */
    static double[] contour(double[] hz, double refHz) {
        int n = hz.length;
        double[] st = new double[n];
        boolean any = false;
        for (int i = 0; i < n; i++) {
            if (hz[i] > 0) {
                st[i] = 12 * Math.log(hz[i] / refHz) / Math.log(2);
                any = true;
            } else {
                st[i] = Double.NaN;
            }
        }
        if (!any) {
            return null;
        }
        // median over 5 frames, voiced only, to drop octave slips
        double[] med = new double[n];
        double[] buf = new double[5];
        for (int i = 0; i < n; i++) {
            int k = 0;
            for (int j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) {
                if (!Double.isNaN(st[j])) {
                    buf[k++] = st[j];
                }
            }
            if (Double.isNaN(st[i]) || k == 0) {
                med[i] = Double.NaN;
            } else {
                Arrays.sort(buf, 0, k);
                med[i] = buf[k / 2];
            }
        }
        // fill gaps: hold the nearest voiced value at the ends, interpolate between
        int first = -1;
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(med[i])) {
                if (first < 0) {
                    Arrays.fill(med, 0, i, med[i]);
                } else if (i - first > 1) {
                    for (int j = first + 1; j < i; j++) {
                        med[j] = med[first] + (med[i] - med[first]) * (j - first) / (i - first);
                    }
                }
                first = i;
            }
        }
        if (first >= 0) {
            Arrays.fill(med, first + 1, n, med[first]);
        }
        return med;
    }

    /**
     * The median of the voiced frames.
     *
     * @param hz pitch per frame, 0 where unvoiced
     * @return Hz, or {@code NaN} if none is voiced
     */
    static double median(double[] hz) {
        double[] v = Arrays.stream(hz).filter(h -> h > 0).sorted().toArray();
        return v.length == 0 ? Double.NaN : v[v.length / 2];
    }

    /**
     * Changes tempo and pitch over time without moving the voice's resonances (TD-PSOLA): voiced
     * stretches are cut into two-period grains centred on each glottal pulse, and the grains are
     * laid down again at the new period, so the pitch changes while each grain keeps the shape
     * of the vocal tract; unvoiced stretches are overlap-added in 5 ms grains at the new tempo.
     *
     * @param x the samples
     * @param rate the sample rate
     * @param speed how much faster, such as 1.2
     * @param ratio the pitch factor for each 10 ms frame of the input, or {@code null} for none
     * @return the samples, about {@code x.length / speed} long
     */
    static float[] psola(float[] x, int rate, double speed, double[] ratio) {
        if (x.length == 0 || ratio == null && Math.abs(speed - 1) < 1e-3) {
            return x.clone();
        }
        int hop = rate / AudioTools.FRAME_RATE;
        double[] hz = pitch(x, rate);
        // analysis marks: one per glottal pulse where voiced, every 5 ms elsewhere
        java.util.List<int[]> marks = new java.util.ArrayList<>();   // {position, period, voiced}
        int unv = Math.max(1, rate / 200);
        int t = 0;
        while (t < x.length) {
            int f = Math.min(hz.length - 1, t / hop);
            if (hz[f] > 0) {
                int period = (int) Math.round(rate / hz[f]);
                // the strongest positive peak, near where the next pulse is due (a whole period
                // to look in at the start of a voiced stretch), so the marks keep one polarity
                boolean continuing = !marks.isEmpty() && marks.get(marks.size() - 1)[2] == 1;
                int reach = continuing ? period / 4 : period / 2;
                int from = Math.max(0, t - reach);
                int to = Math.min(x.length - 1, t + reach);
                int best = t;
                double peak = Double.NEGATIVE_INFINITY;
                for (int i = from; i <= to; i++) {
                    if (x[i] > peak) {
                        peak = x[i];
                        best = i;
                    }
                }
                if (!marks.isEmpty() && best <= marks.get(marks.size() - 1)[0]) {
                    best = marks.get(marks.size() - 1)[0] + period;
                }
                marks.add(new int[] {best, period, 1});
                t = best + period;
            } else {
                marks.add(new int[] {t, unv, 0});
                t += unv;
            }
        }
        int outLen = (int) Math.round(x.length / speed);
        double[] out = new double[outLen + rate / 10];
        double[] norm = new double[out.length];
        double u = 0;
        int k = 0;
        while (u < outLen) {
            double a = u * speed;   // where in the input this output moment comes from
            while (k + 1 < marks.size() && Math.abs(marks.get(k + 1)[0] - a) <= Math.abs(marks.get(k)[0] - a)) {
                k++;
            }
            int[] m = marks.get(k);
            int period = m[1];
            double r = 1;
            if (m[2] == 1 && ratio != null && ratio.length > 0) {
                int f = Math.max(0, Math.min(ratio.length - 1, m[0] / hop));
                r = ratio[f];
            }
            int c = (int) Math.round(u);
            for (int i = -period; i < period; i++) {
                int src = m[0] + i;
                int dst = c + i;
                if (src < 0 || src >= x.length || dst < 0 || dst >= out.length) {
                    continue;
                }
                double w = 0.5 + 0.5 * Math.cos(Math.PI * i / period);
                out[dst] += x[src] * w;
                norm[dst] += w;
            }
            u += m[2] == 1 ? period / r : period;
        }
        float[] y = new float[outLen];
        for (int i = 0; i < outLen; i++) {
            y[i] = (float) (norm[i] > 0.5 ? out[i] / norm[i] : out[i]);
        }
        return y;
    }

    /**
     * Changes tempo and pitch over time, keeping the sound of the voice: a WSOLA time stretch
     * by {@code speed / ratio}, then reading the result {@code ratio} times as fast.
     *
     * @param x the samples
     * @param rate the sample rate
     * @param speed how much faster, such as 1.2
     * @param ratio the pitch factor for each 10 ms frame of the input, or {@code null} for none
     * @return the samples, about {@code x.length / speed} long
     */
    static float[] reshape(float[] x, int rate, double speed, double[] ratio) {
        if (x.length == 0) {
            return x.clone();
        }
        int hop = rate / AudioTools.FRAME_RATE;
        java.util.function.DoubleUnaryOperator r = pos -> {
            if (ratio == null || ratio.length == 0) {
                return 1;
            }
            double f = pos / hop;
            int i = (int) Math.floor(f);
            if (i < 0) {
                return ratio[0];
            }
            if (i >= ratio.length - 1) {
                return ratio[ratio.length - 1];
            }
            return ratio[i] + (ratio[i + 1] - ratio[i]) * (f - i);
        };
        // 1. stretch: analysis advances hs * speed / r per synthesis hop
        int n = Math.max(16, (int) (rate * 0.03));
        int hs = n / 2;
        int tol = (int) (rate * 0.01);
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
        }
        int frames = 1;
        for (double p = 0; p < x.length; p += hs * speed / r.applyAsDouble(p)) {
            frames++;
        }
        int midLen = frames * hs + n;
        double[] mid = new double[midLen];
        double[] norm = new double[midLen];
        double[] midRatio = new double[frames + 1];
        double target = 0;
        int prev = 0;
        int k = 0;
        for (; k < frames && target < x.length; k++) {
            int pos;
            if (k == 0) {
                pos = 0;
            } else {
                int t = (int) Math.round(target);
                int natural = prev + hs;
                double best = Double.NEGATIVE_INFINITY;
                pos = Math.max(0, Math.min(x.length - 1, t));
                for (int dd = -tol; dd <= tol; dd++) {
                    int p = t + dd;
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
                mid[at + i] += x[pos + i] * w[i];
                norm[at + i] += w[i];
            }
            double rr = r.applyAsDouble(target);
            midRatio[k] = rr;
            prev = pos;
            target += hs * speed / rr;
        }
        int used = Math.min(midLen, k * hs + n / 2);
        for (int i = 0; i < used; i++) {
            mid[i] = norm[i] > 1e-3 ? mid[i] / norm[i] : mid[i];
        }
        if (ratio == null) {
            float[] y = new float[Math.min(used, (int) Math.round(x.length / speed))];
            for (int i = 0; i < y.length; i++) {
                y[i] = (float) mid[i];
            }
            return y;
        }
        // 2. read the stretched audio ratio times as fast: duration back, pitch times ratio
        int outLen = (int) Math.round(x.length / speed);
        float[] y = new float[outLen];
        double q = 0;
        int j = 0;
        for (; j < outLen && q < used - 1; j++) {
            int i = (int) q;
            double fr = q - i;
            y[j] = (float) (mid[i] * (1 - fr) + mid[i + 1] * fr);
            double f = q / hs;
            int fi = Math.min(k - 1, (int) f);
            double rr = fi + 1 < k ? midRatio[fi] + (midRatio[fi + 1] - midRatio[fi]) * (f - fi) : midRatio[fi];
            q += rr;
        }
        return j == outLen ? y : Arrays.copyOf(y, j);
    }
}
