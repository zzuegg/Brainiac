package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.Arrays;

/**
 * Symbolic series words: Symbolic Fourier Approximation (SFA; Schäfer and Högqvist, EDBT 2012)
 * and Symbolic Aggregate approXimation (SAX; Lin et al., DMKD 2007), the front ends of the
 * dictionary classifiers {@link Boss} and {@link Weasel}.
 *
 * <p>SFA turns a window into its first {@code wordLength} real and imaginary Fourier
 * coefficients (dropping the mean when the window is mean-normalised), then quantises each
 * coefficient into {@code alphabet} symbols with per-coefficient breakpoints learned from
 * training windows: {@link Binning#EQUI_DEPTH} is multiple coefficient binning (MCB, quantiles of
 * each coefficient), {@link Binning#INFORMATION_GAIN} places the breakpoints where they best
 * separate the classes (WEASEL's supervised binning). SAX z-normalises the window, averages it
 * over {@code wordLength} segments and cuts at standard-normal quantiles. Words are packed into an
 * {@code int}, {@code log2(alphabet)} bits per symbol.
 */
public final class SymbolicFourier {

    /** How SFA's breakpoints are learned. */
    public enum Binning {
        /** Multiple coefficient binning: equal-frequency bins per coefficient. */
        EQUI_DEPTH,
        /** Breakpoints that maximise the information gain about the class. */
        INFORMATION_GAIN
    }

    private final int window;
    private final int wordLength;
    private final int alphabet;
    private final boolean normMean;
    private final boolean normStd;
    private final double[][] breakpoints;
    private final int bits;

    private SymbolicFourier(int window, int wordLength, int alphabet, boolean normMean, boolean normStd,
            double[][] breakpoints) {
        this.normStd = normStd;
        this.window = window;
        this.wordLength = wordLength;
        this.alphabet = alphabet;
        this.normMean = normMean;
        this.breakpoints = breakpoints;
        this.bits = 32 - Integer.numberOfLeadingZeros(alphabet - 1);
    }

    /**
     * Learns SFA breakpoints from the sliding windows of training series.
     *
     * @param series training series
     * @param labels their classes (used by {@link Binning#INFORMATION_GAIN}; may be null for MCB)
     * @param window the window length
     * @param wordLength symbols per word (Fourier values kept), even, at most 16
     * @param alphabet symbols per position, a power of two up to 16
     * @param normMean whether to drop the window mean (the DC coefficient)
     * @param binning how to place the breakpoints
     * @param stride the step between sampled training windows, at least 1
     * @return the transform
     */
    public static SymbolicFourier fit(float[][] series, int[] labels, int window, int wordLength, int alphabet,
            boolean normMean, Binning binning, int stride) {
        return fit(series, labels, window, wordLength, alphabet, normMean, true, binning, stride);
    }

    /**
     * Learns SFA breakpoints, optionally without dividing windows by their standard deviation
     * (so flat, noise-only windows map to small coefficients instead of being blown up).
     *
     * @param series training series
     * @param labels their classes (used by {@link Binning#INFORMATION_GAIN}; may be null for MCB)
     * @param window the window length
     * @param wordLength symbols per word (Fourier values kept), even, at most 16
     * @param alphabet symbols per position, a power of two up to 16
     * @param normMean whether to drop the window mean (the DC coefficient)
     * @param normStd whether to divide each window by its standard deviation (SFA's default)
     * @param binning how to place the breakpoints
     * @param stride the step between sampled training windows, at least 1
     * @return the transform
     */
    public static SymbolicFourier fit(float[][] series, int[] labels, int window, int wordLength, int alphabet,
            boolean normMean, boolean normStd, Binning binning, int stride) {
        int count = 0;
        for (float[] s : series) {
            count += Math.max(0, (s.length - window) / stride + 1);
        }
        double[][] values = new double[wordLength][count];
        int[] cls = new int[count];
        int k = 0;
        for (int i = 0; i < series.length; i++) {
            for (int start = 0; start + window <= series[i].length; start += stride) {
                double[] f = dft(series[i], start, window, wordLength, normMean, normStd);
                for (int c = 0; c < wordLength; c++) {
                    values[c][k] = f[c];
                }
                cls[k] = labels == null ? 0 : labels[i];
                k++;
            }
        }
        double[][] bp = new double[wordLength][];
        for (int c = 0; c < wordLength; c++) {
            bp[c] = binning == Binning.EQUI_DEPTH || labels == null ? equiDepth(values[c], alphabet)
                    : informationGain(values[c], cls, alphabet);
        }
        return new SymbolicFourier(window, wordLength, alphabet, normMean, normStd, bp);
    }

    /**
     * The first Fourier values of a window, normalised by the window's standard deviation:
     * {@code re_1, im_1, re_2, im_2, ...} (starting at the DC term unless {@code normMean}).
     *
     * @param s the series
     * @param start the window's first index
     * @param window the window length
     * @param count how many real values to return
     * @param normMean whether to skip the DC coefficient
     * @return the values
     */
    public static double[] dft(float[] s, int start, int window, int count, boolean normMean) {
        return dft(s, start, window, count, normMean, true);
    }

    /**
     * The first Fourier values of a window, optionally normalised by its standard deviation.
     *
     * @param s the series
     * @param start the window's first index
     * @param window the window length
     * @param count how many real values to return
     * @param normMean whether to skip the DC coefficient
     * @param normStd whether to divide by the window's standard deviation
     * @return the values
     */
    public static double[] dft(float[] s, int start, int window, int count, boolean normMean, boolean normStd) {
        double mean = 0;
        for (int t = 0; t < window; t++) {
            mean += s[start + t];
        }
        mean /= window;
        double var = 0;
        for (int t = 0; t < window; t++) {
            double v = s[start + t] - mean;
            var += v * v;
        }
        double sd = Math.sqrt(var / window);
        double norm = normStd && sd > 1e-8 ? 1 / sd : 1;
        double[] out = new double[count];
        int f0 = normMean ? 1 : 0;
        double[][] table = TRIG.computeIfAbsent(window, SymbolicFourier::trig);
        for (int c = 0; c < count; c += 2) {
            int freq = f0 + c / 2;
            double re = 0;
            double im = 0;
            double[] cos = table[2 * freq];
            double[] sin = table[2 * freq + 1];
            for (int t = 0; t < window; t++) {
                double v = s[start + t] - (normMean ? mean : 0);
                re += v * cos[t];
                im -= v * sin[t];
            }
            out[c] = re * norm / Math.sqrt(window);
            if (c + 1 < count) {
                out[c + 1] = im * norm / Math.sqrt(window);
            }
        }
        return out;
    }

    private static final java.util.Map<Integer, double[][]> TRIG = new java.util.concurrent.ConcurrentHashMap<>();

    /** Cosine and sine tables for the first 17 frequencies of a window length. */
    private static double[][] trig(int window) {
        double[][] t = new double[34][window];
        for (int f = 0; f < 17; f++) {
            for (int u = 0; u < window; u++) {
                double a = 2 * Math.PI * f * u / window;
                t[2 * f][u] = Math.cos(a);
                t[2 * f + 1][u] = Math.sin(a);
            }
        }
        return t;
    }

    private static double[] equiDepth(double[] v, int alphabet) {
        double[] sorted = v.clone();
        Arrays.sort(sorted);
        double[] bp = new double[alphabet - 1];
        for (int b = 1; b < alphabet; b++) {
            bp[b - 1] = sorted[Math.min(sorted.length - 1, (int) ((long) b * sorted.length / alphabet))];
        }
        return bp;
    }

    /** Greedy recursive splits of the sorted values that maximise information gain, alphabet - 1 cuts. */
    private static double[] informationGain(double[] v, int[] cls, int alphabet) {
        int n = v.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(v[a], v[b]));
        double[] sv = new double[n];
        int[] sc = new int[n];
        int classes = 0;
        for (int i = 0; i < n; i++) {
            sv[i] = v[order[i]];
            sc[i] = cls[order[i]];
            classes = Math.max(classes, sc[i] + 1);
        }
        java.util.List<Integer> cuts = new java.util.ArrayList<>();
        split(sv, sc, classes, 0, n, cuts, alphabet - 1);
        if (cuts.size() < alphabet - 1) {
            double[] eq = equiDepth(v, alphabet);
            for (double e : eq) {
                if (cuts.size() >= alphabet - 1) {
                    break;
                }
                int idx = Arrays.binarySearch(sv, e);
                cuts.add(Math.max(1, Math.min(n - 1, idx < 0 ? -idx - 1 : idx)));
            }
        }
        double[] bp = new double[alphabet - 1];
        java.util.Collections.sort(cuts);
        for (int b = 0; b < bp.length; b++) {
            int c = cuts.get(b);
            bp[b] = 0.5 * (sv[c - 1] + sv[c]);
        }
        Arrays.sort(bp);
        return bp;
    }

    private static void split(double[] sv, int[] sc, int classes, int from, int to, java.util.List<Integer> cuts,
            int budget) {
        if (budget <= 0 || to - from < 2) {
            return;
        }
        int[] total = new int[classes];
        for (int i = from; i < to; i++) {
            total[sc[i]]++;
        }
        double hAll = entropy(total, to - from);
        int[] left = new int[classes];
        double bestGain = 1e-12;
        int best = -1;
        for (int i = from + 1; i < to; i++) {
            left[sc[i - 1]]++;
            if (sv[i] == sv[i - 1]) {
                continue;
            }
            int nl = i - from;
            int nr = to - i;
            int[] right = new int[classes];
            for (int c = 0; c < classes; c++) {
                right[c] = total[c] - left[c];
            }
            double gain = hAll - (nl * entropy(left, nl) + nr * entropy(right, nr)) / (to - from);
            if (gain > bestGain) {
                bestGain = gain;
                best = i;
            }
        }
        if (best < 0) {
            return;
        }
        cuts.add(best);
        int remaining = budget - 1;
        int leftBudget = remaining / 2;
        split(sv, sc, classes, from, best, cuts, leftBudget);
        split(sv, sc, classes, best, to, cuts, remaining - leftBudget);
    }

    private static double entropy(int[] counts, int n) {
        double h = 0;
        for (int c : counts) {
            if (c > 0) {
                double p = c / (double) n;
                h -= p * Math.log(p);
            }
        }
        return h;
    }

    /**
     * The word of the window starting at {@code start}.
     *
     * @param s the series
     * @param start the window's first index
     * @return the packed word
     */
    public int word(float[] s, int start) {
        return quantise(dft(s, start, window, wordLength, normMean, normStd));
    }

    /**
     * Quantises Fourier values into a packed word.
     *
     * @param values {@code wordLength} values
     * @return the packed word
     */
    public int quantise(double[] values) {
        int w = 0;
        for (int c = 0; c < wordLength; c++) {
            int sym = 0;
            while (sym < breakpoints[c].length && values[c] > breakpoints[c][sym]) {
                sym++;
            }
            w = (w << bits) | sym;
        }
        return w;
    }

    /**
     * The words of every sliding window.
     *
     * @param s the series
     * @return one word per window start
     */
    public int[] words(float[] s) {
        int m = Math.max(0, s.length - window + 1);
        int[] out = new int[m];
        for (int t = 0; t < m; t++) {
            out[t] = word(s, t);
        }
        return out;
    }

    /** @return the window length */
    public int window() {
        return window;
    }

    /** @return the bits of a packed word, {@code wordLength * log2(alphabet)} */
    public int wordBits() {
        return wordLength * bits;
    }

    /**
     * SAX words of every sliding window: z-normalise, average into {@code wordLength} segments,
     * cut at the standard-normal quantiles {@code Phi^-1(k / alphabet)}.
     *
     * @param s the series
     * @param window the window length
     * @param wordLength segments per window
     * @param alphabet symbols per segment, a power of two up to 16
     * @return one packed word per window start
     */
    public static int[] sax(float[] s, int window, int wordLength, int alphabet) {
        double[] bp = new double[alphabet - 1];
        for (int k = 1; k < alphabet; k++) {
            bp[k - 1] = NormalTails.quantile(k / (double) alphabet);
        }
        int bits = 32 - Integer.numberOfLeadingZeros(alphabet - 1);
        int m = Math.max(0, s.length - window + 1);
        int[] out = new int[m];
        for (int t = 0; t < m; t++) {
            double mean = 0;
            for (int u = 0; u < window; u++) {
                mean += s[t + u];
            }
            mean /= window;
            double var = 0;
            for (int u = 0; u < window; u++) {
                var += (s[t + u] - mean) * (s[t + u] - mean);
            }
            double sd = Math.sqrt(var / window);
            sd = sd > 1e-8 ? sd : 1;
            int w = 0;
            for (int seg = 0; seg < wordLength; seg++) {
                int a = seg * window / wordLength;
                int b = Math.max(a + 1, (seg + 1) * window / wordLength);
                double avg = 0;
                for (int u = a; u < b; u++) {
                    avg += (s[t + u] - mean) / sd;
                }
                avg /= b - a;
                int sym = 0;
                while (sym < bp.length && avg > bp[sym]) {
                    sym++;
                }
                w = (w << bits) | sym;
            }
            out[t] = w;
        }
        return out;
    }
}
