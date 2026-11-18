package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;

/**
 * Classical monophonic pitch trackers, frame by frame on centred frames (frame {@code t} is
 * centred on sample {@code t hop}):
 *
 * <ul>
 *   <li>{@link Method#SPECTRAL_PEAK}: the strongest STFT bin between {@code fmin} and 1 kHz,
 *       refined by parabolic interpolation (a baseline that confuses strong upper harmonics
 *       with the fundamental).
 *   <li>{@link Method#HARMONIC_SUM}: the candidate on a 20-cent grid whose first five
 *       harmonics carry the most magnitude, weighted {@code 0.8^(k-1)} (a baseline).
 *   <li>{@link Method#AUTOCORRELATION}: the lag of the highest normalised autocorrelation
 *       peak; voiced when it exceeds 0.5.
 *   <li>{@link Method#YIN}: the squared difference function normalised by its cumulative
 *       mean (which removes the bias toward lag zero), the first dip below the absolute
 *       threshold 0.1 followed down to its local minimum, and parabolic refinement of the lag;
 *       voiced when the dip is below 0.25.
 *   <li>{@link Method#PYIN}: the probabilistic YIN of Mauch and Dixon: a Beta(2, 18) prior
 *       over 100 thresholds turns each frame's dips into F0 candidates with probabilities;
 *       an HMM over 20-cent pitch bins, each in a voiced and an unvoiced copy, with
 *       transitions limited to 240 cents and a 1% voicing switch probability, is decoded by
 *       {@link HiddenMarkovModel#viterbi(double[], double[][], double[][])} into a smooth,
 *       voicing-aware track.
 * </ul>
 *
 * <p>Papers: de Cheveigne, Kawahara, "YIN, a fundamental frequency estimator for speech and
 * music", JASA 111(4), 2002, <a href="https://doi.org/10.1121/1.1458024">doi:10.1121/1.1458024</a>;
 * Mauch, Dixon, "pYIN: a fundamental frequency estimator using probabilistic threshold
 * distributions", ICASSP 2014,
 * <a href="https://doi.org/10.1109/ICASSP.2014.6853678">doi:10.1109/ICASSP.2014.6853678</a>.
 */
public final class PitchTrackers {

    /** The trackers. */
    public enum Method {
        /** The strongest spectral peak. */
        SPECTRAL_PEAK,
        /** The best harmonic sum. */
        HARMONIC_SUM,
        /** Normalised autocorrelation. */
        AUTOCORRELATION,
        /** YIN. */
        YIN,
        /** Probabilistic YIN with HMM decoding. */
        PYIN
    }

    /**
     * A pitch track.
     *
     * @param f0 per frame a frequency in Hz (a candidate even where unvoiced, 0 if none)
     * @param voiced per frame the voicing decision
     */
    public record Track(float[] f0, boolean[] voiced) {
    }

    /** The integration window of the difference and autocorrelation functions. */
    public static final int WINDOW = 256;
    /** Bins per semitone of the pYIN HMM (20 cents). */
    public static final int BINS_PER_SEMITONE = 5;

    private PitchTrackers() {
    }

    /**
     * Tracks the pitch of a signal.
     *
     * @param method the tracker
     * @param x the signal
     * @param rate the sample rate
     * @param hop samples between frame centres
     * @param fmin the lowest pitch
     * @param fmax the highest pitch
     * @return the track, {@code x.length / hop + 1} frames
     */
    public static Track track(Method method, float[] x, int rate, int hop, double fmin, double fmax) {
        return switch (method) {
            case SPECTRAL_PEAK, HARMONIC_SUM -> spectral(method, x, rate, hop, fmin, fmax);
            case AUTOCORRELATION -> autocorrelation(x, rate, hop, fmin, fmax);
            case YIN -> yin(x, rate, hop, fmin, fmax);
            case PYIN -> pyin(x, rate, hop, fmin, fmax);
        };
    }

    /** Voicing by an energy gate: above 10% of the 95th-percentile frame RMS. */
    private static boolean[] energyGate(float[] x, int hop) {
        float[] e = MusicFeatures.energy(x, WINDOW, hop);
        float[] sorted = e.clone();
        java.util.Arrays.sort(sorted);
        float ref = sorted[(int) (0.95 * (sorted.length - 1))];
        boolean[] v = new boolean[e.length];
        for (int i = 0; i < e.length; i++) {
            v[i] = e[i] > 0.1f * ref && ref > 0;
        }
        return v;
    }

    private static Track spectral(Method method, float[] x, int rate, int hop, double fmin, double fmax) {
        int nFft = 1024;
        float[][] mag = MusicFeatures.magnitude(x, nFft, hop);
        double binHz = rate / (double) nFft;
        float[] f0 = new float[mag.length];
        for (int t = 0; t < mag.length; t++) {
            float[] m = mag[t];
            if (method == Method.SPECTRAL_PEAK) {
                int lo = (int) Math.ceil(fmin / binHz);
                int hi = Math.min(m.length - 2, (int) (Math.min(1000, fmax * 2) / binHz));
                int best = lo;
                for (int k = lo; k <= hi; k++) {
                    if (m[k] > m[best]) {
                        best = k;
                    }
                }
                f0[t] = (float) (parabolic(m[best - 1], m[best], m[best + 1], best) * binHz);
            } else {
                double bestS = -1;
                double bestF = 0;
                for (double f = fmin; f <= fmax; f *= Math.pow(2, 0.2 / 12)) {
                    double s = 0;
                    for (int k = 1; k <= 5; k++) {
                        double p = k * f / binHz;
                        int a = (int) p;
                        if (a + 1 >= m.length) {
                            break;
                        }
                        double fr = p - a;
                        s += Math.pow(0.8, k - 1) * (m[a] * (1 - fr) + m[a + 1] * fr);
                    }
                    if (s > bestS) {
                        bestS = s;
                        bestF = f;
                    }
                }
                f0[t] = (float) bestF;
            }
        }
        return new Track(f0, energyGate(x, hop));
    }

    private static double parabolic(double a, double b, double c, double x) {
        double den = a - 2 * b + c;
        return Math.abs(den) < 1e-12 ? x : x + 0.5 * (a - c) / den;
    }

    private static float[] frameAt(float[] x, int centre, int maxLag) {
        float[] f = new float[WINDOW + maxLag + 2];
        int start = centre - WINDOW / 2;
        for (int j = 0; j < f.length; j++) {
            int n = start + j;
            f[j] = n >= 0 && n < x.length ? x[n] : 0;
        }
        return f;
    }

    private static Track autocorrelation(float[] x, int rate, int hop, double fmin, double fmax) {
        int maxLag = (int) Math.ceil(rate / fmin);
        int minLag = (int) Math.floor(rate / fmax);
        int frames = x.length / hop + 1;
        float[] f0 = new float[frames];
        boolean[] voiced = new boolean[frames];
        for (int t = 0; t < frames; t++) {
            float[] f = frameAt(x, t * hop, maxLag);
            double e0 = 0;
            for (int j = 0; j < WINDOW; j++) {
                e0 += f[j] * f[j];
            }
            double[] r = new double[maxLag + 2];
            for (int tau = minLag - 1; tau <= maxLag + 1; tau++) {
                double s = 0;
                double e1 = 0;
                for (int j = 0; j < WINDOW; j++) {
                    s += f[j] * f[j + tau];
                    e1 += f[j + tau] * f[j + tau];
                }
                r[tau] = s / Math.sqrt(e0 * e1 + 1e-12);
            }
            int best = minLag;
            for (int tau = minLag; tau <= maxLag; tau++) {
                if (r[tau] > r[best]) {
                    best = tau;
                }
            }
            double lag = parabolic(r[best - 1], r[best], r[best + 1], best);
            f0[t] = (float) (rate / lag);
            voiced[t] = r[best] > 0.5;
        }
        return new Track(f0, voiced);
    }

    /**
     * The cumulative-mean-normalised difference function of one frame.
     *
     * @param x the signal
     * @param centre the frame centre
     * @param maxLag the largest lag
     * @return {@code d'(tau)} for {@code tau = 0..maxLag + 1}
     */
    public static double[] cmndf(float[] x, int centre, int maxLag) {
        float[] f = frameAt(x, centre, maxLag);
        double[] d = new double[maxLag + 2];
        double running = 0;
        d[0] = 1;
        for (int tau = 1; tau <= maxLag + 1; tau++) {
            double s = 0;
            for (int j = 0; j < WINDOW; j++) {
                double diff = f[j] - f[j + tau];
                s += diff * diff;
            }
            running += s;
            d[tau] = running < 1e-12 ? 1 : s * tau / running;
        }
        return d;
    }

    private static Track yin(float[] x, int rate, int hop, double fmin, double fmax) {
        int maxLag = (int) Math.ceil(rate / fmin);
        int minLag = Math.max(2, (int) Math.floor(rate / fmax));
        int frames = x.length / hop + 1;
        float[] f0 = new float[frames];
        boolean[] voiced = new boolean[frames];
        for (int t = 0; t < frames; t++) {
            double[] d = cmndf(x, t * hop, maxLag);
            int tau = -1;
            for (int k = minLag; k <= maxLag; k++) {
                if (d[k] < 0.1) {
                    while (k + 1 <= maxLag && d[k + 1] < d[k]) {
                        k++;
                    }
                    tau = k;
                    break;
                }
            }
            if (tau < 0) {
                tau = minLag;
                for (int k = minLag; k <= maxLag; k++) {
                    if (d[k] < d[tau]) {
                        tau = k;
                    }
                }
            }
            double lag = parabolic(d[tau - 1], d[tau], d[tau + 1], tau);
            f0[t] = (float) (rate / lag);
            voiced[t] = d[tau] < 0.25;
        }
        return new Track(f0, voiced);
    }

    /** The Beta(2, 18) prior over thresholds 0.01..1.00. */
    private static double[] thresholdPrior() {
        double[] p = new double[100];
        double sum = 0;
        for (int i = 0; i < 100; i++) {
            double s = (i + 0.5) / 100;
            p[i] = s * Math.pow(1 - s, 17);
            sum += p[i];
        }
        for (int i = 0; i < 100; i++) {
            p[i] /= sum;
        }
        return p;
    }

    /**
     * The pYIN F0 candidates of one frame: for every threshold the first dip below it (or
     * with 1% weight the global minimum when none is below), weighted by the threshold prior.
     *
     * @param d the cumulative-mean-normalised difference
     * @param minLag the smallest lag
     * @param maxLag the largest lag
     * @return per lag its probability of being the period (sums to at most 1)
     */
    public static double[] pyinCandidates(double[] d, int minLag, int maxLag) {
        double[] prior = thresholdPrior();
        double[] p = new double[maxLag + 2];
        java.util.List<Integer> troughs = new java.util.ArrayList<>();
        for (int k = minLag; k <= maxLag; k++) {
            if (d[k] <= d[k - 1] && d[k] <= d[k + 1]) {
                troughs.add(k);
            }
        }
        int global = minLag;
        for (int k = minLag; k <= maxLag; k++) {
            if (d[k] < d[global]) {
                global = k;
            }
        }
        for (int i = 0; i < 100; i++) {
            double s = (i + 1) / 100.0;
            int found = -1;
            for (int k : troughs) {
                if (d[k] < s) {
                    found = k;
                    break;
                }
            }
            if (found >= 0) {
                p[found] += prior[i];
            } else {
                p[global] += 0.01 * prior[i];
            }
        }
        return p;
    }

    private static Track pyin(float[] x, int rate, int hop, double fmin, double fmax) {
        int maxLag = (int) Math.ceil(rate / fmin);
        int minLag = Math.max(2, (int) Math.floor(rate / fmax));
        int frames = x.length / hop + 1;
        int bins = (int) Math.ceil(12 * BINS_PER_SEMITONE * Math.log(fmax / fmin) / Math.log(2)) + 1;
        int states = 2 * bins;
        double[][] logEmit = new double[frames][states];
        double[][] bestLag = new double[frames][bins];
        for (int t = 0; t < frames; t++) {
            double[] d = cmndf(x, t * hop, maxLag);
            double[] cand = pyinCandidates(d, minLag, maxLag);
            double[] binP = new double[bins];
            double[] binBest = new double[bins];
            double total = 0;
            for (int k = minLag; k <= maxLag; k++) {
                if (cand[k] <= 0) {
                    continue;
                }
                double lag = parabolic(d[k - 1], d[k], d[k + 1], k);
                double f = rate / lag;
                int b = (int) Math.round(12 * BINS_PER_SEMITONE * Math.log(f / fmin) / Math.log(2));
                if (b < 0 || b >= bins) {
                    continue;
                }
                if (cand[k] > binP[b]) {
                    binBest[b] = lag;
                }
                binP[b] += cand[k];
                total += cand[k];
            }
            bestLag[t] = binBest;
            double unvoiced = Math.max(0, 1 - total) / bins;
            for (int b = 0; b < bins; b++) {
                logEmit[t][b] = Math.log(binP[b] * 0.5 + 1e-7);
                logEmit[t][bins + b] = Math.log(unvoiced * 0.5 + 1e-7);
            }
        }
        int jump = 12;
        double[][] logTrans = new double[states][states];
        for (double[] row : logTrans) {
            java.util.Arrays.fill(row, Double.NEGATIVE_INFINITY);
        }
        double stay = 0.99;
        for (int b = 0; b < bins; b++) {
            double norm = 0;
            for (int c = Math.max(0, b - jump); c <= Math.min(bins - 1, b + jump); c++) {
                norm += jump + 1 - Math.abs(c - b);
            }
            for (int c = Math.max(0, b - jump); c <= Math.min(bins - 1, b + jump); c++) {
                double w = (jump + 1 - Math.abs(c - b)) / norm;
                logTrans[b][c] = Math.log(stay * w);
                logTrans[b][bins + c] = Math.log((1 - stay) * w);
                logTrans[bins + b][bins + c] = Math.log(stay * w);
                logTrans[bins + b][c] = Math.log((1 - stay) * w);
            }
        }
        double[] logStart = new double[states];
        java.util.Arrays.fill(logStart, -Math.log(states));
        int[] path = HiddenMarkovModel.viterbi(logStart, logTrans, logEmit);
        float[] f0 = new float[frames];
        boolean[] voiced = new boolean[frames];
        for (int t = 0; t < frames; t++) {
            int s = path[t];
            int b = s % bins;
            voiced[t] = s < bins;
            double lag = bestLag[t][b];
            f0[t] = (float) (lag > 0 ? rate / lag : fmin * Math.pow(2, b / (12.0 * BINS_PER_SEMITONE)));
        }
        return new Track(f0, voiced);
    }
}
