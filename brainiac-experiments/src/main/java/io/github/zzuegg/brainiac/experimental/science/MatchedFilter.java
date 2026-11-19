package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.tensor.spi.Fft;

import java.util.ArrayList;
import java.util.List;

/**
 * Matched filtering for chirps in coloured Gaussian noise (Allen et al., "FINDCHIRP", PRD
 * 2012): the noise spectrum is estimated by Welch's averaged periodograms, data and templates
 * are whitened by it, and each template of a bank, placed by a minimal-match criterion over the
 * chirp duration {@code tau} (the chirp-mass proxy), is correlated with the data by FFT. The
 * template is complex (its 0 and 90 degree phases), so the modulus maximises over the phase;
 * the detection statistic is the largest SNR over templates and times. A chi-squared veto splits
 * the template into {@code p} pieces of equal power and tests that each contributes its share;
 * the re-weighted SNR {@code rho / [(1 + chi_r^3) / 2]^(1/6)} suppresses loud glitches. Also an
 * excess-power detector (band-limited sliding-window energy of the whitened data) and a 1-D CNN
 * on whitened data ({@link CnnDetector}, as in George and Huerta, PRD 2018). FFTs are core
 * {@link Fft#transform}.
 */
public final class MatchedFilter {

    /**
     * A detection.
     *
     * @param snr the peak matched-filter SNR
     * @param reweighted the largest chi-squared re-weighted SNR over the bank
     * @param chi2r the reduced chi-squared of that template
     * @param tau the duration of the template with the largest SNR
     * @param shift its time shift (samples)
     */
    public record Detection(double snr, double reweighted, double chi2r, double tau, int shift) {
    }

    private final int n;
    private final double rate;
    private final double[] psd;
    private final double f0;
    private final double fMax;
    private final List<double[]> bankRe = new ArrayList<>();
    private final List<double[]> bankIm = new ArrayList<>();
    private final List<double[][]> bankTime = new ArrayList<>();
    private final List<Double> taus = new ArrayList<>();
    private int pieces = 4;

    /**
     * A filter for segments of {@code n} samples.
     *
     * @param psd the two-sided noise power per FFT bin, {@code E|X_k|^2 / n}, of length {@code n}
     * @param rate the sampling rate
     * @param f0 the chirps' starting frequency
     * @param fMax their cut-off frequency
     */
    public MatchedFilter(double[] psd, double rate, double f0, double fMax) {
        this.n = psd.length;
        this.psd = psd.clone();
        this.rate = rate;
        this.f0 = f0;
        this.fMax = fMax;
    }

    /**
     * Welch's estimate: Hann-windowed periodograms of segments of {@code n} with 50% overlap.
     *
     * @param noise a noise stretch, at least {@code n} long
     * @param n the segment length (a power of two)
     * @return the power per bin, {@code E|X_k|^2 / n} for unit-variance white noise equal to 1
     */
    public static double[] welch(double[] noise, int n) {
        double[] w = new double[n];
        double w2 = 0;
        for (int i = 0; i < n; i++) {
            w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
            w2 += w[i] * w[i];
        }
        double[] s = new double[n];
        int count = 0;
        for (int start = 0; start + n <= noise.length; start += n / 2) {
            double[] re = new double[n];
            double[] im = new double[n];
            for (int i = 0; i < n; i++) {
                re[i] = noise[start + i] * w[i];
            }
            fft(re, im, false);
            for (int k = 0; k < n; k++) {
                s[k] += re[k] * re[k] + im[k] * im[k];
            }
            count++;
        }
        for (int k = 0; k < n; k++) {
            s[k] /= count * w2;
        }
        return s;
    }

    /**
     * The exact spectrum of unit-variance AR(1) noise, {@code (1 - phi^2) / |1 - phi e^(-i w)|^2}.
     *
     * @param n bins
     * @param phi the AR coefficient
     * @return the power per bin
     */
    public static double[] ar1Psd(int n, double phi) {
        double[] s = new double[n];
        for (int k = 0; k < n; k++) {
            double w = 2 * Math.PI * k / n;
            s[k] = (1 - phi * phi) / (1 - 2 * phi * Math.cos(w) + phi * phi);
        }
        return s;
    }

    static void fft(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        float[] d = new float[2 * n];
        for (int i = 0; i < n; i++) {
            d[2 * i] = (float) re[i];
            d[2 * i + 1] = (float) im[i];
        }
        Fft.transform(d, 1, n, inverse);
        for (int i = 0; i < n; i++) {
            re[i] = d[2 * i];
            im[i] = d[2 * i + 1];
        }
    }

    /**
     * Whitens a segment: divides its spectrum by the square root of the noise spectrum.
     *
     * @param x a segment of {@code n}
     * @return the whitened segment (unit-variance noise)
     */
    public double[] whiten(double[] x) {
        double[] re = x.clone();
        double[] im = new double[n];
        fft(re, im, false);
        for (int k = 0; k < n; k++) {
            double s = Math.sqrt(psd[k]);
            re[k] /= s;
            im[k] /= s;
        }
        fft(re, im, true);
        return re;
    }

    /**
     * The whitened norm of a waveform, its optimal SNR in this noise.
     *
     * @param h a waveform of {@code n}
     * @return {@code ||whiten(h)||}
     */
    public double optimalSnr(double[] h) {
        double s = 0;
        for (double v : whiten(h)) {
            s += v * v;
        }
        return Math.sqrt(s);
    }

    private double[][] whitenedPair(double tau) {
        double[] a = whiten(ScienceSims.chirp(n, rate, tau, f0, fMax, 0, 0));
        double[] b = whiten(ScienceSims.chirp(n, rate, tau, f0, fMax, 0, Math.PI / 2));
        double na = norm(a);
        double dot = 0;
        for (int i = 0; i < n; i++) {
            a[i] /= na;
            dot += a[i] * b[i];
        }
        for (int i = 0; i < n; i++) {
            b[i] -= dot * a[i];
        }
        double nb = norm(b);
        for (int i = 0; i < n; i++) {
            b[i] /= nb;
        }
        return new double[][] {a, b};
    }

    private static double norm(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x * x;
        }
        return Math.sqrt(s);
    }

    /**
     * The match of two templates: the largest normalised complex overlap over time shifts.
     */
    private double match(double[][] a, double[][] b) {
        double[] re = a[0].clone();
        double[] im = a[1].clone();
        fft(re, im, false);
        double[] re2 = b[0].clone();
        double[] im2 = b[1].clone();
        fft(re2, im2, false);
        double[] cr = new double[n];
        double[] ci = new double[n];
        for (int k = 0; k < n; k++) {
            cr[k] = re[k] * re2[k] + im[k] * im2[k];
            ci[k] = im[k] * re2[k] - re[k] * im2[k];
        }
        fft(cr, ci, true);
        double best = 0;
        for (int i = 0; i < n; i++) {
            best = Math.max(best, Math.hypot(cr[i], ci[i]));
        }
        // complex templates have norm sqrt(2)
        return best / 2;
    }

    /**
     * Places a bank over {@code [tauMin, tauMax]}: a template is added whenever the match with
     * the last one falls below {@code minimalMatch} on a fine grid.
     *
     * @param tauMin the shortest duration
     * @param tauMax the longest duration
     * @param minimalMatch such as 0.97
     * @return this filter
     */
    public MatchedFilter placeBank(double tauMin, double tauMax, double minimalMatch) {
        bankRe.clear();
        bankIm.clear();
        bankTime.clear();
        taus.clear();
        double[][] last = whitenedPair(tauMin);
        add(tauMin, last);
        int grid = 400;
        double prevTau = tauMin;
        double[][] prev = last;
        for (int g = 1; g <= grid; g++) {
            double tau = tauMin * Math.pow(tauMax / tauMin, g / (double) grid);
            double[][] cand = whitenedPair(tau);
            if (match(last, cand) < minimalMatch) {
                add(prevTau, prev);
                last = prev;
            }
            prevTau = tau;
            prev = cand;
        }
        if (taus.get(taus.size() - 1) < prevTau) {
            add(prevTau, prev);
        }
        return this;
    }

    private void add(double tau, double[][] pair) {
        if (!taus.isEmpty() && taus.get(taus.size() - 1) == tau) {
            return;
        }
        double[] re = pair[0].clone();
        double[] im = pair[1].clone();
        fft(re, im, false);
        bankRe.add(re);
        bankIm.add(im);
        bankTime.add(pair);
        taus.add(tau);
    }

    /** @return the bank's durations */
    public double[] bank() {
        return taus.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * @param p chi-squared pieces (default 4)
     * @return this filter
     */
    public MatchedFilter pieces(int p) {
        this.pieces = p;
        return this;
    }

    /**
     * Searches a segment.
     *
     * @param x the raw segment
     * @return the loudest trigger over the bank
     */
    public Detection search(double[] x) {
        double[] w = whiten(x);
        double[] dr = w.clone();
        double[] di = new double[n];
        fft(dr, di, false);
        double maxPeak = -1;
        double tauAtPeak = Double.NaN;
        int shiftAtPeak = 0;
        double maxRw = -1;
        double chiAtRw = 0;
        for (int t = 0; t < taus.size(); t++) {
            double[] hr = bankRe.get(t);
            double[] hi = bankIm.get(t);
            double[] cr = new double[n];
            double[] ci = new double[n];
            // correlation with h = a + i b: sum_t d(t) conj(h(t - s))
            for (int k = 0; k < n; k++) {
                cr[k] = dr[k] * hr[k] + di[k] * hi[k];
                ci[k] = di[k] * hr[k] - dr[k] * hi[k];
            }
            fft(cr, ci, true);
            int arg = 0;
            double peak = 0;
            for (int i = 0; i < n; i++) {
                double v = Math.hypot(cr[i], ci[i]);
                if (v > peak) {
                    peak = v;
                    arg = i;
                }
            }
            double chi = chi2r(w, bankTime.get(t), arg);
            double rw = chi <= 1 ? peak : peak / Math.pow((1 + chi * chi * chi) / 2, 1.0 / 6);
            if (peak > maxPeak) {
                maxPeak = peak;
                tauAtPeak = taus.get(t);
                shiftAtPeak = arg;
            }
            if (rw > maxRw) {
                maxRw = rw;
                chiAtRw = chi;
            }
        }
        return new Detection(maxPeak, maxRw, chiAtRw, tauAtPeak, shiftAtPeak);
    }

    private double chi2r(double[] w, double[][] h, int shift) {
        // split the template into pieces of equal power
        double total = 0;
        for (int i = 0; i < n; i++) {
            total += h[0][i] * h[0][i];
        }
        double[] zr = new double[pieces];
        double[] zi = new double[pieces];
        double acc = 0;
        int piece = 0;
        for (int i = 0; i < n; i++) {
            acc += h[0][i] * h[0][i];
            int j = (i + shift) % n;
            zr[piece] += w[j] * h[0][i];
            zi[piece] += w[j] * h[1][i];
            if (acc > total * (piece + 1) / pieces && piece < pieces - 1) {
                piece++;
            }
        }
        double sr = 0;
        double si = 0;
        for (int k = 0; k < pieces; k++) {
            sr += zr[k];
            si += zi[k];
        }
        double chi = 0;
        for (int k = 0; k < pieces; k++) {
            double er = zr[k] - sr / pieces;
            double ei = zi[k] - si / pieces;
            chi += pieces * (er * er + ei * ei);
        }
        return chi / (2 * pieces - 2);
    }

    /**
     * The excess-power statistic: the largest energy of the whitened, band-passed data over
     * sliding windows, standardised by the window length.
     *
     * @param x the raw segment
     * @param window the window in samples
     * @return {@code max (E - window) / sqrt(2 window)}
     */
    public double excessPower(double[] x, int window) {
        double[] re = whiten(x);
        double[] im = new double[n];
        fft(re, im, false);
        int passed = 0;
        for (int k = 0; k < n; k++) {
            double f = Math.min(k, n - k) * rate / n;
            if (f < f0 || f > fMax) {
                re[k] = 0;
                im[k] = 0;
            } else {
                passed++;
            }
        }
        fft(re, im, true);
        double scale = n / (double) passed;
        double[] c = new double[n + 1];
        for (int i = 0; i < n; i++) {
            c[i + 1] = c[i] + re[i] * re[i] * scale;
        }
        double best = Double.NEGATIVE_INFINITY;
        for (int i = window; i <= n; i++) {
            best = Math.max(best, (c[i] - c[i - window] - window) / Math.sqrt(2 * window));
        }
        return best;
    }

    /** A 1-D CNN detector on whitened, standardised segments. */
    public static final class CnnDetector {

        private final Cnn1d net;

        /**
         * An untrained detector.
         *
         * @param seed the seed
         */
        public CnnDetector(long seed) {
            this.net = new Cnn1d(1, new int[] {16, 16, 16, 32}, 15, seed);
        }

        private static float[][][] input(double[][] whitened) {
            float[][][] x = new float[whitened.length][1][];
            for (int i = 0; i < whitened.length; i++) {
                x[i][0] = new float[whitened[i].length];
                for (int j = 0; j < whitened[i].length; j++) {
                    x[i][0][j] = (float) whitened[i][j];
                }
            }
            return x;
        }

        /**
         * Trains on labelled whitened segments.
         *
         * @param whitened segments
         * @param labels 1 for a chirp, 0 for noise
         * @param steps Adam steps
         * @param batch the batch
         * @param lr the step
         * @param seed the seed
         * @return this detector
         */
        public CnnDetector train(double[][] whitened, int[] labels, int steps, int batch, double lr, long seed) {
            net.train(input(whitened), labels, steps, batch, lr, seed);
            return this;
        }

        /**
         * @param whitened segments
         * @return the logits
         */
        public double[] scores(double[][] whitened) {
            return net.scores(input(whitened));
        }
    }
}
