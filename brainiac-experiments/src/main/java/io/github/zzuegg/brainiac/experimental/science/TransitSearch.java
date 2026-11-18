package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.tensor.spi.Fft;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Periodic transit search in light curves: detrending by a moving median or by a Gaussian
 * process ({@link GaussianProcess} with a Matérn-5/2 kernel on binned flux), box least squares
 * (Kovács, Zucker and Mazeh, A&A 2002) over a period–duration grid with the binned fold, the
 * signal detection efficiency (SDE) of the peak, folding to AstroNet's global and local views
 * (Shallue and Vanderburg, AJ 2018), and, as the baseline, the peak of the FFT power spectrum of
 * the detrended flux (core {@link Fft#transform}).
 */
public final class TransitSearch {

    /**
     * A BLS result.
     *
     * @param period the best period (days)
     * @param epoch the mid-time of a transit
     * @param duration the best duration (days)
     * @param depth the box depth
     * @param sde the signal detection efficiency of the peak
     */
    public record Result(double period, double epoch, double duration, double depth, double sde) {
    }

    private static final double[] DURATIONS = {0.04, 0.06, 0.08, 0.11, 0.15, 0.2, 0.27};

    private TransitSearch() {
    }

    /**
     * Moving-median detrending: each point divided by the median of the points within
     * {@code window / 2} days (evaluated every tenth of a window and interpolated), minus one.
     *
     * @param time times
     * @param flux fluxes
     * @param window the window in days
     * @return the relative residual flux
     */
    public static double[] detrendMedian(double[] time, double[] flux, double window) {
        int n = time.length;
        // medians at knots a tenth of the window apart, linearly interpolated
        double step = window / 10;
        int knots = (int) Math.ceil((time[n - 1] - time[0]) / step) + 1;
        double[] med = new double[knots];
        int lo = 0;
        int hi = 0;
        for (int k = 0; k < knots; k++) {
            double c = time[0] + k * step;
            while (lo < n && time[lo] < c - window / 2) {
                lo++;
            }
            while (hi < n && time[hi] <= c + window / 2) {
                hi++;
            }
            if (hi > lo) {
                double[] w = Arrays.copyOfRange(flux, lo, hi);
                Arrays.sort(w);
                med[k] = w[w.length / 2];
            } else {
                med[k] = Double.NaN;
            }
        }
        for (int k = 0; k < knots; k++) {
            if (Double.isNaN(med[k])) {
                med[k] = k > 0 ? med[k - 1] : 1;
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double u = (time[i] - time[0]) / step;
            int k = Math.min(knots - 2, (int) u);
            double a = u - k;
            out[i] = flux[i] / ((1 - a) * med[k] + a * med[k + 1]) - 1;
        }
        return out;
    }

    /**
     * Gaussian-process detrending: the flux is binned to {@code bin} days, a GP with a
     * Matérn-5/2 kernel of the given length scale is fitted to the bins, and its posterior mean
     * is divided out.
     *
     * @param time times
     * @param flux fluxes
     * @param lengthScale the kernel length scale in days
     * @param bin the bin width in days
     * @return the relative residual flux
     */
    public static double[] detrendGp(double[] time, double[] flux, double lengthScale, double bin) {
        List<double[]> bins = new ArrayList<>();
        int i = 0;
        double mean = 0;
        for (double f : flux) {
            mean += f / flux.length;
        }
        double var = 0;
        while (i < time.length) {
            double start = time[i];
            double s = 0;
            int c = 0;
            double[] vals = new double[time.length];
            while (i < time.length && time[i] < start + bin) {
                s += time[i];
                vals[c++] = flux[i];
                i++;
            }
            double[] v = Arrays.copyOf(vals, c);
            Arrays.sort(v);
            bins.add(new double[] {s / c, v[c / 2] - mean});
            var += (v[c / 2] - mean) * (v[c / 2] - mean);
        }
        var /= bins.size();
        double[][] x = new double[bins.size()][1];
        double[] y = new double[bins.size()];
        for (int k = 0; k < y.length; k++) {
            x[k][0] = bins.get(k)[0];
            y[k] = bins.get(k)[1];
        }
        GaussianProcess gp = GaussianProcess.fit(x, y, StationaryKernel.matern52(lengthScale, Math.max(var, 1e-12)),
                Math.max(var, 1e-12) * 0.01);
        double[][] xs = new double[time.length][1];
        for (int k = 0; k < time.length; k++) {
            xs[k][0] = time[k];
        }
        double[] trend = gp.predict(xs).mean();
        double[] out = new double[time.length];
        for (int k = 0; k < time.length; k++) {
            out[k] = flux[k] / (trend[k] + mean) - 1;
        }
        return out;
    }

    /**
     * Box least squares over periods {@code [pMin, pMax]} on a frequency grid whose step keeps
     * the phase drift over the baseline below the shortest duration, with each fold binned to
     * {@code bins} phase bins.
     *
     * @param time times
     * @param y the detrended flux
     * @param pMin the shortest period
     * @param pMax the longest period
     * @param bins phase bins per fold
     * @return the best box and its SDE
     */
    public static Result bls(double[] time, double[] y, double pMin, double pMax, int bins) {
        int n = time.length;
        double t0 = time[0];
        double span = time[n - 1] - t0;
        double mean = 0;
        for (double v : y) {
            mean += v / n;
        }
        double[] yc = new double[n];
        for (int i = 0; i < n; i++) {
            yc[i] = y[i] - mean;
        }
        List<Double> freqs = new ArrayList<>();
        // step: the phase drift over the baseline stays below the expected duration 0.05 P^(1/3)
        for (double f = 1 / pMax; f <= 1 / pMin; f += 0.05 * Math.pow(f, 2.0 / 3) / span) {
            freqs.add(f);
        }
        double[] power = new double[freqs.size()];
        double best = -1;
        Result bestR = null;
        double[] sumY = new double[bins];
        int[] count = new int[bins];
        for (int k = 0; k < power.length; k++) {
            double f = freqs.get(k);
            double period = 1 / f;
            Arrays.fill(sumY, 0);
            Arrays.fill(count, 0);
            for (int i = 0; i < n; i++) {
                double ph = (time[i] - t0) * f;
                ph -= Math.floor(ph);
                int b = (int) (ph * bins);
                sumY[b] += yc[i];
                count[b]++;
            }
            double kBest = 0;
            int argStart = 0;
            int argWidth = 1;
            for (double d : DURATIONS) {
                int w = Math.max(1, (int) Math.round(d / period * bins));
                if (w >= bins / 2) {
                    continue;
                }
                double s = 0;
                int c = 0;
                for (int b = 0; b < w; b++) {
                    s += sumY[b];
                    c += count[b];
                }
                for (int b = 0; b < bins; b++) {
                    if (c > 0 && c < n && s < 0) {
                        double r = c / (double) n;
                        double sr = s * s / (n * n * r * (1 - r));
                        if (sr > kBest) {
                            kBest = sr;
                            argStart = b;
                            argWidth = w;
                        }
                    }
                    int out = b;
                    int in = (b + w) % bins;
                    s += sumY[in] - sumY[out];
                    c += count[in] - count[out];
                }
            }
            power[k] = Math.sqrt(kBest);
            if (power[k] > best) {
                best = power[k];
                double mid = (argStart + argWidth / 2.0) / bins;
                double depth = 0;
                int cnt = 0;
                for (int b = argStart; b < argStart + argWidth; b++) {
                    depth -= sumY[b % bins];
                    cnt += count[b % bins];
                }
                bestR = new Result(period, t0 + mid * period, argWidth * period / bins, depth / Math.max(1, cnt), 0);
            }
        }
        double m = 0;
        for (double p : power) {
            m += p / power.length;
        }
        double sd = 0;
        for (double p : power) {
            sd += (p - m) * (p - m) / power.length;
        }
        sd = Math.sqrt(sd);
        return new Result(bestR.period(), bestR.epoch(), bestR.duration(), bestR.depth(), (best - m) / (sd + 1e-30));
    }

    /**
     * The FFT periodogram baseline: the detrended flux on a uniform grid of {@code cadence}
     * days (gaps zero-filled), zero-padded to a power of two, and the period of the highest
     * peak in {@code [pMin, pMax]}.
     *
     * @param time times
     * @param y the detrended flux
     * @param cadence the grid step in days
     * @param pMin the shortest period
     * @param pMax the longest period
     * @return the best period
     */
    public static double fftPeriod(double[] time, double[] y, double cadence, double pMin, double pMax) {
        int m = (int) Math.ceil((time[time.length - 1] - time[0]) / cadence) + 1;
        int n = Integer.highestOneBit(m - 1) << 2;
        float[] d = new float[2 * n];
        int[] c = new int[m];
        double[] s = new double[m];
        for (int i = 0; i < time.length; i++) {
            int k = (int) Math.round((time[i] - time[0]) / cadence);
            s[k] += y[i];
            c[k]++;
        }
        for (int k = 0; k < m; k++) {
            d[2 * k] = c[k] > 0 ? (float) (s[k] / c[k]) : 0;
        }
        Fft.transform(d, 1, n, false);
        double best = -1;
        double period = Double.NaN;
        for (int k = 1; k < n / 2; k++) {
            double f = k / (n * cadence);
            if (f < 1 / pMax || f > 1 / pMin) {
                continue;
            }
            double p = d[2 * k] * d[2 * k] + d[2 * k + 1] * d[2 * k + 1];
            if (p > best) {
                best = p;
                period = 1 / f;
            }
        }
        return period;
    }

    /**
     * AstroNet views: the flux folded on the ephemeris and median-binned, over the whole phase
     * ({@code global} bins) and over {@code +-2} durations ({@code local} bins), each shifted to
     * median 0 and scaled so the transit bottom is -1; empty bins are interpolated.
     *
     * @param time times
     * @param y the detrended flux
     * @param period the period
     * @param epoch a mid-transit time
     * @param duration the duration
     * @param global global bins
     * @param local local bins
     * @return {global view, local view}
     */
    public static double[][] views(double[] time, double[] y, double period, double epoch, double duration,
            int global, int local) {
        double[] phase = new double[time.length];
        for (int i = 0; i < time.length; i++) {
            double p = (time[i] - epoch) / period;
            phase[i] = p - Math.floor(p + 0.5);
        }
        double[] g = fold(phase, y, -0.5, 0.5, global);
        double half = Math.min(0.5, 2 * duration / period);
        double[] l = fold(phase, y, -half, half, local);
        double depth = Math.max(1e-9, -min(l));
        for (int i = 0; i < g.length; i++) {
            g[i] /= depth;
        }
        for (int i = 0; i < l.length; i++) {
            l[i] /= depth;
        }
        return new double[][] {g, l};
    }

    private static double min(double[] v) {
        double m = Double.POSITIVE_INFINITY;
        for (double x : v) {
            m = Math.min(m, x);
        }
        return m;
    }

    private static double[] fold(double[] phase, double[] y, double lo, double hi, int bins) {
        List<List<Double>> b = new ArrayList<>();
        for (int k = 0; k < bins; k++) {
            b.add(new ArrayList<>());
        }
        for (int i = 0; i < phase.length; i++) {
            if (phase[i] >= lo && phase[i] < hi) {
                int k = (int) ((phase[i] - lo) / (hi - lo) * bins);
                b.get(Math.min(bins - 1, k)).add(y[i]);
            }
        }
        double[] out = new double[bins];
        boolean[] has = new boolean[bins];
        List<Double> all = new ArrayList<>();
        for (int k = 0; k < bins; k++) {
            List<Double> v = b.get(k);
            if (!v.isEmpty()) {
                double[] a = v.stream().mapToDouble(Double::doubleValue).sorted().toArray();
                out[k] = a[a.length / 2];
                has[k] = true;
                all.add(out[k]);
            }
        }
        double med = all.isEmpty() ? 0 : all.stream().mapToDouble(Double::doubleValue).sorted().toArray()[all.size() / 2];
        for (int k = 0; k < bins; k++) {
            if (!has[k]) {
                int a = k - 1;
                while (a >= 0 && !has[a]) {
                    a--;
                }
                int c = k + 1;
                while (c < bins && !has[c]) {
                    c++;
                }
                out[k] = a >= 0 && c < bins ? (out[a] + out[c]) / 2 : a >= 0 ? out[a] : c < bins ? out[c] : med;
            }
        }
        for (int k = 0; k < bins; k++) {
            out[k] -= med;
        }
        return out;
    }
}
