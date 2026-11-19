package io.github.zzuegg.brainiac.experiments.sensing;

import io.github.zzuegg.brainiac.experimental.data.ArraySignals;
import io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.Method;
import io.github.zzuegg.brainiac.experimental.timeseries.Arima;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Classical spectral estimation (wave 13, item 13.4.16): log-spectral distance to the true AR(2)
 * spectrum, frequency RMSE of a line, resolution of two close lines from a short record (against the
 * raw periodogram and an {@link Arima}-fitted spectrum), and detection of a weak line on irregularly
 * sampled data by Lomb-Scargle against interpolate-then-FFT.
 */
public final class SpectralEstimation {

    private static final double[] AR = {0.75, -0.5};

    private SpectralEstimation() {
    }

    private static double[] estimate(String name, double[] x, int nfft) {
        return switch (name) {
            case "periodogram" -> io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.psd(
                    Method.PERIODOGRAM, x, nfft, 0);
            case "Welch (segments n/4)" -> io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.psd(
                    Method.WELCH, x, nfft, x.length / 4);
            case "multitaper (NW=4)" -> io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.psd(
                    Method.MULTITAPER, x, nfft, 4);
            default -> io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.psd(Method.BURG, x, nfft,
                    20);
        };
    }

    private static double[] arimaSpectrum(double[] x, int nfft) {
        double mean = 0;
        for (double v : x) {
            mean += v / x.length;
        }
        double[] c = new double[x.length];
        for (int i = 0; i < c.length; i++) {
            c[i] = x[i] - mean;
        }
        Arima fit = Arima.fit(c, 4, 0, 0, 0, 0, 30);
        double[] phi = fit.ar();
        double[] theta = fit.ma();
        double s2 = fit.sigma() * fit.sigma();
        double[] out = new double[nfft / 2 + 1];
        for (int k = 0; k < out.length; k++) {
            double w = 2 * Math.PI * k / nfft;
            double ar = 1;
            double ai = 0;
            for (int j = 0; j < phi.length; j++) {
                ar -= phi[j] * Math.cos(w * (j + 1));
                ai += phi[j] * Math.sin(w * (j + 1));
            }
            double mr = 1;
            double mi = 0;
            for (int j = 0; j < theta.length; j++) {
                mr += theta[j] * Math.cos(w * (j + 1));
                mi -= theta[j] * Math.sin(w * (j + 1));
            }
            out[k] = s2 * (mr * mr + mi * mi) / (ar * ar + ai * ai);
        }
        return out;
    }

    private static int[] topPeaks(double[] p, int count, int from, int to) {
        int[] best = new int[count];
        java.util.Arrays.fill(best, -1);
        for (int i = Math.max(from, 1); i < Math.min(to, p.length - 1); i++) {
            if (p[i] > p[i - 1] && p[i] >= p[i + 1]) {
                for (int j = 0; j < count; j++) {
                    if (best[j] < 0 || p[i] > p[best[j]]) {
                        System.arraycopy(best, j, best, j + 1, count - j - 1);
                        best[j] = i;
                        break;
                    }
                }
            }
        }
        return best;
    }

    /**
     * The table.
     *
     * @param trials the trials per cell
     * @param seed the seed
     * @return rows: "lsd / method" {log-spectral distance dB at n = 256 on pure AR(2) noise, at n = 64};
     *     "freq / method" {line-frequency RMSE at n = 128 (in 1/n units)}; "resolve / method" {probability of
     *     resolving lines 0.8/n apart from n = 64}; "irregular / method" {weak-line detection AUC, RMSE of the
     *     strong line's frequency}; "F-test" {weak-line detection AUC on regular data}
     */
    public static Map<String, double[]> table(int trials, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"periodogram", "Welch (segments n/4)", "multitaper (NW=4)", "Burg (AIC, p<=20)"};
        String arima = "ARIMA(4,0,0) fit (first 5 trials)";
        int arimaTrials = Math.min(trials, 5);
        int nfft = 1024;
        out.put("lsd / " + arima, new double[2]);
        for (String name : names) {
            out.put("lsd / " + name, new double[2]);
            out.put("freq / " + name, new double[1]);
            out.put("resolve / " + name, new double[1]);
        }
        int[] lens = {256, 64};
        for (int t = 0; t < trials; t++) {
            for (int li = 0; li < lens.length; li++) {
                ArraySignals.Sinusoids s = ArraySignals.sinusoids(lens[li], new double[0], new double[0], AR, 1, 0, r);
                for (int m = 0; m <= names.length; m++) {
                    if (m == names.length && t >= arimaTrials) {
                        continue;
                    }
                    double[] p = m == names.length ? arimaSpectrum(s.x(), nfft) : estimate(names[m], s.x(), nfft);
                    double d = 0;
                    for (int k = 1; k < p.length - 1; k++) {
                        double db = 10 * Math.log10(Math.max(p[k], 1e-30) / s.noisePsd(k / (double) nfft));
                        d += db * db / (p.length - 2);
                    }
                    out.get("lsd / " + (m == names.length ? arima : names[m]))[li] += Math.sqrt(d)
                            / (m == names.length ? arimaTrials : trials);
                }
            }
            double f0 = 0.1 + 0.25 * r.nextDouble();
            ArraySignals.Sinusoids line = ArraySignals.sinusoids(128, new double[] {f0}, new double[] {1.5}, AR, 1, 0,
                    r);
            double f1 = 0.15 + 0.1 * r.nextDouble();
            double gap = 0.8 / 64;
            ArraySignals.Sinusoids pair = ArraySignals.sinusoids(64, new double[] {f1, f1 + gap},
                    new double[] {1, 1}, AR, 0.3, 0, r);
            for (String name : names) {
                double[] p = estimate(name, line.x(), nfft);
                int pk = topPeaks(p, 1, (int) (0.05 * nfft), (int) (0.45 * nfft))[0];
                double err = (pk / (double) nfft - f0) * 128;
                out.get("freq / " + name)[0] += err * err / trials;
                double[] q = estimate(name, pair.x(), nfft);
                int[] two = topPeaks(q, 2, (int) ((f1 - 4 * gap) * nfft), (int) ((f1 + 5 * gap) * nfft));
                boolean ok = two[1] >= 0;
                if (ok) {
                    double a = Math.min(two[0], two[1]) / (double) nfft;
                    double b = Math.max(two[0], two[1]) / (double) nfft;
                    ok = Math.abs(a - f1) < gap / 2 && Math.abs(b - f1 - gap) < gap / 2;
                }
                out.get("resolve / " + name)[0] += (ok ? 1.0 : 0) / trials;
            }
        }
        for (String name : names) {
            double[] row = out.get("freq / " + name);
            row[0] = Math.sqrt(row[0]);
        }
        irregular(out, trials, r);
        return out;
    }

    private static void irregular(Map<String, double[]> out, int trials, SplittableRandom r) {
        double fWeak = 0.23;
        double fStrong = 0.08;
        double[] freqs = new double[600];
        for (int i = 0; i < freqs.length; i++) {
            freqs[i] = (i + 1) * 0.5 / freqs.length;
        }
        double[][] scores = new double[3][2 * trials];
        double[] strongErr = new double[2];
        for (int t = 0; t < 2 * trials; t++) {
            boolean present = t < trials;
            double[] lines = present ? new double[] {fStrong, fWeak} : new double[] {fStrong};
            double[] amps = present ? new double[] {1, 0.3} : new double[] {1};
            ArraySignals.Sinusoids s = ArraySignals.sinusoids(160, lines, amps, new double[] {0, 0}, 0.7, 0.3, r);
            double[] ls = io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.lombScargle(s.t(), s.x(),
                    freqs);
            double[] it = interpolatedFft(s.t(), s.x(), freqs);
            scores[0][t] = bandMax(ls, freqs, fWeak) / median(ls);
            scores[1][t] = bandMax(it, freqs, fWeak) / median(it);
            if (present) {
                strongErr[0] += sq(freqs[argmax(ls)] - fStrong) / trials;
                strongErr[1] += sq(freqs[argmax(it)] - fStrong) / trials;
            }
            ArraySignals.Sinusoids reg = ArraySignals.sinusoids(160, lines, amps, new double[] {0, 0}, 0.7, 0, r);
            double[] f = io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.fTest(reg.x(), 4, 1024);
            double m = 0;
            for (int k = (int) ((fWeak - 0.004) * 1024); k <= (int) ((fWeak + 0.004) * 1024); k++) {
                m = Math.max(m, f[k]);
            }
            scores[2][t] = m;
        }
        out.put("irregular / Lomb-Scargle", new double[] {auc(scores[0], trials), Math.sqrt(strongErr[0])});
        out.put("irregular / linear interpolation then FFT", new double[] {auc(scores[1], trials),
            Math.sqrt(strongErr[1])});
        out.put("F-test (multitaper, regular sampling)", new double[] {auc(scores[2], trials)});
    }

    private static double sq(double v) {
        return v * v;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[b]) {
                b = i;
            }
        }
        return b;
    }

    private static double bandMax(double[] p, double[] freqs, double f) {
        double m = 0;
        for (int i = 0; i < freqs.length; i++) {
            if (Math.abs(freqs[i] - f) <= 0.004) {
                m = Math.max(m, p[i]);
            }
        }
        return m;
    }

    private static double median(double[] v) {
        double[] c = v.clone();
        java.util.Arrays.sort(c);
        return Math.max(c[c.length / 2], 1e-30);
    }

    private static double auc(double[] s, int positives) {
        double a = 0;
        for (int i = 0; i < positives; i++) {
            for (int j = positives; j < s.length; j++) {
                a += s[i] > s[j] ? 1 : s[i] == s[j] ? 0.5 : 0;
            }
        }
        return a / ((double) positives * (s.length - positives));
    }

    /** Linear interpolation onto the unit grid, then the periodogram at the given frequencies. */
    private static double[] interpolatedFft(double[] t, double[] x, double[] freqs) {
        int n = (int) Math.round(t[t.length - 1] - t[0]) + 1;
        double[] g = new double[n];
        int j = 0;
        for (int i = 0; i < n; i++) {
            double ti = t[0] + i;
            while (j < t.length - 2 && t[j + 1] < ti) {
                j++;
            }
            double w = (ti - t[j]) / (t[j + 1] - t[j]);
            w = Math.max(0, Math.min(1, w));
            g[i] = (1 - w) * x[j] + w * x[j + 1];
        }
        int nfft = Integer.highestOneBit(n - 1) << 2;
        double[] p = io.github.zzuegg.brainiac.experimental.sensing.SpectralEstimation.psd(Method.PERIODOGRAM, g,
                nfft, 0);
        double[] out = new double[freqs.length];
        for (int i = 0; i < freqs.length; i++) {
            out[i] = p[(int) Math.round(freqs[i] * nfft)];
        }
        return out;
    }
}
