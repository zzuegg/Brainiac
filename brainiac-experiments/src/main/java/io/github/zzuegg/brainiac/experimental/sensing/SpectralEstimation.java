package io.github.zzuegg.brainiac.experimental.sensing;

import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.spi.Fft;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Classical power-spectral-density estimators for a real series with unit sampling, all returning
 * the two-sided PSD per unit frequency on the grid {@code f_k = k / nfft},
 * {@code k = 0..nfft/2}:
 * <ul>
 *   <li>{@link Method#PERIODOGRAM}: {@code |FFT(x - mean)|^2 / n} (core {@link Fft});</li>
 *   <li>{@link Method#WELCH} (Welch 1967): averaged Hann-windowed periodograms of half-overlapping
 *       segments;</li>
 *   <li>{@link Method#MULTITAPER} (Thomson, Proc. IEEE 1982): the average of {@code K = 2NW - 1}
 *       eigenspectra with discrete prolate spheroidal (Slepian) tapers from the tridiagonal
 *       commuting tridiagonal matrix (core {@link Matrices#lanczos}), and the harmonic F-test
 *       {@link #fTest};</li>
 *   <li>{@link Method#BURG} (Burg 1967): an autoregressive fit minimising forward and backward
 *       prediction errors, order chosen by AIC;</li>
 * </ul>
 * and {@link #lombScargle} (Lomb 1976; Scargle 1982) for irregular sampling, with the false-alarm
 * probability {@link #falseAlarm}.
 */
public final class SpectralEstimation {

    /** The regular-sampling estimators. */
    public enum Method {
        /** The raw periodogram. */
        PERIODOGRAM,
        /** Welch's averaged periodogram. */
        WELCH,
        /** Thomson's multitaper. */
        MULTITAPER,
        /** Burg's autoregressive spectrum with AIC order. */
        BURG
    }

    /** Recently used tapers, keyed by length, bandwidth and count. */
    private static final Map<String, double[][]> TAPERS = Collections.synchronizedMap(
            new LinkedHashMap<String, double[][]>(16, 0.75f, true) {
                private static final long serialVersionUID = 1L;

                @Override
                protected boolean removeEldestEntry(Map.Entry<String, double[][]> eldest) {
                    return size() > 16;
                }
            });

    private SpectralEstimation() {
    }

    private static double[][] cachedDpss(int n, double nw, int count) {
        return TAPERS.computeIfAbsent(n + ":" + nw + ":" + count, key -> dpss(n, nw, count));
    }

    /**
     * Estimates the PSD.
     *
     * @param method the estimator
     * @param x the series
     * @param nfft the grid size (a power of two, at least {@code x.length})
     * @param parameter Welch's segment length, the multitaper's time-bandwidth {@code NW}, or
     *     Burg's maximum order
     * @return the PSD at {@code k / nfft}, {@code k = 0..nfft/2}
     */
    public static double[] psd(Method method, double[] x, int nfft, double parameter) {
        double[] c = centred(x);
        return switch (method) {
            case PERIODOGRAM -> tapered(c, ones(c.length), nfft);
            case WELCH -> welch(c, (int) parameter, nfft);
            case MULTITAPER -> multitaper(c, parameter, nfft);
            case BURG -> arSpectrum(burg(c, (int) parameter), nfft);
        };
    }

    private static double[] centred(double[] x) {
        double m = 0;
        for (double v : x) {
            m += v / x.length;
        }
        double[] c = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            c[i] = x[i] - m;
        }
        return c;
    }

    private static double[] ones(int n) {
        double[] w = new double[n];
        java.util.Arrays.fill(w, 1 / Math.sqrt(n));
        return w;
    }

    /** {@code |FFT(w x)|^2} for a unit-energy window {@code w}. */
    private static double[] tapered(double[] x, double[] w, int nfft) {
        float[] d = new float[2 * nfft];
        for (int i = 0; i < x.length; i++) {
            d[2 * i] = (float) (x[i] * w[i]);
        }
        Fft.transform(d, 1, nfft, false);
        double[] p = new double[nfft / 2 + 1];
        for (int k = 0; k <= nfft / 2; k++) {
            p[k] = (double) d[2 * k] * d[2 * k] + (double) d[2 * k + 1] * d[2 * k + 1];
        }
        return p;
    }

    private static double[] hann(int n) {
        double[] w = new double[n];
        double e = 0;
        for (int i = 0; i < n; i++) {
            w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
            e += w[i] * w[i];
        }
        for (int i = 0; i < n; i++) {
            w[i] /= Math.sqrt(e);
        }
        return w;
    }

    private static double[] welch(double[] x, int segment, int nfft) {
        int len = Math.min(segment, x.length);
        double[] w = hann(len);
        double[] out = new double[nfft / 2 + 1];
        int count = 0;
        for (int s = 0; s + len <= x.length; s += len / 2) {
            double[] p = tapered(java.util.Arrays.copyOfRange(x, s, s + len), w, nfft);
            for (int k = 0; k < out.length; k++) {
                out[k] += p[k];
            }
            count++;
        }
        for (int k = 0; k < out.length; k++) {
            out[k] /= count;
        }
        return out;
    }

    /**
     * The discrete prolate spheroidal sequences: the leading eigenvectors of the tridiagonal
     * matrix commuting with the time-frequency concentration problem (Percival &amp; Walden).
     *
     * @param n the length
     * @param nw the time-bandwidth product
     * @param count how many
     * @return {@code [count][n]} unit-energy tapers
     */
    public static double[][] dpss(int n, double nw, int count) {
        double w = nw / n;
        double[] diag = new double[n];
        double[] off = new double[n];
        for (int i = 0; i < n; i++) {
            double c = (n - 1 - 2.0 * i) / 2;
            diag[i] = c * c * Math.cos(2 * Math.PI * w);
            off[i] = i + 1 < n ? (i + 1) * (n - i - 1) / 2.0 : 0;
        }
        Matrices.Operator op = (in, o) -> {
            for (int i = 0; i < n; i++) {
                double s = diag[i] * in[i];
                if (i > 0) {
                    s += off[i - 1] * in[i - 1];
                }
                if (i + 1 < n) {
                    s += off[i] * in[i + 1];
                }
                o[i] = s;
            }
        };
        Matrices.Eigenpairs eig = Matrices.lanczos(op, n, count, true, Math.min(n, 4 * count + 60), 1);
        double[][] out = new double[count][n];
        for (int k = 0; k < count; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                out[k][i] = eig.vectors()[k][i];
                sum += out[k][i];
            }
            if ((k % 2 == 0 && sum < 0) || (k % 2 == 1 && out[k][1] < 0)) {
                for (int i = 0; i < n; i++) {
                    out[k][i] = -out[k][i];
                }
            }
        }
        return out;
    }

    private static double[] multitaper(double[] x, double nw, int nfft) {
        int k = Math.max(1, (int) Math.round(2 * nw) - 1);
        double[][] tapers = cachedDpss(x.length, nw, k);
        double[] out = new double[nfft / 2 + 1];
        for (double[] taper : tapers) {
            double[] p = tapered(x, taper, nfft);
            for (int j = 0; j < out.length; j++) {
                out[j] += p[j] / k;
            }
        }
        return out;
    }

    /**
     * Thomson's harmonic F-test for a line at each grid frequency.
     *
     * @param x the series
     * @param nw the time-bandwidth product
     * @param nfft the grid size
     * @return the F statistic (2 and {@code 2K - 2} degrees of freedom) at {@code k / nfft}
     */
    public static double[] fTest(double[] x, double nw, int nfft) {
        double[] c = centred(x);
        int k = Math.max(2, (int) Math.round(2 * nw) - 1);
        double[][] tapers = cachedDpss(c.length, nw, k);
        double[] u0 = new double[k];
        double u2 = 0;
        float[][] spectra = new float[k][];
        for (int j = 0; j < k; j++) {
            for (double v : tapers[j]) {
                u0[j] += v;
            }
            u2 += u0[j] * u0[j];
            float[] d = new float[2 * nfft];
            for (int i = 0; i < c.length; i++) {
                d[2 * i] = (float) (c[i] * tapers[j][i]);
            }
            Fft.transform(d, 1, nfft, false);
            spectra[j] = d;
        }
        double[] f = new double[nfft / 2 + 1];
        for (int b = 0; b <= nfft / 2; b++) {
            double mr = 0;
            double mi = 0;
            for (int j = 0; j < k; j++) {
                mr += u0[j] * spectra[j][2 * b] / u2;
                mi += u0[j] * spectra[j][2 * b + 1] / u2;
            }
            double res = 0;
            for (int j = 0; j < k; j++) {
                double er = spectra[j][2 * b] - mr * u0[j];
                double ei = spectra[j][2 * b + 1] - mi * u0[j];
                res += er * er + ei * ei;
            }
            f[b] = (k - 1) * (mr * mr + mi * mi) * u2 / Math.max(res, 1e-30);
        }
        return f;
    }

    /**
     * Burg's method with the order chosen by AIC up to {@code maxOrder}.
     *
     * @param x the centred series
     * @param maxOrder the largest order
     * @return {@code {a_1..a_p, sigma^2}} for {@code x_t = sum a_k x_{t-k} + e_t}
     */
    public static double[] burg(double[] x, int maxOrder) {
        int n = x.length;
        double[] f = x.clone();
        double[] b = x.clone();
        double e = 0;
        for (double v : x) {
            e += v * v / n;
        }
        double[] a = new double[0];
        double[] best = {e};
        double bestAic = n * Math.log(e);
        for (int p = 1; p <= maxOrder; p++) {
            double num = 0;
            double den = 0;
            for (int t = p; t < n; t++) {
                num += f[t] * b[t - 1];
                den += f[t] * f[t] + b[t - 1] * b[t - 1];
            }
            double k = 2 * num / den;
            double[] na = new double[p];
            for (int i = 0; i < p - 1; i++) {
                na[i] = a[i] - k * a[p - 2 - i];
            }
            na[p - 1] = k;
            a = na;
            for (int t = n - 1; t >= p; t--) {
                double ft = f[t];
                f[t] = ft - k * b[t - 1];
                b[t] = b[t - 1] - k * ft;
            }
            e *= 1 - k * k;
            double aic = n * Math.log(e) + 2 * p;
            if (aic < bestAic) {
                bestAic = aic;
                best = new double[p + 1];
                System.arraycopy(a, 0, best, 0, p);
                best[p] = e;
            }
        }
        return best;
    }

    /**
     * The PSD of an AR model.
     *
     * @param model {@code {a_1..a_p, sigma^2}}
     * @param nfft the grid size
     * @return the PSD at {@code k / nfft}
     */
    public static double[] arSpectrum(double[] model, int nfft) {
        int p = model.length - 1;
        double s2 = model[p];
        double[] out = new double[nfft / 2 + 1];
        for (int k = 0; k < out.length; k++) {
            double w = 2 * Math.PI * k / nfft;
            double re = 1;
            double im = 0;
            for (int j = 1; j <= p; j++) {
                re -= model[j - 1] * Math.cos(w * j);
                im += model[j - 1] * Math.sin(w * j);
            }
            out[k] = s2 / (re * re + im * im);
        }
        return out;
    }

    /**
     * The normalised Lomb-Scargle periodogram.
     *
     * @param t the sample times
     * @param x the values
     * @param frequencies the frequencies (cycles per unit time)
     * @return the power, in units of the data variance (exponentially distributed under noise)
     */
    public static double[] lombScargle(double[] t, double[] x, double[] frequencies) {
        int n = x.length;
        double m = 0;
        for (double v : x) {
            m += v / n;
        }
        double var = 0;
        for (double v : x) {
            var += (v - m) * (v - m) / (n - 1);
        }
        double[] out = new double[frequencies.length];
        for (int k = 0; k < frequencies.length; k++) {
            double w = 2 * Math.PI * frequencies[k];
            if (w == 0) {
                continue;
            }
            double s2 = 0;
            double c2 = 0;
            for (double ti : t) {
                s2 += Math.sin(2 * w * ti);
                c2 += Math.cos(2 * w * ti);
            }
            double tau = Math.atan2(s2, c2) / (2 * w);
            double yc = 0;
            double ys = 0;
            double cc = 0;
            double ss = 0;
            for (int i = 0; i < n; i++) {
                double c = Math.cos(w * (t[i] - tau));
                double s = Math.sin(w * (t[i] - tau));
                yc += (x[i] - m) * c;
                ys += (x[i] - m) * s;
                cc += c * c;
                ss += s * s;
            }
            out[k] = (yc * yc / cc + ys * ys / ss) / (2 * var);
        }
        return out;
    }

    /**
     * The false-alarm probability of a Lomb-Scargle peak (Scargle 1982), with the number of
     * independent frequencies taken as the number of samples.
     *
     * @param power the normalised peak power
     * @param samples the number of samples
     * @return {@code 1 - (1 - e^-P)^n}
     */
    public static double falseAlarm(double power, int samples) {
        return -Math.expm1(samples * Math.log1p(-Math.exp(-power)));
    }
}
