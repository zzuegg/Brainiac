package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Spike sorting of multi-channel extracellular traces.
 *
 * <ul>
 *   <li>{@link #detect}: negative threshold crossings at {@code k} robust noise standard
 *       deviations ({@code median|x| / 0.6745}), each aligned to its trough within a short
 *       window; {@link #snippets} cuts aligned waveforms on all channels.</li>
 *   <li>{@link Method#PCA_GMM}: PCA of the snippets and a Gaussian mixture on the scores
 *       (Lewicki, Network 1998).</li>
 *   <li>{@link Method#WAVELET_GMM}: Haar wavelet coefficients of each channel, the coefficients
 *       whose distribution departs most from normal (largest Kolmogorov–Smirnov distance, as
 *       Quiroga et al., Neural Comput. 2004 select them), and a Gaussian mixture.</li>
 *   <li>{@link Method#TEMPLATE_MATCHING}: templates from the PCA-GMM clusters, then greedy
 *       matching pursuit on the trace (Kilosort-style, Pachitariu et al. 2016): at each event
 *       of the residual the unit and shift with the largest fit improvement
 *       {@code 2 <r, T> - |T|^2} is subtracted, events are re-detected on the residual to
 *       resolve overlapping spikes, and the templates are re-estimated once from their
 *       assigned spikes.</li>
 * </ul>
 */
public final class SpikeSorting {

    /** The sorter. */
    public enum Method {
        /** PCA and a Gaussian mixture. */
        PCA_GMM,
        /** Haar wavelets and a Gaussian mixture. */
        WAVELET_GMM,
        /** Template matching by greedy subtraction. */
        TEMPLATE_MATCHING
    }

    /**
     * Sorted spikes.
     *
     * @param times the trough sample of each spike
     * @param labels the cluster of each spike
     * @param clusters the number of clusters
     */
    public record Sorted(int[] times, int[] labels, int clusters) {
    }

    private SpikeSorting() {
    }

    /**
     * Robust per-channel noise standard deviations.
     *
     * @param trace {@code [channel][sample]}
     * @return {@code median|x| / 0.6745} per channel
     */
    public static double[] noiseSd(double[][] trace) {
        double[] sd = new double[trace.length];
        for (int c = 0; c < trace.length; c++) {
            double[] a = new double[trace[c].length];
            for (int i = 0; i < a.length; i++) {
                a[i] = Math.abs(trace[c][i]);
            }
            Arrays.sort(a);
            sd[c] = a[a.length / 2] / 0.6745;
        }
        return sd;
    }

    /**
     * Detects negative threshold crossings and aligns each to its trough.
     *
     * @param trace {@code [channel][sample]}
     * @param sd per-channel noise standard deviations
     * @param k the threshold in standard deviations
     * @param window the trough search window and dead time in samples
     * @param margin samples kept free at both ends
     * @return trough samples, ascending
     */
    public static int[] detect(double[][] trace, double[] sd, double k, int window, int margin) {
        int n = trace[0].length;
        List<Integer> out = new ArrayList<>();
        int i = margin;
        while (i < n - margin) {
            boolean cross = false;
            for (int c = 0; c < trace.length; c++) {
                if (trace[c][i] < -k * sd[c]) {
                    cross = true;
                    break;
                }
            }
            if (!cross) {
                i++;
                continue;
            }
            int best = i;
            double low = Double.POSITIVE_INFINITY;
            for (int j = i; j < Math.min(n - margin, i + window); j++) {
                for (int c = 0; c < trace.length; c++) {
                    double v = trace[c][j] / sd[c];
                    if (v < low) {
                        low = v;
                        best = j;
                    }
                }
            }
            out.add(best);
            i = best + window;
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Cuts aligned snippets.
     *
     * @param trace {@code [channel][sample]}
     * @param times trough samples
     * @param width the snippet width
     * @param before samples before the trough
     * @return {@code [spike][channel * width]}
     */
    public static double[][] snippets(double[][] trace, int[] times, int width, int before) {
        int ch = trace.length;
        double[][] s = new double[times.length][ch * width];
        for (int k = 0; k < times.length; k++) {
            for (int c = 0; c < ch; c++) {
                for (int i = 0; i < width; i++) {
                    int at = times[k] - before + i;
                    s[k][c * width + i] = at >= 0 && at < trace[c].length ? trace[c][at] : 0;
                }
            }
        }
        return s;
    }

    /**
     * Haar wavelet coefficients of each channel's snippet (zero-padded to a power of two).
     *
     * @param snippet {@code channels * width} values
     * @param channels channels
     * @return the coefficients, channel by channel
     */
    public static double[] haar(double[] snippet, int channels) {
        int width = snippet.length / channels;
        int m = Integer.highestOneBit(Math.max(1, width - 1)) << 1;
        double[] out = new double[channels * m];
        for (int c = 0; c < channels; c++) {
            double[] v = new double[m];
            System.arraycopy(snippet, c * width, v, 0, width);
            double[] tmp = new double[m];
            for (int len = m; len > 1; len /= 2) {
                for (int i = 0; i < len / 2; i++) {
                    tmp[i] = (v[2 * i] + v[2 * i + 1]) / Math.sqrt(2);
                    tmp[len / 2 + i] = (v[2 * i] - v[2 * i + 1]) / Math.sqrt(2);
                }
                System.arraycopy(tmp, 0, v, 0, len);
            }
            System.arraycopy(v, 0, out, c * m, m);
        }
        return out;
    }

    private static double ksNormalDistance(double[] v) {
        int n = v.length;
        double m = 0;
        for (double x : v) {
            m += x / n;
        }
        double s = 0;
        for (double x : v) {
            s += (x - m) * (x - m) / n;
        }
        s = Math.sqrt(s) + 1e-12;
        double[] z = v.clone();
        Arrays.sort(z);
        double d = 0;
        for (int i = 0; i < n; i++) {
            double f = 0.5 * (1 + erf((z[i] - m) / (s * Math.sqrt(2))));
            d = Math.max(d, Math.max(Math.abs(f - i / (double) n), Math.abs(f - (i + 1.0) / n)));
        }
        return d;
    }

    private static double erf(double x) {
        double t = 1 / (1 + 0.3275911 * Math.abs(x));
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
                * Math.exp(-x * x);
        return x >= 0 ? y : -y;
    }

    /**
     * Features for clustering.
     *
     * @param snippets aligned snippets
     * @param channels channels
     * @param method {@link Method#PCA_GMM} (PCA scores) or {@link Method#WAVELET_GMM}
     * @param dims the number of features kept
     * @return {@code [spike][dims]}
     */
    public static double[][] features(double[][] snippets, int channels, Method method, int dims) {
        if (method == Method.WAVELET_GMM) {
            double[][] w = new double[snippets.length][];
            for (int i = 0; i < w.length; i++) {
                w[i] = haar(snippets[i], channels);
            }
            int d = w[0].length;
            double[] ks = new double[d];
            Integer[] order = new Integer[d];
            for (int j = 0; j < d; j++) {
                double[] col = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    col[i] = w[i][j];
                }
                ks[j] = ksNormalDistance(col);
                order[j] = j;
            }
            Arrays.sort(order, (a, b) -> Double.compare(ks[b], ks[a]));
            double[][] f = new double[w.length][dims];
            for (int i = 0; i < w.length; i++) {
                for (int k = 0; k < dims; k++) {
                    f[i][k] = w[i][order[k]];
                }
            }
            return f;
        }
        return ComponentAnalysis.pca(snippets, dims).transform(snippets);
    }

    /**
     * Sorts a recording.
     *
     * @param trace {@code [channel][sample]}
     * @param units the number of clusters to fit
     * @param method the method
     * @param rate the sampling rate
     * @param seed the seed
     * @return the sorted spikes
     */
    public static Sorted sort(double[][] trace, int units, Method method, double rate, long seed) {
        int width = (int) (rate * 0.0015);
        int before = width / 3;
        int window = Math.max(2, (int) (rate * 0.0004));
        double[] sd = noiseSd(trace);
        int[] times = detect(trace, sd, 4.5, window, width);
        double[][] snips = snippets(trace, times, width, before);
        Method clusterBy = method == Method.TEMPLATE_MATCHING ? Method.PCA_GMM : method;
        double[][] f = features(snips, trace.length, clusterBy, clusterBy == Method.WAVELET_GMM ? 10 : 5);
        int[] labels = GaussianMixture.fit(f, units, 200, 1e-4, seed).predict(f);
        if (method != Method.TEMPLATE_MATCHING) {
            return new Sorted(times, labels, units);
        }
        double[][] tpl = new double[units][snips[0].length];
        int[] count = new int[units];
        for (int k = 0; k < times.length; k++) {
            count[labels[k]]++;
            for (int j = 0; j < tpl[0].length; j++) {
                tpl[labels[k]][j] += snips[k][j];
            }
        }
        for (int u = 0; u < units; u++) {
            for (int j = 0; j < tpl[0].length; j++) {
                tpl[u][j] /= Math.max(1, count[u]);
            }
        }
        Sorted s = null;
        for (int round = 0; round < 2; round++) {
            s = matchingPursuit(trace, tpl, sd, width, before, window);
            if (round == 0) {
                tpl = reestimate(trace, s, tpl, width, before);
            }
        }
        return s;
    }

    private static Sorted matchingPursuit(double[][] trace, double[][] tpl, double[] sd, int width, int before,
            int window) {
        int ch = trace.length;
        int n = trace[0].length;
        double[][] res = new double[ch][];
        for (int c = 0; c < ch; c++) {
            res[c] = trace[c].clone();
        }
        int units = tpl.length;
        double[] energy = new double[units];
        for (int u = 0; u < units; u++) {
            for (double v : tpl[u]) {
                energy[u] += v * v;
            }
        }
        List<int[]> found = new ArrayList<>();
        int shift = Math.max(2, window);
        for (int pass = 0; pass < 3; pass++) {
            int[] events = detect(res, sd, 4.5, window, width);
            if (events.length == 0) {
                break;
            }
            for (int e : events) {
                double bestGain = 0;
                int bestU = -1;
                int bestT = -1;
                for (int u = 0; u < units; u++) {
                    for (int t = e - shift; t <= e + shift; t++) {
                        if (t - before < 0 || t - before + width > n) {
                            continue;
                        }
                        double dot = 0;
                        for (int c = 0; c < ch; c++) {
                            for (int i = 0; i < width; i++) {
                                dot += res[c][t - before + i] * tpl[u][c * width + i];
                            }
                        }
                        double gain = 2 * dot - energy[u];
                        if (gain > bestGain) {
                            bestGain = gain;
                            bestU = u;
                            bestT = t;
                        }
                    }
                }
                if (bestU >= 0) {
                    for (int c = 0; c < ch; c++) {
                        for (int i = 0; i < width; i++) {
                            res[c][bestT - before + i] -= tpl[bestU][c * width + i];
                        }
                    }
                    found.add(new int[] {bestT, bestU});
                }
            }
        }
        found.sort((x, y) -> Integer.compare(x[0], y[0]));
        int[] times = new int[found.size()];
        int[] labels = new int[found.size()];
        for (int k = 0; k < times.length; k++) {
            times[k] = found.get(k)[0];
            labels[k] = found.get(k)[1];
        }
        return new Sorted(times, labels, units);
    }

    private static double[][] reestimate(double[][] trace, Sorted s, double[][] old, int width, int before) {
        // average the trace around each assigned spike with the other assigned spikes subtracted
        int units = old.length;
        int ch = trace.length;
        int n = trace[0].length;
        double[][] res = new double[ch][];
        for (int c = 0; c < ch; c++) {
            res[c] = trace[c].clone();
        }
        for (int k = 0; k < s.times().length; k++) {
            for (int c = 0; c < ch; c++) {
                for (int i = 0; i < width; i++) {
                    int at = s.times()[k] - before + i;
                    if (at >= 0 && at < n) {
                        res[c][at] -= old[s.labels()[k]][c * width + i];
                    }
                }
            }
        }
        double[][] tpl = new double[units][old[0].length];
        int[] count = new int[units];
        for (int k = 0; k < s.times().length; k++) {
            int u = s.labels()[k];
            count[u]++;
            for (int c = 0; c < ch; c++) {
                for (int i = 0; i < width; i++) {
                    int at = s.times()[k] - before + i;
                    if (at >= 0 && at < n) {
                        tpl[u][c * width + i] += res[c][at] + old[u][c * width + i];
                    }
                }
            }
        }
        for (int u = 0; u < units; u++) {
            for (int j = 0; j < tpl[u].length; j++) {
                tpl[u][j] = count[u] > 0 ? tpl[u][j] / count[u] : old[u][j];
            }
        }
        return tpl;
    }
}
