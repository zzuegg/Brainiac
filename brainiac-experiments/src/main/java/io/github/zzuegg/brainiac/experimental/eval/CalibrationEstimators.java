package io.github.zzuegg.brainiac.experimental.eval;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Estimators of calibration error that fix the binned ECE's bias and bin dependence. Inputs
 * are top-label confidences {@code c} and 0/1 correctness {@code y} (see
 * {@link #topLabel(double[][], int[])}).
 * <ul>
 *   <li>{@link #equalWidth}: the usual plug-in ECE, {@code Σ_b (n_b/n) |acc_b - conf_b|} over
 *       equal-width bins: biased upwards with few samples per bin;</li>
 *   <li>{@link #equalMass}: the same over equal-mass (quantile) bins;</li>
 *   <li>{@link #monotoneSweep}: Roelofs et al. (AISTATS 2022,
 *       <a href="https://arxiv.org/abs/2012.08668">arXiv:2012.08668</a>): equal-mass bins, with
 *       the largest bin count whose bin accuracies are still monotone in confidence;</li>
 *   <li>{@link #debiasedL2}: Kumar et al. (NeurIPS 2019,
 *       <a href="https://arxiv.org/abs/1909.10155">arXiv:1909.10155</a>): the plug-in squared
 *       ℓ2-ECE minus each bin's sampling variance {@code acc_b (1 - acc_b) / (n_b - 1)}, an
 *       unbiased estimate of the binned squared error; reported as its root;</li>
 *   <li>{@link #smooth}: smooth ECE (Błasiok and Nakkiran, ICLR 2024,
 *       <a href="https://arxiv.org/abs/2309.12236">arXiv:2309.12236</a>): residuals
 *       {@code y - c} smoothed by a Gaussian kernel reflected at 0 and 1, integrated in absolute
 *       value, with the bandwidth σ at the fixed point {@code smECE_σ = σ};</li>
 *   <li>{@link #classwise}: the classwise ECE of multiclass probabilities;</li>
 *   <li>{@link #reliabilityBootstrap}: per-bin accuracy intervals of the reliability diagram by
 *       bootstrap.</li>
 * </ul>
 */
public final class CalibrationEstimators {

    private CalibrationEstimators() {
    }

    /**
     * Top-label confidences and correctness of multiclass predictions.
     *
     * @param probs {@code [n][classes]}
     * @param labels true labels
     * @return {@code {confidence, correct}} as two arrays of length n
     */
    public static double[][] topLabel(double[][] probs, int[] labels) {
        double[] c = new double[probs.length];
        double[] y = new double[probs.length];
        for (int i = 0; i < probs.length; i++) {
            int best = 0;
            for (int k = 1; k < probs[i].length; k++) {
                best = probs[i][k] > probs[i][best] ? k : best;
            }
            c[i] = probs[i][best];
            y[i] = best == labels[i] ? 1 : 0;
        }
        return new double[][] {c, y};
    }

    private static double binned(double[] c, double[] y, int[] bin, int bins, boolean squared) {
        double[] acc = new double[bins];
        double[] conf = new double[bins];
        int[] n = new int[bins];
        for (int i = 0; i < c.length; i++) {
            acc[bin[i]] += y[i];
            conf[bin[i]] += c[i];
            n[bin[i]]++;
        }
        double e = 0;
        for (int b = 0; b < bins; b++) {
            if (n[b] > 0) {
                double d = (acc[b] - conf[b]) / n[b];
                e += n[b] / (double) c.length * (squared ? d * d : Math.abs(d));
            }
        }
        return e;
    }

    private static int[] widthBins(double[] c, int bins) {
        int[] b = new int[c.length];
        for (int i = 0; i < c.length; i++) {
            b[i] = Math.min(bins - 1, Math.max(0, (int) (c[i] * bins)));
        }
        return b;
    }

    private static int[] massBins(double[] c, int bins) {
        Integer[] order = new Integer[c.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(c[a], c[b]));
        int[] b = new int[c.length];
        for (int r = 0; r < order.length; r++) {
            b[order[r]] = (int) ((long) r * bins / c.length);
        }
        return b;
    }

    /**
     * The plug-in ℓ1 ECE with equal-width bins on [0, 1].
     *
     * @param c confidences
     * @param y correctness, 0 or 1
     * @param bins the bin count (15 is the usual default)
     * @return the estimate
     */
    public static double equalWidth(double[] c, double[] y, int bins) {
        return binned(c, y, widthBins(c, bins), bins, false);
    }

    /**
     * The plug-in ℓ1 ECE with equal-mass bins.
     *
     * @param c confidences
     * @param y correctness
     * @param bins the bin count
     * @return the estimate
     */
    public static double equalMass(double[] c, double[] y, int bins) {
        return binned(c, y, massBins(c, bins), bins, false);
    }

    /**
     * The monotone sweep: the ℓ1 equal-mass ECE at the largest bin count for which the bin
     * accuracies (sorted by confidence) do not decrease.
     *
     * @param c confidences
     * @param y correctness
     * @return the estimate
     */
    public static double monotoneSweep(double[] c, double[] y) {
        return equalMass(c, y, sweepBins(c, y));
    }

    /**
     * The bin count the monotone sweep picks.
     *
     * @param c confidences
     * @param y correctness
     * @return the largest monotone equal-mass bin count (at least 1)
     */
    public static int sweepBins(double[] c, double[] y) {
        Integer[] order = new Integer[c.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(c[a], c[b]));
        double[] sortedY = new double[c.length];
        for (int r = 0; r < order.length; r++) {
            sortedY[r] = y[order[r]];
        }
        int best = 1;
        for (int bins = 2; bins <= c.length; bins++) {
            boolean monotone = true;
            double prev = -1;
            for (int k = 0; k < bins && monotone; k++) {
                int from = (int) (((long) k * c.length + bins - 1) / bins);
                int to = (int) (((long) (k + 1) * c.length + bins - 1) / bins);
                if (to <= from) {
                    continue;
                }
                double a = 0;
                for (int r = from; r < to; r++) {
                    a += sortedY[r];
                }
                a /= to - from;
                monotone = a >= prev;
                prev = a;
            }
            if (!monotone) {
                break;
            }
            best = bins;
        }
        return best;
    }

    /**
     * Kumar et al.'s debiased ℓ2 ECE over equal-mass bins: the root of
     * {@code max(0, Σ_b (n_b/n) [(acc_b - conf_b)² - acc_b (1 - acc_b) / (n_b - 1)])}.
     *
     * @param c confidences
     * @param y correctness
     * @param bins the bin count
     * @return the estimate of the ℓ2 calibration error
     */
    public static double debiasedL2(double[] c, double[] y, int bins) {
        int[] bin = massBins(c, bins);
        double[] acc = new double[bins];
        double[] conf = new double[bins];
        int[] n = new int[bins];
        for (int i = 0; i < c.length; i++) {
            acc[bin[i]] += y[i];
            conf[bin[i]] += c[i];
            n[bin[i]]++;
        }
        double e = 0;
        for (int b = 0; b < bins; b++) {
            if (n[b] > 1) {
                double a = acc[b] / n[b];
                double d = a - conf[b] / n[b];
                e += n[b] / (double) c.length * (d * d - a * (1 - a) / (n[b] - 1));
            }
        }
        return Math.sqrt(Math.max(0, e));
    }

    /**
     * The plug-in ℓ2 ECE (root of the binned squared error) over equal-mass bins.
     *
     * @param c confidences
     * @param y correctness
     * @param bins the bin count
     * @return the estimate
     */
    public static double pluginL2(double[] c, double[] y, int bins) {
        return Math.sqrt(binned(c, y, massBins(c, bins), bins, true));
    }

    /**
     * Smooth ECE at a fixed bandwidth: {@code ∫ |(1/n) Σ_i K_σ(t, c_i) (y_i - c_i)| dt} with a
     * Gaussian kernel reflected at 0 and 1; confidences are binned to a grid of 200 cells.
     *
     * @param c confidences
     * @param y correctness
     * @param sigma the bandwidth
     * @return the estimate
     */
    public static double smooth(double[] c, double[] y, double sigma) {
        return smoothResiduals(residualHistogram(c, y), c.length, sigma);
    }

    private static final int GRID = 200;

    private static double[] residualHistogram(double[] c, double[] y) {
        double[] h = new double[GRID];
        for (int i = 0; i < c.length; i++) {
            h[Math.min(GRID - 1, Math.max(0, (int) (c[i] * GRID)))] += y[i] - c[i];
        }
        return h;
    }

    private static double smoothResiduals(double[] h, int n, double sigma) {
        double norm = 1 / (Math.sqrt(2 * Math.PI) * sigma);
        double total = 0;
        for (int g = 0; g < GRID; g++) {
            double t = (g + 0.5) / GRID;
            double r = 0;
            for (int j = 0; j < GRID; j++) {
                if (h[j] == 0) {
                    continue;
                }
                double cj = (j + 0.5) / GRID;
                r += (gauss(t - cj, sigma) + gauss(t + cj, sigma) + gauss(t - (2 - cj), sigma)) * h[j];
            }
            total += Math.abs(r * norm / n) / GRID;
        }
        return total;
    }

    private static double gauss(double d, double sigma) {
        return Math.exp(-d * d / (2 * sigma * sigma));
    }

    /**
     * Smooth ECE with the bandwidth at its fixed point {@code σ = smECE_σ}, found by bisection.
     *
     * @param c confidences
     * @param y correctness
     * @return the estimate
     */
    public static double smooth(double[] c, double[] y) {
        double[] h = residualHistogram(c, y);
        double lo = 1e-3;
        double hi = 1;
        if (smoothResiduals(h, c.length, lo) <= lo) {
            return smoothResiduals(h, c.length, lo);
        }
        for (int it = 0; it < 25; it++) {
            double mid = 0.5 * (lo + hi);
            if (smoothResiduals(h, c.length, mid) > mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return smoothResiduals(h, c.length, 0.5 * (lo + hi));
    }

    /**
     * Classwise ECE: the mean over classes of the equal-width ECE of "is class k" against
     * {@code p_k}.
     *
     * @param probs {@code [n][classes]}
     * @param labels true labels
     * @param bins the bin count
     * @return the estimate
     */
    public static double classwise(double[][] probs, int[] labels, int bins) {
        int k = probs[0].length;
        double sum = 0;
        for (int cl = 0; cl < k; cl++) {
            double[] c = new double[probs.length];
            double[] y = new double[probs.length];
            for (int i = 0; i < probs.length; i++) {
                c[i] = probs[i][cl];
                y[i] = labels[i] == cl ? 1 : 0;
            }
            sum += equalWidth(c, y, bins);
        }
        return sum / k;
    }

    /**
     * Reliability diagram with bootstrap intervals over equal-width bins.
     *
     * @param c confidences
     * @param y correctness
     * @param bins the bin count
     * @param resamples bootstrap resamples
     * @param seed the seed
     * @return {@code [bins][4]}: mean confidence, accuracy, 2.5% and 97.5% accuracy quantiles
     *         (NaN for empty bins)
     */
    public static double[][] reliabilityBootstrap(double[] c, double[] y, int bins, int resamples, long seed) {
        double[][] out = new double[bins][4];
        double[][] boot = new double[bins][resamples];
        SplittableRandom r = new SplittableRandom(seed);
        int[] bin = widthBins(c, bins);
        for (int k = -1; k < resamples; k++) {
            double[] acc = new double[bins];
            double[] conf = new double[bins];
            int[] n = new int[bins];
            for (int j = 0; j < c.length; j++) {
                int i = k < 0 ? j : r.nextInt(c.length);
                acc[bin[i]] += y[i];
                conf[bin[i]] += c[i];
                n[bin[i]]++;
            }
            for (int b = 0; b < bins; b++) {
                double a = n[b] == 0 ? Double.NaN : acc[b] / n[b];
                if (k < 0) {
                    out[b][0] = n[b] == 0 ? Double.NaN : conf[b] / n[b];
                    out[b][1] = a;
                } else {
                    boot[b][k] = a;
                }
            }
        }
        for (int b = 0; b < bins; b++) {
            double[] v = Arrays.stream(boot[b]).filter(Double::isFinite).sorted().toArray();
            out[b][2] = v.length == 0 ? Double.NaN : v[(int) (0.025 * (v.length - 1))];
            out[b][3] = v.length == 0 ? Double.NaN : v[(int) Math.ceil(0.975 * (v.length - 1))];
        }
        return out;
    }

    /**
     * Brier score of top-label confidences: {@code mean (c - y)²}.
     *
     * @param c confidences
     * @param y correctness
     * @return the score
     */
    public static double brier(double[] c, double[] y) {
        double s = 0;
        for (int i = 0; i < c.length; i++) {
            s += (c[i] - y[i]) * (c[i] - y[i]);
        }
        return s / c.length;
    }

    /**
     * Negative log-likelihood of top-label correctness.
     *
     * @param c confidences
     * @param y correctness
     * @return the mean NLL
     */
    public static double nll(double[] c, double[] y) {
        double s = 0;
        for (int i = 0; i < c.length; i++) {
            double p = Math.min(1 - 1e-9, Math.max(1e-9, c[i]));
            s -= y[i] > 0.5 ? Math.log(p) : Math.log(1 - p);
        }
        return s / c.length;
    }
}
