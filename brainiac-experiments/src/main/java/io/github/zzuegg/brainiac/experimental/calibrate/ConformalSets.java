package io.github.zzuegg.brainiac.experimental.calibrate;

import io.github.zzuegg.brainiac.experimental.train.ConformalClassifier;

import java.util.Arrays;

/**
 * Conformal prediction sets beyond {@link ConformalClassifier}'s marginal LAC and APS sets:
 *
 * <ul>
 *   <li>{@link #raps}: regularised adaptive prediction sets (Angelopoulos et al., ICLR 2021,
 *       <a href="https://arxiv.org/abs/2009.14193">arXiv:2009.14193</a>), the APS score plus
 *       {@code λ (rank - k_reg)⁺}, which stops noisy tail probabilities from inflating the
 *       sets; {@link #selectRaps} picks {@code k_reg} (the {@code 1 - α} quantile of the true
 *       labels' ranks) and {@code λ} (the smallest mean set size) on a tuning split. This is the
 *       non-randomised score, so {@code λ = 0} is exactly {@link ConformalClassifier.Score#APS};</li>
 *   <li>{@link #mondrian}: class-conditional (Mondrian) conformal prediction (Vovk et al.
 *       2005): one {@link ConformalClassifier} threshold per class, calibrated on that class's
 *       samples only, so every class gets {@code 1 - α} coverage;</li>
 *   <li>{@link #riskControl}: conformal risk control (Angelopoulos et al., ICLR 2024,
 *       <a href="https://arxiv.org/abs/2208.02814">arXiv:2208.02814</a>), which bounds the
 *       expected value of any bounded loss that is non-increasing in a set parameter
 *       {@code λ}: {@code λ̂ = inf{λ : (n R̂(λ) + B) / (n + 1) <= α}}; {@link #multilabelFnr}
 *       applies it to the false-negative rate of multilabel sets {@code {k : p_k >= 1 - λ}};</li>
 *   <li>{@link #topK}: fixed top-k sets, the size-matched baseline.</li>
 * </ul>
 */
public final class ConformalSets {

    /** A set-valued classifier. */
    @FunctionalInterface
    public interface SetPredictor {

        /**
         * @param p the model's probabilities for one input
         * @return the classes in the set
         */
        int[] predictSet(double[] p);

        /**
         * @param probs probabilities
         * @param labels true classes
         * @return the share of sets holding the true class
         */
        default double coverage(double[][] probs, int[] labels) {
            int hit = 0;
            for (int i = 0; i < labels.length; i++) {
                hit += contains(predictSet(probs[i]), labels[i]) ? 1 : 0;
            }
            return hit / (double) labels.length;
        }

        /**
         * @param probs probabilities
         * @return the mean set size
         */
        default double meanSetSize(double[][] probs) {
            double s = 0;
            for (double[] p : probs) {
                s += predictSet(p).length;
            }
            return s / probs.length;
        }

        /**
         * @param probs probabilities
         * @param labels true classes
         * @param classes the number of classes
         * @return the coverage of each class (NaN for absent classes)
         */
        default double[] classCoverage(double[][] probs, int[] labels, int classes) {
            double[] hit = new double[classes];
            double[] n = new double[classes];
            for (int i = 0; i < labels.length; i++) {
                n[labels[i]]++;
                hit[labels[i]] += contains(predictSet(probs[i]), labels[i]) ? 1 : 0;
            }
            for (int c = 0; c < classes; c++) {
                hit[c] = n[c] == 0 ? Double.NaN : hit[c] / n[c];
            }
            return hit;
        }

        /**
         * @param probs probabilities
         * @param labels true classes
         * @param classes the number of classes
         * @return the lowest per-class coverage over present classes
         */
        default double worstClassCoverage(double[][] probs, int[] labels, int classes) {
            double w = 1;
            for (double v : classCoverage(probs, labels, classes)) {
                if (!Double.isNaN(v)) {
                    w = Math.min(w, v);
                }
            }
            return w;
        }
    }

    private ConformalSets() {
    }

    static boolean contains(int[] set, int c) {
        for (int v : set) {
            if (v == c) {
                return true;
            }
        }
        return false;
    }

    /** Classes by descending probability. */
    static int[] order(double[] p) {
        Integer[] o = new Integer[p.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = i;
        }
        Arrays.sort(o, (a, b) -> Double.compare(p[b], p[a]));
        int[] out = new int[o.length];
        for (int i = 0; i < o.length; i++) {
            out[i] = o[i];
        }
        return out;
    }

    /**
     * The non-randomised RAPS score of a class.
     *
     * @param p probabilities
     * @param label the class
     * @param lambda the rank penalty
     * @param kReg the penalty-free ranks
     * @return the mass of the classes ranked at or above it plus {@code λ (rank - k_reg)⁺}
     */
    public static double rapsScore(double[] p, int label, double lambda, int kReg) {
        double mass = 0;
        int rank = 0;
        for (double v : p) {
            if (v >= p[label]) {
                mass += v;
                rank++;
            }
        }
        return mass + lambda * Math.max(0, rank - kReg);
    }

    /**
     * RAPS calibrated with given {@code λ} and {@code k_reg}.
     *
     * @param probs calibration probabilities
     * @param labels calibration labels
     * @param alpha the allowed miscoverage
     * @param lambda the rank penalty
     * @param kReg the penalty-free ranks
     * @return the set predictor
     */
    public static SetPredictor raps(double[][] probs, int[] labels, double alpha, double lambda, int kReg) {
        int n = labels.length;
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = rapsScore(probs[i], labels[i], lambda, kReg);
        }
        double q = quantile(s, alpha);
        return p -> {
            int[] o = order(p);
            int[] out = new int[p.length];
            int m = 0;
            for (int c : o) {
                if (rapsScore(p, c, lambda, kReg) <= q) {
                    out[m++] = c;
                }
            }
            return Arrays.copyOf(out, m);
        };
    }

    /**
     * RAPS with {@code k_reg} and {@code λ} chosen on the first {@code tuneShare} of the
     * calibration rows and the threshold calibrated on the rest.
     *
     * @param probs calibration probabilities
     * @param labels calibration labels
     * @param alpha the allowed miscoverage
     * @param tuneShare the share of rows used for tuning, such as 0.3
     * @return {@code {chosen λ, chosen k_reg}} and the predictor
     */
    public static Tuned selectRaps(double[][] probs, int[] labels, double alpha, double tuneShare) {
        int t = (int) Math.round(labels.length * tuneShare);
        double[][] tp = Arrays.copyOfRange(probs, 0, t);
        int[] ty = Arrays.copyOfRange(labels, 0, t);
        double[][] cp = Arrays.copyOfRange(probs, t, probs.length);
        int[] cy = Arrays.copyOfRange(labels, t, labels.length);
        double[] ranks = new double[t];
        for (int i = 0; i < t; i++) {
            int r = 0;
            for (double v : tp[i]) {
                r += v >= tp[i][ty[i]] ? 1 : 0;
            }
            ranks[i] = r;
        }
        int kReg = (int) Math.max(1, quantile(ranks, alpha));
        double best = Double.POSITIVE_INFINITY;
        double lambda = 0;
        for (double l : new double[] {0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5}) {
            double size = raps(tp, ty, alpha, l, kReg).meanSetSize(tp);
            if (size < best - 1e-9) {
                best = size;
                lambda = l;
            }
        }
        return new Tuned(lambda, kReg, raps(cp, cy, alpha, lambda, kReg));
    }

    /**
     * A tuned RAPS predictor.
     *
     * @param lambda the chosen rank penalty
     * @param kReg the chosen penalty-free ranks
     * @param predictor the calibrated predictor
     */
    public record Tuned(double lambda, int kReg, SetPredictor predictor) {
    }

    /**
     * Mondrian (class-conditional) sets: one {@link ConformalClassifier} per class calibrated on
     * that class's rows, a class joining the set when its own threshold admits it.
     *
     * @param probs calibration probabilities
     * @param labels calibration labels
     * @param classes the number of classes
     * @param alpha the allowed miscoverage per class
     * @param score the nonconformity score
     * @return the set predictor
     */
    public static SetPredictor mondrian(double[][] probs, int[] labels, int classes, double alpha,
            ConformalClassifier.Score score) {
        ConformalClassifier[] per = new ConformalClassifier[classes];
        for (int c = 0; c < classes; c++) {
            int count = 0;
            for (int y : labels) {
                count += y == c ? 1 : 0;
            }
            double[][] p = new double[count][];
            int[] y = new int[count];
            int m = 0;
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] == c) {
                    p[m] = probs[i];
                    y[m++] = c;
                }
            }
            per[c] = count == 0 ? null : ConformalClassifier.calibrate(p, y, alpha, score);
        }
        return p -> {
            int[] out = new int[classes];
            int m = 0;
            for (int c : order(p)) {
                if (per[c] == null || contains(per[c].predictSet(p), c)) {
                    out[m++] = c;
                }
            }
            return Arrays.copyOf(out, m);
        };
    }

    /**
     * Fixed top-k sets.
     *
     * @param k the set size
     * @return the predictor
     */
    public static SetPredictor topK(int k) {
        return p -> Arrays.copyOf(order(p), Math.min(k, p.length));
    }

    /**
     * Conformal risk control over a grid of set parameters.
     *
     * @param losses {@code [n][grid]}: each calibration sample's loss at each {@code λ}, in
     *     {@code [0, B]} and non-increasing along the grid
     * @param lambdas the grid, increasing
     * @param alpha the target risk
     * @param bound the loss bound {@code B}
     * @return the smallest {@code λ} with {@code (n R̂(λ) + B) / (n + 1) <= α}, or the last grid
     *     value
     */
    public static double riskControl(double[][] losses, double[] lambdas, double alpha, double bound) {
        int n = losses.length;
        for (int g = 0; g < lambdas.length; g++) {
            double r = 0;
            for (double[] l : losses) {
                r += l[g];
            }
            if ((r + bound) / (n + 1) <= alpha) {
                return lambdas[g];
            }
        }
        return lambdas[lambdas.length - 1];
    }

    /**
     * The false-negative rate of a multilabel set {@code {k : p_k >= 1 - λ}}.
     *
     * @param p per-label probabilities
     * @param truth the true labels
     * @param lambda the set parameter
     * @return the share of true labels missed (0 when there are none)
     */
    public static double falseNegativeRate(double[] p, boolean[] truth, double lambda) {
        int pos = 0;
        int miss = 0;
        for (int k = 0; k < p.length; k++) {
            if (truth[k]) {
                pos++;
                miss += p[k] >= 1 - lambda ? 0 : 1;
            }
        }
        return pos == 0 ? 0 : miss / (double) pos;
    }

    /**
     * Conformal risk control of the multilabel false-negative rate.
     *
     * @param probs calibration per-label probabilities {@code [n][labels]}
     * @param truth calibration truth
     * @param alpha the target false-negative rate
     * @return the calibrated {@code λ}; predict {@code {k : p_k >= 1 - λ}}
     */
    public static double multilabelFnr(double[][] probs, boolean[][] truth, double alpha) {
        int grid = 1001;
        double[] lambdas = new double[grid];
        for (int g = 0; g < grid; g++) {
            lambdas[g] = g / (grid - 1.0);
        }
        double[][] losses = new double[probs.length][grid];
        for (int i = 0; i < probs.length; i++) {
            for (int g = 0; g < grid; g++) {
                losses[i][g] = falseNegativeRate(probs[i], truth[i], lambdas[g]);
            }
        }
        return riskControl(losses, lambdas, alpha, 1);
    }

    /** The conformal {@code ceil((n+1)(1-α))}-th smallest score (infinite if out of range). */
    static double quantile(double[] scores, double alpha) {
        double[] s = scores.clone();
        Arrays.sort(s);
        int n = s.length;
        int k = (int) Math.ceil((n + 1) * (1 - alpha));
        return k > n ? Double.POSITIVE_INFINITY : s[Math.max(0, k - 1)];
    }
}
