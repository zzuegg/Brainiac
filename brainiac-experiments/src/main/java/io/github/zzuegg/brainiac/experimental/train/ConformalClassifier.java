package io.github.zzuegg.brainiac.experimental.train;

import java.util.Arrays;

/**
 * Split conformal prediction for classifiers: turns any model's class probabilities into
 * <em>prediction sets</em> that contain the true class with probability at least
 * {@code 1 - alpha}, guaranteed for exchangeable data whatever the model's quality. The
 * sets grow on hard or unfamiliar inputs, so their size is itself an uncertainty measure.
 * A held-out calibration set gives the nonconformity scores, and the threshold is their
 * {@code ceil((n + 1)(1 - alpha)) / n} quantile.
 *
 * <p>Two scores: {@link Score#LAC} ({@code 1 - p(true class)}, the smallest sets; Sadinle
 * et al. 2019, <a href="https://arxiv.org/abs/1609.00451">arXiv:1609.00451</a>) and
 * {@link Score#APS} (the probability mass down to the true class, sets adapting better to
 * difficulty; Romano et al. 2020, <a href="https://arxiv.org/abs/2006.02544">arXiv:2006.02544</a>).
 * Angelopoulos and Bates 2021, "A Gentle Introduction to Conformal Prediction",
 * <a href="https://arxiv.org/abs/2107.07511">arXiv:2107.07511</a>.
 */
public final class ConformalClassifier {

    /** The nonconformity score. */
    public enum Score {
        /** One minus the true class's probability. */
        LAC,
        /** The total probability of the classes at least as likely as the true one. */
        APS
    }

    private final Score score;
    private final double threshold;
    private final double alpha;

    private ConformalClassifier(Score score, double threshold, double alpha) {
        this.score = score;
        this.threshold = threshold;
        this.alpha = alpha;
    }

    /**
     * Calibrates on held-out data the model has not trained on.
     *
     * @param probs the model's probabilities {@code [sample][class]}
     * @param labels the true classes
     * @param alpha the allowed miscoverage, such as 0.1 for 90% coverage
     * @param score the nonconformity score
     * @return the calibrated predictor
     */
    public static ConformalClassifier calibrate(double[][] probs, int[] labels, double alpha, Score score) {
        if (alpha <= 0 || alpha >= 1) {
            throw new IllegalArgumentException("alpha must be in (0, 1)");
        }
        int n = labels.length;
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = score(probs[i], labels[i], score);
        }
        Arrays.sort(s);
        int k = (int) Math.ceil((n + 1) * (1 - alpha));
        double q = k > n ? Double.POSITIVE_INFINITY : s[k - 1];
        return new ConformalClassifier(score, q, alpha);
    }

    private static double score(double[] p, int label, Score kind) {
        if (kind == Score.LAC) {
            return 1 - p[label];
        }
        double mass = 0;
        for (double v : p) {
            if (v >= p[label]) {
                mass += v;
            }
        }
        return mass;
    }

    /**
     * The prediction set of one input: every class whose score is within the threshold.
     *
     * @param p the model's probabilities for the input
     * @return the classes in the set, most probable first
     */
    public int[] predictSet(double[] p) {
        Integer[] order = new Integer[p.length];
        for (int c = 0; c < p.length; c++) {
            order[c] = c;
        }
        Arrays.sort(order, (a, b) -> Double.compare(p[b], p[a]));
        int[] out = new int[p.length];
        int n = 0;
        for (int c : order) {
            if (score(p, c, score) <= threshold) {
                out[n++] = c;
            }
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * The share of inputs whose prediction set holds the true class.
     *
     * @param probs probabilities
     * @param labels true classes
     * @return the empirical coverage
     */
    public double coverage(double[][] probs, int[] labels) {
        int hit = 0;
        for (int i = 0; i < labels.length; i++) {
            for (int c : predictSet(probs[i])) {
                if (c == labels[i]) {
                    hit++;
                    break;
                }
            }
        }
        return hit / (double) labels.length;
    }

    /**
     * The mean prediction-set size.
     *
     * @param probs probabilities
     * @return the average number of classes per set
     */
    public double meanSetSize(double[][] probs) {
        double s = 0;
        for (double[] p : probs) {
            s += predictSet(p).length;
        }
        return s / probs.length;
    }

    /** @return the score threshold */
    public double threshold() {
        return threshold;
    }

    /** @return the miscoverage it was calibrated for */
    public double alpha() {
        return alpha;
    }
}
