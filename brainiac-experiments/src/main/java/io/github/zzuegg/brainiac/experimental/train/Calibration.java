package io.github.zzuegg.brainiac.experimental.train;

import java.util.Arrays;

/**
 * How trustworthy predicted probabilities are: calibration metrics, uncertainty measures and
 * temperature scaling, over class probabilities {@code [samples][classes]}.
 *
 * <ul>
 *   <li>{@link #expectedCalibrationError}: bin predictions by confidence and average
 *       {@code |accuracy - confidence|} weighted by bin size (Naeini et al. 2015;
 *       Guo et al. 2017, <a href="https://arxiv.org/abs/1706.04599">arXiv:1706.04599</a>);</li>
 *   <li>{@link #maximumCalibrationError}, {@link #negativeLogLikelihood},
 *       {@link #brierScore};</li>
 *   <li>{@link #entropy} (total uncertainty) and {@link #mutualInformation} (the part due to
 *       members disagreeing, BALD; Houlsby et al. 2011,
 *       <a href="https://arxiv.org/abs/1112.5745">arXiv:1112.5745</a>);</li>
 *   <li>{@link #fitTemperature}: one scalar that rescales logits to minimise validation
 *       NLL, the strongest simple recalibration (Guo et al. 2017).</li>
 * </ul>
 */
public final class Calibration {

    private Calibration() {
    }

    /**
     * Expected calibration error with equal-width confidence bins.
     *
     * @param probs class probabilities per sample
     * @param labels the true classes
     * @param bins the number of bins (15 is common)
     * @return the ECE, in {@code [0, 1]}
     */
    public static double expectedCalibrationError(double[][] probs, int[] labels, int bins) {
        double[][] b = reliability(probs, labels, bins);
        double ece = 0;
        for (double[] bin : b) {
            if (bin[2] > 0) {
                ece += bin[2] / probs.length * Math.abs(bin[0] - bin[1]);
            }
        }
        return ece;
    }

    /**
     * The largest gap between accuracy and confidence over non-empty bins.
     *
     * @param probs class probabilities per sample
     * @param labels the true classes
     * @param bins the number of bins
     * @return the MCE
     */
    public static double maximumCalibrationError(double[][] probs, int[] labels, int bins) {
        double mce = 0;
        for (double[] bin : reliability(probs, labels, bins)) {
            if (bin[2] > 0) {
                mce = Math.max(mce, Math.abs(bin[0] - bin[1]));
            }
        }
        return mce;
    }

    /**
     * The reliability diagram: per confidence bin its mean confidence, accuracy and count.
     *
     * @param probs class probabilities per sample
     * @param labels the true classes
     * @param bins the number of bins
     * @return {@code [bin][confidence, accuracy, count]}
     */
    public static double[][] reliability(double[][] probs, int[] labels, int bins) {
        double[][] out = new double[bins][3];
        for (int i = 0; i < probs.length; i++) {
            int pred = argmax(probs[i]);
            double conf = probs[i][pred];
            int b = Math.min(bins - 1, (int) (conf * bins));
            out[b][0] += conf;
            out[b][1] += pred == labels[i] ? 1 : 0;
            out[b][2]++;
        }
        for (double[] bin : out) {
            if (bin[2] > 0) {
                bin[0] /= bin[2];
                bin[1] /= bin[2];
            }
        }
        return out;
    }

    /**
     * The mean negative log-likelihood of the true classes.
     *
     * @param probs class probabilities per sample
     * @param labels the true classes
     * @return the NLL in nats
     */
    public static double negativeLogLikelihood(double[][] probs, int[] labels) {
        double s = 0;
        for (int i = 0; i < probs.length; i++) {
            s -= Math.log(Math.max(probs[i][labels[i]], 1e-12));
        }
        return s / probs.length;
    }

    /**
     * The mean Brier score, {@code sum_c (p_c - y_c)^2}.
     *
     * @param probs class probabilities per sample
     * @param labels the true classes
     * @return the Brier score
     */
    public static double brierScore(double[][] probs, int[] labels) {
        double s = 0;
        for (int i = 0; i < probs.length; i++) {
            for (int c = 0; c < probs[i].length; c++) {
                double d = probs[i][c] - (c == labels[i] ? 1 : 0);
                s += d * d;
            }
        }
        return s / probs.length;
    }

    /**
     * The accuracy of the most probable class.
     *
     * @param probs class probabilities per sample
     * @param labels the true classes
     * @return the accuracy
     */
    public static double accuracy(double[][] probs, int[] labels) {
        int ok = 0;
        for (int i = 0; i < probs.length; i++) {
            ok += argmax(probs[i]) == labels[i] ? 1 : 0;
        }
        return ok / (double) probs.length;
    }

    /**
     * The entropy of a distribution.
     *
     * @param p the probabilities
     * @return the entropy in nats
     */
    public static double entropy(double[] p) {
        double h = 0;
        for (double x : p) {
            if (x > 0) {
                h -= x * Math.log(x);
            }
        }
        return h;
    }

    /**
     * The mutual information between the prediction and the member, BALD: entropy of the
     * mean minus the mean entropy. Zero when all members agree.
     *
     * @param members each member's probabilities for one sample
     * @return the disagreement in nats
     */
    public static double mutualInformation(double[][] members) {
        double[] mean = new double[members[0].length];
        double meanEntropy = 0;
        for (double[] m : members) {
            for (int c = 0; c < mean.length; c++) {
                mean[c] += m[c] / members.length;
            }
            meanEntropy += entropy(m) / members.length;
        }
        return Math.max(0, entropy(mean) - meanEntropy);
    }

    /**
     * The index of the largest value.
     *
     * @param p the values
     * @return its index
     */
    public static int argmax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Softmax of logits divided by a temperature.
     *
     * @param logits the logits
     * @param temperature the temperature
     * @return the probabilities
     */
    public static double[] softmax(double[] logits, double temperature) {
        double max = Arrays.stream(logits).max().orElse(0);
        double[] out = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.exp((logits[i] - max) / temperature);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    /**
     * Finds the temperature that minimises the NLL of {@code softmax(logits / T)} on held-out
     * data (golden-section search over {@code log T}).
     *
     * @param logits validation logits
     * @param labels validation classes
     * @return the temperature
     */
    public static double fitTemperature(double[][] logits, int[] labels) {
        double lo = Math.log(0.05);
        double hi = Math.log(20);
        double g = (Math.sqrt(5) - 1) / 2;
        for (int it = 0; it < 60; it++) {
            double a = hi - g * (hi - lo);
            double b = lo + g * (hi - lo);
            if (nll(logits, labels, Math.exp(a)) < nll(logits, labels, Math.exp(b))) {
                hi = b;
            } else {
                lo = a;
            }
        }
        return Math.exp((lo + hi) / 2);
    }

    private static double nll(double[][] logits, int[] labels, double t) {
        double s = 0;
        for (int i = 0; i < logits.length; i++) {
            s -= Math.log(Math.max(softmax(logits[i], t)[labels[i]], 1e-12));
        }
        return s / logits.length;
    }

    /**
     * Applies a temperature to logits.
     *
     * @param logits the logits per sample
     * @param temperature the temperature
     * @return probabilities per sample
     */
    public static double[][] withTemperature(double[][] logits, double temperature) {
        double[][] out = new double[logits.length][];
        for (int i = 0; i < logits.length; i++) {
            out[i] = softmax(logits[i], temperature);
        }
        return out;
    }
}
