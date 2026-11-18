package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * OpenMax open-set recognition (Bendale and Boult, CVPR 2016,
 * <a href="https://arxiv.org/abs/1511.06233">arXiv:1511.06233</a>). The activation vectors
 * (logits) of correctly classified training inputs give each class a mean activation vector;
 * a Weibull model fitted by maximum likelihood to the largest {@code tail} distances from that
 * mean (extreme value theory) says how unusual a distance is. At test time the top
 * {@code alpha} classes lose a share of their activation in proportion to that Weibull CDF
 * (rank weights {@code (alpha + 1 - i) / alpha}), the lost mass becomes the activation of an
 * explicit "unknown" class, and a softmax over {@code classes + 1} recalibrates.
 *
 * <p>Vaze et al. (ICLR 2022, <a href="https://arxiv.org/abs/2110.06207">arXiv:2110.06207</a>)
 * found that the max logit of a well-trained closed-set classifier matches dedicated open-set
 * methods; the experiment compares both.
 */
public final class OpenMax {

    /**
     * A two-parameter Weibull distribution.
     *
     * @param shape {@code k}
     * @param scale {@code lambda}
     */
    public record Weibull(double shape, double scale) {

        /**
         * @param x a value
         * @return {@code 1 - exp(-(x / lambda)^k)} for {@code x > 0}, else 0
         */
        public double cdf(double x) {
            return x <= 0 ? 0 : 1 - Math.exp(-Math.pow(x / scale, shape));
        }

        /**
         * The maximum-likelihood fit (bisection on the shape's score equation).
         *
         * @param samples positive samples
         * @return the fitted distribution
         */
        public static Weibull fit(double[] samples) {
            double meanLog = 0;
            for (double s : samples) {
                meanLog += Math.log(Math.max(s, 1e-12)) / samples.length;
            }
            double lo = 0.05;
            double hi = 200;
            for (int it = 0; it < 100; it++) {
                double k = Math.sqrt(lo * hi);
                if (score(samples, k, meanLog) > 0) {
                    hi = k;
                } else {
                    lo = k;
                }
            }
            double k = Math.sqrt(lo * hi);
            double m = 0;
            for (double s : samples) {
                m += Math.pow(Math.max(s, 1e-12), k) / samples.length;
            }
            return new Weibull(k, Math.pow(m, 1 / k));
        }

        /** {@code sum x^k ln x / sum x^k - 1/k - mean ln x}, increasing in k. */
        private static double score(double[] x, double k, double meanLog) {
            double a = 0;
            double b = 0;
            for (double s : x) {
                double v = Math.max(s, 1e-12);
                double p = Math.pow(v, k);
                a += p * Math.log(v);
                b += p;
            }
            return a / b - 1 / k - meanLog;
        }
    }

    private final double[][] means;
    private final Weibull[] models;
    private final int alpha;

    private OpenMax(double[][] means, Weibull[] models, int alpha) {
        this.means = means;
        this.models = models;
        this.alpha = alpha;
    }

    /**
     * Fits the class means and Weibull tails.
     *
     * @param logits training activation vectors {@code [n, classes]}
     * @param labels their classes
     * @param tail the number of largest distances per class to fit, such as 20
     * @param alpha the number of top classes to revise, such as 2
     * @return the model
     */
    public static OpenMax fit(Tensor logits, int[] labels, int tail, int alpha) {
        int n = labels.length;
        int c = logits.shape().dim(1);
        double[] v = logits.toDoubleArray();
        double[][] means = new double[c][c];
        int[] count = new int[c];
        for (int i = 0; i < n; i++) {
            if (argmax(v, i * c, c) == labels[i]) {
                count[labels[i]]++;
                for (int j = 0; j < c; j++) {
                    means[labels[i]][j] += v[i * c + j];
                }
            }
        }
        for (int k = 0; k < c; k++) {
            for (int j = 0; j < c; j++) {
                means[k][j] /= Math.max(1, count[k]);
            }
        }
        Weibull[] models = new Weibull[c];
        for (int k = 0; k < c; k++) {
            double[] d = new double[count[k]];
            int m = 0;
            for (int i = 0; i < n; i++) {
                if (labels[i] == k && argmax(v, i * c, c) == k) {
                    d[m++] = distance(v, i * c, means[k]);
                }
            }
            Arrays.sort(d);
            double[] top = Arrays.copyOfRange(d, Math.max(0, d.length - tail), d.length);
            models[k] = top.length < 2 ? new Weibull(1, 1) : Weibull.fit(top);
        }
        return new OpenMax(means, models, Math.min(alpha, c));
    }

    /**
     * The recalibrated probabilities.
     *
     * @param logits activation vectors {@code [n, classes]}
     * @return {@code [n][classes + 1]}, the last column the unknown class
     */
    public double[][] probabilities(Tensor logits) {
        int c = means.length;
        double[] v = logits.toDoubleArray();
        int n = v.length / c;
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] row = Arrays.copyOfRange(v, i * c, (i + 1) * c);
            Integer[] order = new Integer[c];
            for (int j = 0; j < c; j++) {
                order[j] = j;
            }
            Arrays.sort(order, (a, b) -> Double.compare(row[b], row[a]));
            double[] revised = row.clone();
            double unknown = 0;
            for (int r = 0; r < alpha; r++) {
                int k = order[r];
                double w = (alpha - r) / (double) alpha * models[k].cdf(distance(v, i * c, means[k]));
                revised[k] = row[k] * (1 - w);
                unknown += row[k] * w;
            }
            double[] z = new double[c + 1];
            System.arraycopy(revised, 0, z, 0, c);
            z[c] = unknown;
            double max = Double.NEGATIVE_INFINITY;
            for (double x : z) {
                max = Math.max(max, x);
            }
            double s = 0;
            for (int j = 0; j <= c; j++) {
                z[j] = Math.exp(z[j] - max);
                s += z[j];
            }
            for (int j = 0; j <= c; j++) {
                z[j] /= s;
            }
            out[i] = z;
        }
        return out;
    }

    /**
     * The known-class score for open-set AUROC.
     *
     * @param logits activation vectors
     * @return {@code 1 - p(unknown)} per row, higher meaning known
     */
    public double[] knownScore(Tensor logits) {
        double[][] p = probabilities(logits);
        double[] s = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            s[i] = 1 - p[i][p[i].length - 1];
        }
        return s;
    }

    /** @return the fitted Weibull tail per class */
    public Weibull[] tails() {
        return models.clone();
    }

    private static int argmax(double[] v, int off, int c) {
        int best = 0;
        for (int j = 1; j < c; j++) {
            if (v[off + j] > v[off + best]) {
                best = j;
            }
        }
        return best;
    }

    private static double distance(double[] v, int off, double[] mean) {
        double s = 0;
        for (int j = 0; j < mean.length; j++) {
            double d = v[off + j] - mean[j];
            s += d * d;
        }
        return Math.sqrt(s);
    }
}
