package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Scores for the classic estimators: the adjusted Rand index of a clustering (Hubert and
 * Arabie 1985), the trustworthiness of a low-dimensional embedding (Venna and Kaski 2001),
 * the area under the ROC curve of an anomaly score, and the negative log-likelihood, CRPS and
 * interval coverage of Gaussian predictions.
 */
public final class MlMetrics {

    private MlMetrics() {
    }

    /**
     * The adjusted Rand index: pair-counting agreement between two labelings, 1 for identical
     * partitions (up to renaming), about 0 for a random one. Negative labels (noise points) are
     * treated as one more cluster.
     *
     * @param truth the reference labels
     * @param predicted the cluster labels
     * @return the index, at most 1
     */
    public static double adjustedRandIndex(int[] truth, int[] predicted) {
        Map<Integer, Integer> ra = relabel(truth);
        Map<Integer, Integer> rb = relabel(predicted);
        long[][] table = new long[ra.size()][rb.size()];
        for (int i = 0; i < truth.length; i++) {
            table[ra.get(truth[i])][rb.get(predicted[i])]++;
        }
        double index = 0;
        long[] rows = new long[ra.size()];
        long[] cols = new long[rb.size()];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < cols.length; j++) {
                index += pairs(table[i][j]);
                rows[i] += table[i][j];
                cols[j] += table[i][j];
            }
        }
        double a = 0;
        for (long r : rows) {
            a += pairs(r);
        }
        double b = 0;
        for (long c : cols) {
            b += pairs(c);
        }
        double expected = a * b / pairs(truth.length);
        double max = (a + b) / 2;
        return max == expected ? 1 : (index - expected) / (max - expected);
    }

    private static Map<Integer, Integer> relabel(int[] labels) {
        Map<Integer, Integer> m = new HashMap<>();
        for (int l : labels) {
            m.putIfAbsent(l, m.size());
        }
        return m;
    }

    private static double pairs(long n) {
        return n * (n - 1) / 2.0;
    }

    /**
     * Trustworthiness of an embedding: how few of each point's {@code k} nearest neighbours in
     * the embedding are intruders that were far away in the original space. 1 means every
     * embedded neighbourhood is genuine.
     *
     * @param original the points in the input space {@code [n][d]}
     * @param embedded the same points in the embedding {@code [n][e]}
     * @param k the neighbourhood size, below {@code n / 2}
     * @return the trustworthiness in {@code [0, 1]}
     */
    public static double trustworthiness(double[][] original, double[][] embedded, int k) {
        int n = original.length;
        double[][] dIn = MlMatrices.squaredDistances(original);
        double[][] dOut = MlMatrices.squaredDistances(embedded);
        double penalty = 0;
        for (int i = 0; i < n; i++) {
            int[] rankIn = ranks(dIn[i], i);
            int[] nearOut = nearest(dOut[i], i, k);
            for (int j : nearOut) {
                if (rankIn[j] > k) {
                    penalty += rankIn[j] - k;
                }
            }
        }
        return 1 - 2.0 / (n * k * (2.0 * n - 3 * k - 1)) * penalty;
    }

    /**
     * The k-nearest-neighbour classification accuracy of a representation, leave-one-out.
     *
     * @param x the points {@code [n][d]}
     * @param labels their classes
     * @param k the number of voting neighbours
     * @return the accuracy in {@code [0, 1]}
     */
    public static double knnAccuracy(double[][] x, int[] labels, int k) {
        double[][] d = MlMatrices.squaredDistances(x);
        int classes = Arrays.stream(labels).max().orElse(0) + 1;
        int right = 0;
        for (int i = 0; i < x.length; i++) {
            int[] votes = new int[classes];
            for (int j : nearest(d[i], i, k)) {
                votes[labels[j]]++;
            }
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (votes[c] > votes[best]) {
                    best = c;
                }
            }
            right += best == labels[i] ? 1 : 0;
        }
        return right / (double) x.length;
    }

    /** Rank (1 = nearest) of every point by distance from point {@code self}. */
    private static int[] ranks(double[] dist, int self) {
        Integer[] order = order(dist);
        int[] rank = new int[dist.length];
        int r = 0;
        for (int idx : order) {
            if (idx != self) {
                rank[idx] = ++r;
            }
        }
        return rank;
    }

    /**
     * The indices of the {@code k} smallest distances, excluding one index.
     *
     * @param dist distances
     * @param self the index to skip, or -1
     * @param k how many
     * @return the indices, nearest first
     */
    public static int[] nearest(double[] dist, int self, int k) {
        Integer[] order = order(dist);
        int[] out = new int[k];
        int c = 0;
        for (int idx : order) {
            if (idx != self && c < k) {
                out[c++] = idx;
            }
        }
        return out;
    }

    private static Integer[] order(double[] dist) {
        Integer[] order = new Integer[dist.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(dist[a], dist[b]));
        return order;
    }

    /**
     * The area under the ROC curve: the probability that a random positive scores above a
     * random negative (ties count half).
     *
     * @param scores higher means more likely positive
     * @param positive the true classes
     * @return the AUROC in {@code [0, 1]}
     */
    public static double auroc(double[] scores, boolean[] positive) {
        Integer[] order = order(scores);
        double rankSum = 0;
        long pos = 0;
        int i = 0;
        while (i < order.length) {
            int j = i;
            while (j + 1 < order.length && scores[order[j + 1]] == scores[order[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0 + 1;
            for (int t = i; t <= j; t++) {
                if (positive[order[t]]) {
                    rankSum += rank;
                    pos++;
                }
            }
            i = j + 1;
        }
        long neg = scores.length - pos;
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    /**
     * The mean negative log-likelihood of targets under Gaussian predictions.
     *
     * @param mean the predicted means
     * @param variance the predicted variances, positive
     * @param target the targets
     * @return the mean NLL in nats
     */
    public static double gaussianNll(double[] mean, double[] variance, double[] target) {
        double s = 0;
        for (int i = 0; i < mean.length; i++) {
            double d = target[i] - mean[i];
            s += 0.5 * (Math.log(2 * Math.PI * variance[i]) + d * d / variance[i]);
        }
        return s / mean.length;
    }

    /**
     * The mean continuous ranked probability score of Gaussian predictions (closed form of
     * Gneiting and Raftery 2007); lower is better and it is in the units of the target.
     *
     * @param mean the predicted means
     * @param variance the predicted variances
     * @param target the targets
     * @return the mean CRPS
     */
    public static double gaussianCrps(double[] mean, double[] variance, double[] target) {
        double s = 0;
        for (int i = 0; i < mean.length; i++) {
            double sd = Math.sqrt(variance[i]);
            double z = (target[i] - mean[i]) / sd;
            double pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
            s += sd * (z * (2 * normalCdf(z) - 1) + 2 * pdf - 1 / Math.sqrt(Math.PI));
        }
        return s / mean.length;
    }

    /**
     * The share of targets inside central Gaussian intervals.
     *
     * @param mean the predicted means
     * @param variance the predicted variances
     * @param target the targets
     * @param z the half-width in standard deviations (1.96 for 95%)
     * @return the coverage in {@code [0, 1]}
     */
    public static double coverage(double[] mean, double[] variance, double[] target, double z) {
        int in = 0;
        for (int i = 0; i < mean.length; i++) {
            in += Math.abs(target[i] - mean[i]) <= z * Math.sqrt(variance[i]) ? 1 : 0;
        }
        return in / (double) mean.length;
    }

    /**
     * The standard normal cumulative distribution (Abramowitz-Stegun 7.1.26 through erf, error
     * below 1.5e-7).
     *
     * @param z the point
     * @return {@code Phi(z)}
     */
    public static double normalCdf(double z) {
        double x = Math.abs(z) / Math.sqrt(2);
        double t = 1 / (1 + 0.3275911 * x);
        double erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592) * t * Math.exp(-x * x);
        return z >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
    }

    /**
     * The classification accuracy.
     *
     * @param predicted predicted classes
     * @param truth true classes
     * @return the share equal
     */
    public static double accuracy(int[] predicted, int[] truth) {
        int right = 0;
        for (int i = 0; i < truth.length; i++) {
            right += predicted[i] == truth[i] ? 1 : 0;
        }
        return right / (double) truth.length;
    }
}
