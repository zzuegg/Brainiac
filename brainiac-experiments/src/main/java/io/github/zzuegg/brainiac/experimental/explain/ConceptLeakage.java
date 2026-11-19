package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.experimental.estimators.MutualInformation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Information-theoretic leakage scores of concept-based models (Parisini et al., "Leakage and
 * interpretability in concept-based models", 2025, <a href="https://arxiv.org/abs/2504.14094">arXiv:2504.14094</a>):
 * how much more the predicted concepts {@code c_hat} tell about the label, or about each other,
 * than the ground-truth concepts do.
 *
 * <ul>
 *   <li>{@link #ctl}, concepts-task leakage:
 *       {@code CTL_i = max(0, I(c_hat_i; y) / H(y) - I(c_i; y) / H(y))}, averaged over concepts.</li>
 *   <li>{@link #icl}, inter-concept leakage:
 *       {@code ICL_ij = max(0, I(c_hat_i; c_hat_j) / sqrt(H(c_hat_i) H(c_hat_j)) - I(c_i; c_j) / sqrt(H(c_i) H(c_j)))},
 *       averaged over pairs.</li>
 * </ul>
 * Continuous concept probabilities are discretised into equal-frequency bins (a plug-in
 * estimate); {@link Estimator#KSG} uses the Kraskov estimator of
 * {@link MutualInformation#ksg} for the mutual information between two continuous scores, the
 * estimator of the paper, keeping binned entropies for the normalisation.
 */
public final class ConceptLeakage {

    private ConceptLeakage() {
    }

    /** How mutual information between continuous scores is estimated. */
    public enum Estimator {
        /** Plug-in on equal-frequency bins. */
        BINNED,
        /** Kraskov-Stoegbauer-Grassberger. */
        KSG
    }

    /**
     * Equal-frequency bin indices; values with few distinct levels keep one bin per level.
     *
     * @param v the values
     * @param bins the bin count
     * @return a bin per value
     */
    static int[] discretise(double[] v, int bins) {
        double[] distinct = Arrays.stream(v).distinct().sorted().toArray();
        int[] out = new int[v.length];
        if (distinct.length <= bins) {
            for (int i = 0; i < v.length; i++) {
                out[i] = Arrays.binarySearch(distinct, v[i]);
            }
            return out;
        }
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(v[a], v[b]));
        for (int r = 0; r < idx.length; r++) {
            out[idx[r]] = (int) ((long) r * bins / v.length);
        }
        return out;
    }

    static double entropy(int[] a) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int v : a) {
            counts.merge(v, 1, Integer::sum);
        }
        double h = 0;
        for (int c : counts.values()) {
            double p = c / (double) a.length;
            h -= p * Math.log(p);
        }
        return h;
    }

    static double mutualInformation(int[] a, int[] b) {
        int[] joint = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            joint[i] = a[i] * 1_000_003 + b[i];
        }
        return Math.max(0, entropy(a) + entropy(b) - entropy(joint));
    }

    private static double[] column(double[][] x, int j) {
        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            v[i] = x[i][j];
        }
        return v;
    }

    /**
     * Concepts-task leakage.
     *
     * @param predicted predicted concept scores {@code [n][k]}
     * @param truth true concepts {@code [n][k]}
     * @param labels the task labels
     * @param bins bins for continuous scores, such as 8
     * @return the mean CTL over concepts, in {@code [0, 1]}
     */
    public static double ctl(double[][] predicted, double[][] truth, int[] labels, int bins) {
        int k = predicted[0].length;
        double hy = entropy(labels);
        double s = 0;
        for (int i = 0; i < k; i++) {
            int[] ph = discretise(column(predicted, i), bins);
            int[] ti = discretise(column(truth, i), bins);
            s += Math.max(0, mutualInformation(ph, labels) / hy - mutualInformation(ti, labels) / hy);
        }
        return s / k;
    }

    /**
     * Inter-concept leakage.
     *
     * @param predicted predicted concept scores {@code [n][k]}
     * @param truth true concepts {@code [n][k]}
     * @param bins bins for continuous scores
     * @param estimator the mutual-information estimator for the predicted scores
     * @return the mean ICL over ordered pairs, in {@code [0, 1]}
     */
    public static double icl(double[][] predicted, double[][] truth, int bins, Estimator estimator) {
        int k = predicted[0].length;
        int[][] ph = new int[k][];
        int[][] th = new int[k][];
        for (int i = 0; i < k; i++) {
            ph[i] = discretise(column(predicted, i), bins);
            th[i] = discretise(column(truth, i), bins);
        }
        double s = 0;
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                double hp = Math.sqrt(entropy(ph[i]) * entropy(ph[j]));
                double ht = Math.sqrt(entropy(th[i]) * entropy(th[j]));
                double ip;
                if (estimator == Estimator.KSG) {
                    ip = Math.max(0, MutualInformation.ksg(columnMatrix(predicted, i), columnMatrix(predicted, j), 3));
                } else {
                    ip = mutualInformation(ph[i], ph[j]);
                }
                double it = mutualInformation(th[i], th[j]);
                double a = hp > 0 ? ip / hp : 0;
                double b = ht > 0 ? it / ht : 0;
                s += Math.max(0, Math.min(1, a) - b);
            }
        }
        return s / (k * (k - 1) / 2.0);
    }

    private static double[][] columnMatrix(double[][] x, int j) {
        double[][] out = new double[x.length][1];
        for (int i = 0; i < x.length; i++) {
            out[i][0] = x[i][j] + 1e-9 * i;   // break ties for the k-NN estimator
        }
        return out;
    }
}
