package io.github.zzuegg.brainiac.experimental.stats;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Resampling for imbalanced binary classification (label 1 is the minority), and the prior-shift
 * correction that undoes what resampling does to predicted probabilities.
 *
 * <ul>
 *   <li>{@link Method#SMOTE}: interpolate each minority point towards one of its {@code k}
 *       minority neighbours until the classes balance.</li>
 *   <li>{@link Method#BORDERLINE_SMOTE} (Han, Wang and Mao, ICIC 2005): SMOTE only from
 *       "danger" minority points, those with at least half (but not all) of their {@code k}
 *       neighbours in the majority.</li>
 *   <li>{@link Method#ADASYN} (He et al., IJCNN 2008): synthetic counts proportional to each
 *       minority point's share of majority neighbours.</li>
 *   <li>{@link Method#TOMEK}: drop majority points in Tomek links (mutual nearest neighbours of
 *       opposite class); {@link Method#ENN}: drop majority points misclassified by their
 *       {@code k} neighbours (Wilson's edited nearest neighbours).</li>
 *   <li>{@link Method#RANDOM_UNDER}: subsample the majority to the minority size;
 *       {@link #easyEnsemble} returns several such balanced subsets for balanced bagging (Liu, Wu
 *       and Zhou, IEEE TSMC-B 2009).</li>
 *   <li>{@link #priorShift} maps a probability learnt at training prior {@code pi_s} back to the
 *       deployment prior {@code pi}: {@code p' = p r / (p r + (1 - p) r0)} with
 *       {@code r = pi / pi_s}, {@code r0 = (1 - pi) / (1 - pi_s)} (the
 *       {@code PRIOR_SHIFT_CORRECTION}).</li>
 * </ul>
 * Liu (2026, arXiv:2606.29720) finds that resampling leaves ranking metrics flat while
 * miscalibrating probabilities; the correction repairs undersampling but not SMOTE.
 */
public final class Resampling {

    /** The resampling method. */
    public enum Method {
        /** SMOTE oversampling. */
        SMOTE,
        /** Borderline-SMOTE. */
        BORDERLINE_SMOTE,
        /** ADASYN. */
        ADASYN,
        /** Tomek-link cleaning. */
        TOMEK,
        /** Edited nearest neighbours. */
        ENN,
        /** Random undersampling to balance. */
        RANDOM_UNDER
    }

    /**
     * A resampled training set.
     *
     * @param x rows
     * @param y labels in {0, 1}
     */
    public record Sample(double[][] x, int[] y) {

        /**
         * The positive share.
         *
         * @return the minority prior of this sample
         */
        public double positiveRate() {
            double s = 0;
            for (int v : y) {
                s += v;
            }
            return s / y.length;
        }
    }

    private Resampling() {
    }

    /**
     * Resamples a training set.
     *
     * @param method the method
     * @param x rows
     * @param y labels (1 = minority)
     * @param k neighbours
     * @param seed the seed
     * @return the new set
     */
    public static Sample resample(Method method, double[][] x, int[] y, int k, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] minority = indices(y, 1);
        int[] majority = indices(y, 0);
        int need = majority.length - minority.length;
        return switch (method) {
            case SMOTE -> {
                double[] w = new double[minority.length];
                Arrays.fill(w, 1);
                yield oversample(x, y, minority, w, need, k, r);
            }
            case BORDERLINE_SMOTE -> {
                double[] w = new double[minority.length];
                for (int a = 0; a < minority.length; a++) {
                    int maj = majorityNeighbours(x, y, minority[a], k);
                    w[a] = maj >= k / 2.0 && maj < k ? 1 : 0;
                }
                if (Arrays.stream(w).sum() == 0) {
                    Arrays.fill(w, 1);
                }
                yield oversample(x, y, minority, w, need, k, r);
            }
            case ADASYN -> {
                double[] w = new double[minority.length];
                for (int a = 0; a < minority.length; a++) {
                    w[a] = majorityNeighbours(x, y, minority[a], k) / (double) k;
                }
                if (Arrays.stream(w).sum() == 0) {
                    Arrays.fill(w, 1);
                }
                yield oversample(x, y, minority, w, need, k, r);
            }
            case TOMEK -> {
                boolean[] drop = new boolean[y.length];
                for (int i = 0; i < y.length; i++) {
                    int j = nearest(x, i, -1)[0];
                    if (y[i] != y[j] && nearest(x, j, -1)[0] == i) {
                        drop[y[i] == 0 ? i : j] = true;
                    }
                }
                yield keep(x, y, drop);
            }
            case ENN -> {
                boolean[] drop = new boolean[y.length];
                for (int i : majority) {
                    drop[i] = majorityNeighboursOf(x, y, i, k, 0) * 2 < k;
                }
                yield keep(x, y, drop);
            }
            case RANDOM_UNDER -> easyEnsemble(x, y, 1, seed).get(0);
        };
    }

    /**
     * Balanced subsets for EasyEnsemble / balanced bagging: all minority rows plus an equal
     * random draw of majority rows, per member.
     *
     * @param x rows
     * @param y labels
     * @param members subsets
     * @param seed the seed
     * @return the subsets
     */
    public static List<Sample> easyEnsemble(double[][] x, int[] y, int members, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] minority = indices(y, 1);
        int[] majority = indices(y, 0);
        List<Sample> out = new ArrayList<>();
        for (int m = 0; m < members; m++) {
            int[] perm = majority.clone();
            for (int i = perm.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }
            int take = Math.min(minority.length, majority.length);
            double[][] sx = new double[minority.length + take][];
            int[] sy = new int[minority.length + take];
            for (int a = 0; a < minority.length; a++) {
                sx[a] = x[minority[a]];
                sy[a] = 1;
            }
            for (int a = 0; a < take; a++) {
                sx[minority.length + a] = x[perm[a]];
            }
            out.add(new Sample(sx, sy));
        }
        return out;
    }

    /**
     * The prior-shift correction of a probability.
     *
     * @param p the probability under the training prior
     * @param trainPrior the positive share the model was trained on
     * @param truePrior the deployment positive share
     * @return the corrected probability
     */
    public static double priorShift(double p, double trainPrior, double truePrior) {
        double a = p * truePrior / trainPrior;
        double b = (1 - p) * (1 - truePrior) / (1 - trainPrior);
        return a / (a + b);
    }

    private static Sample oversample(double[][] x, int[] y, int[] minority, double[] weight, int need, int k,
            SplittableRandom r) {
        int d = x[0].length;
        double total = Arrays.stream(weight).sum();
        double[][] nx = Arrays.copyOf(x, x.length + Math.max(0, need));
        int[] ny = Arrays.copyOf(y, y.length + Math.max(0, need));
        int[][] nbCache = new int[minority.length][];
        for (int s = 0; s < need; s++) {
            double u = r.nextDouble() * total;
            int a = 0;
            while (a < minority.length - 1 && (u -= weight[a]) > 0) {
                a++;
            }
            if (nbCache[a] == null) {
                nbCache[a] = minorityNeighbours(x, minority, a, k);
            }
            int b = nbCache[a][r.nextInt(nbCache[a].length)];
            double g = r.nextDouble();
            double[] row = new double[d];
            for (int j = 0; j < d; j++) {
                row[j] = x[minority[a]][j] + g * (x[b][j] - x[minority[a]][j]);
            }
            nx[x.length + s] = row;
            ny[y.length + s] = 1;
        }
        return new Sample(nx, ny);
    }

    private static int[] minorityNeighbours(double[][] x, int[] minority, int a, int k) {
        int kk = Math.min(k, minority.length - 1);
        if (kk <= 0) {
            return new int[] {minority[a]};
        }
        double[] best = new double[kk];
        int[] idx = new int[kk];
        Arrays.fill(best, Double.POSITIVE_INFINITY);
        for (int b : minority) {
            if (b == minority[a]) {
                continue;
            }
            double dd = dist(x[minority[a]], x[b]);
            insert(best, idx, dd, b);
        }
        return idx;
    }

    private static int majorityNeighbours(double[][] x, int[] y, int i, int k) {
        return majorityNeighboursOf(x, y, i, k, 0);
    }

    private static int majorityNeighboursOf(double[][] x, int[] y, int i, int k, int label) {
        int c = 0;
        for (int j : nearest(x, i, k)) {
            if (y[j] == label) {
                c++;
            }
        }
        return c;
    }

    /** The k (or 1 when k < 1) nearest rows to row i, excluding i. */
    private static int[] nearest(double[][] x, int i, int k) {
        int kk = Math.max(1, k);
        double[] best = new double[kk];
        int[] idx = new int[kk];
        Arrays.fill(best, Double.POSITIVE_INFINITY);
        for (int j = 0; j < x.length; j++) {
            if (j != i) {
                insert(best, idx, dist(x[i], x[j]), j);
            }
        }
        return idx;
    }

    private static void insert(double[] best, int[] idx, double d, int j) {
        int k = best.length;
        if (d >= best[k - 1]) {
            return;
        }
        int a = k - 1;
        while (a > 0 && best[a - 1] > d) {
            best[a] = best[a - 1];
            idx[a] = idx[a - 1];
            a--;
        }
        best[a] = d;
        idx[a] = j;
    }

    private static double dist(double[] a, double[] b) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            s += (a[j] - b[j]) * (a[j] - b[j]);
        }
        return s;
    }

    private static Sample keep(double[][] x, int[] y, boolean[] drop) {
        List<double[]> kx = new ArrayList<>();
        List<Integer> ky = new ArrayList<>();
        for (int i = 0; i < y.length; i++) {
            if (!drop[i]) {
                kx.add(x[i]);
                ky.add(y[i]);
            }
        }
        return new Sample(kx.toArray(new double[0][]), ky.stream().mapToInt(Integer::intValue).toArray());
    }

    private static int[] indices(int[] y, int label) {
        return java.util.stream.IntStream.range(0, y.length).filter(i -> y[i] == label).toArray();
    }
}
