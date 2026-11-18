package io.github.zzuegg.brainiac.experiments.statistics;

import java.util.Arrays;

/** Small scoring helpers shared by the statistics experiments. */
final class Scores {

    private Scores() {
    }

    static double f1(boolean[] estimate, boolean[] truth) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int j = 0; j < truth.length; j++) {
            if (estimate[j] && truth[j]) {
                tp++;
            } else if (estimate[j]) {
                fp++;
            } else if (truth[j]) {
                fn++;
            }
        }
        return tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
    }

    static int falsePositives(boolean[] estimate, boolean[] truth) {
        int fp = 0;
        for (int j = 0; j < truth.length; j++) {
            if (estimate[j] && !truth[j]) {
                fp++;
            }
        }
        return fp;
    }

    static double mse(double[] prediction, double[] y) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            s += (prediction[i] - y[i]) * (prediction[i] - y[i]);
        }
        return s / y.length;
    }

    static double[] predict(double[][] x, double intercept, double[] beta) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double s = intercept;
            for (int j = 0; j < beta.length; j++) {
                s += x[i][j] * beta[j];
            }
            out[i] = s;
        }
        return out;
    }

    static double distance(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return Math.sqrt(s);
    }

    static boolean[] nonZero(double[] beta) {
        boolean[] s = new boolean[beta.length];
        for (int j = 0; j < beta.length; j++) {
            s[j] = beta[j] != 0;
        }
        return s;
    }

    /** Area under the ROC curve (ties count half). */
    static double auc(double[] score, boolean[] positive) {
        Integer[] o = new Integer[score.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = i;
        }
        Arrays.sort(o, (a, b) -> Double.compare(score[a], score[b]));
        double rankSum = 0;
        int pos = 0;
        int i = 0;
        while (i < o.length) {
            int j = i;
            while (j + 1 < o.length && score[o[j + 1]] == score[o[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) {
                if (positive[o[k]]) {
                    rankSum += rank;
                    pos++;
                }
            }
            i = j + 1;
        }
        int neg = score.length - pos;
        if (pos == 0 || neg == 0) {
            return 0.5;
        }
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    static double auc(double[] score, int[] label) {
        boolean[] pos = new boolean[label.length];
        for (int i = 0; i < label.length; i++) {
            pos[i] = label[i] == 1;
        }
        return auc(score, pos);
    }

    /** Kendall's tau-a between two score vectors. */
    static double kendall(double[] a, double[] b) {
        int n = a.length;
        long c = 0;
        long d = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = Math.signum(a[i] - a[j]) * Math.signum(b[i] - b[j]);
                if (s > 0) {
                    c++;
                } else if (s < 0) {
                    d++;
                }
            }
        }
        return (c - d) / (n * (n - 1) / 2.0);
    }

    /** Binary expected calibration error with equal-width bins. */
    static double ece(double[] prob, int[] y, int bins) {
        double[] sp = new double[bins];
        double[] sy = new double[bins];
        int[] cnt = new int[bins];
        for (int i = 0; i < prob.length; i++) {
            int b = Math.min(bins - 1, (int) (prob[i] * bins));
            sp[b] += prob[i];
            sy[b] += y[i];
            cnt[b]++;
        }
        double e = 0;
        for (int b = 0; b < bins; b++) {
            if (cnt[b] > 0) {
                e += Math.abs(sp[b] - sy[b]);
            }
        }
        return e / prob.length;
    }

    static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    static double[][] standardise(double[][] train, double[][] other) {
        int p = train[0].length;
        double[] m = new double[p];
        double[] sd = new double[p];
        for (double[] r : train) {
            for (int j = 0; j < p; j++) {
                m[j] += r[j] / train.length;
            }
        }
        for (double[] r : train) {
            for (int j = 0; j < p; j++) {
                sd[j] += (r[j] - m[j]) * (r[j] - m[j]) / train.length;
            }
        }
        double[][] out = new double[other.length][p];
        for (int i = 0; i < other.length; i++) {
            for (int j = 0; j < p; j++) {
                out[i][j] = (other[i][j] - m[j]) / Math.sqrt(sd[j] + 1e-12);
            }
        }
        return out;
    }
}
