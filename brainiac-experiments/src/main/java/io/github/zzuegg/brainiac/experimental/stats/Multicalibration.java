package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Post-hoc calibration on subgroups for a binary scorer.
 *
 * <ul>
 *   <li>{@link Method#MULTICALIBRATE} (Hebert-Johnson, Kim, Reingold and Rothblum, ICML 2018,
 *       <a href="https://arxiv.org/abs/1711.08513">arXiv:1711.08513</a>): with an explicit,
 *       possibly intersecting collection of groups, repeatedly finds the (group, prediction
 *       bin) cell with the largest mean residual above {@code alpha} and shifts the predictions
 *       in it by that residual. The patches are replayed in order on new data.</li>
 *   <li>{@link Method#MULTIACCURACY_BOOST} (Kim, Ghorbani and Zou, AIES 2019,
 *       <a href="https://arxiv.org/abs/1805.12317">arXiv:1805.12317</a>): without group labels,
 *       a small auditor ({@link Auditor#RIDGE} regression or a depth-limited regression
 *       {@link Auditor#TREE}) is fitted to the logistic working residuals on the features and
 *       added to the logit, as long as it finds a correlation above {@code alpha}.</li>
 * </ul>
 */
public final class Multicalibration {

    /** The method. */
    public enum Method {
        /** multicalibration with explicit groups */
        MULTICALIBRATE,
        /** multiaccuracy boosting with an auditor */
        MULTIACCURACY_BOOST
    }

    /** The auditor of {@link Method#MULTIACCURACY_BOOST}. */
    public enum Auditor {
        /** ridge regression on the features */
        RIDGE,
        /** a regression tree of depth 3 */
        TREE
    }

    private static final double EPS = 1e-4;

    private final Method method;
    private final int bins;
    private final List<int[]> cells = new ArrayList<>();
    private final List<Double> shifts = new ArrayList<>();
    private final List<ToDoubleFunction<double[]>> auditors = new ArrayList<>();
    private double eta;

    private Multicalibration(Method method, int bins) {
        this.method = method;
        this.bins = bins;
    }

    /**
     * Multicalibrates predictions on explicit groups.
     *
     * @param p calibration-set predictions
     * @param y calibration-set labels
     * @param groups group membership {@code [group][example]}
     * @param bins equal-width prediction bins
     * @param alpha the tolerated mean residual in a cell
     * @param minShare the smallest share of examples a patched cell must hold
     * @param maxRounds the most patches
     * @return the fitted patcher
     */
    public static Multicalibration multicalibrate(double[] p, int[] y, boolean[][] groups, int bins, double alpha,
            double minShare, int maxRounds) {
        Multicalibration mc = new Multicalibration(Method.MULTICALIBRATE, bins);
        double[] q = clip(p.clone());
        int n = q.length;
        for (int round = 0; round < maxRounds; round++) {
            double[][] sum = new double[groups.length][bins];
            int[][] cnt = new int[groups.length][bins];
            for (int i = 0; i < n; i++) {
                int b = mc.bin(q[i]);
                for (int g = 0; g < groups.length; g++) {
                    if (groups[g][i]) {
                        sum[g][b] += y[i] - q[i];
                        cnt[g][b]++;
                    }
                }
            }
            int bg = -1;
            int bb = -1;
            double worst = 0;
            for (int g = 0; g < groups.length; g++) {
                for (int b = 0; b < bins; b++) {
                    if (cnt[g][b] < Math.max(1, minShare * n)) {
                        continue;
                    }
                    double m = sum[g][b] / cnt[g][b];
                    // HKRR's size-weighted violation: |mean residual| beyond alpha, weighted by mass
                    if (Math.abs(m) > alpha && Math.abs(sum[g][b]) > worst) {
                        worst = Math.abs(sum[g][b]);
                        bg = g;
                        bb = b;
                    }
                }
            }
            if (bg < 0) {
                break;
            }
            double delta = sum[bg][bb] / cnt[bg][bb];
            mc.cells.add(new int[] {bg, bb});
            mc.shifts.add(delta);
            for (int i = 0; i < n; i++) {
                if (groups[bg][i] && mc.bin(q[i]) == bb) {
                    q[i] = clip(q[i] + delta);
                }
            }
        }
        return mc;
    }

    /**
     * Multiaccuracy boosting with an auditor on the features.
     *
     * @param p calibration-set predictions
     * @param y calibration-set labels
     * @param features calibration-set features the auditor sees
     * @param auditor the auditor
     * @param eta the step on the logit
     * @param alpha the smallest weighted correlation between auditor and residual to keep going
     * @param rounds the most rounds
     * @return the fitted booster
     */
    public static Multicalibration multiaccuracy(double[] p, int[] y, double[][] features, Auditor auditor,
            double eta, double alpha, int rounds) {
        Multicalibration mc = new Multicalibration(Method.MULTIACCURACY_BOOST, 0);
        mc.eta = eta;
        int n = p.length;
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = logit(clip(p[i]));
        }
        for (int round = 0; round < rounds; round++) {
            double[] w = new double[n];
            double[] r = new double[n];
            double[] res = new double[n];
            for (int i = 0; i < n; i++) {
                double q = sigmoid(z[i]);
                w[i] = Math.max(1e-6, q * (1 - q));
                res[i] = y[i] - q;
                r[i] = res[i] / w[i];
            }
            ToDoubleFunction<double[]> h = auditor == Auditor.RIDGE ? ridge(features, r, w, 1.0)
                    : tree(features, r, w, 3, Math.max(20, n / 50));
            double corr = 0;
            for (int i = 0; i < n; i++) {
                corr += res[i] * h.applyAsDouble(features[i]);
            }
            if (corr / n < alpha) {
                break;
            }
            mc.auditors.add(h);
            for (int i = 0; i < n; i++) {
                z[i] += eta * h.applyAsDouble(features[i]);
            }
        }
        return mc;
    }

    /**
     * Applies the fitted patches to new predictions.
     *
     * @param p predictions
     * @param groups group membership {@code [group][example]} (multicalibration)
     * @param features features (multiaccuracy)
     * @return the corrected predictions
     */
    public double[] apply(double[] p, boolean[][] groups, double[][] features) {
        double[] q = clip(p.clone());
        if (method == Method.MULTICALIBRATE) {
            for (int k = 0; k < cells.size(); k++) {
                int g = cells.get(k)[0];
                int b = cells.get(k)[1];
                double d = shifts.get(k);
                for (int i = 0; i < q.length; i++) {
                    if (groups[g][i] && bin(q[i]) == b) {
                        q[i] = clip(q[i] + d);
                    }
                }
            }
            return q;
        }
        for (int i = 0; i < q.length; i++) {
            double z = logit(q[i]);
            for (ToDoubleFunction<double[]> h : auditors) {
                z += eta * h.applyAsDouble(features[i]);
            }
            q[i] = sigmoid(z);
        }
        return q;
    }

    /**
     * The number of patches or boosting rounds.
     *
     * @return the rounds
     */
    public int rounds() {
        return method == Method.MULTICALIBRATE ? cells.size() : auditors.size();
    }

    private int bin(double q) {
        return Math.min(bins - 1, (int) (q * bins));
    }

    private static double clip(double q) {
        return Math.min(1 - EPS, Math.max(EPS, q));
    }

    private static double[] clip(double[] q) {
        for (int i = 0; i < q.length; i++) {
            q[i] = clip(q[i]);
        }
        return q;
    }

    private static double logit(double q) {
        return Math.log(q / (1 - q));
    }

    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    /** Weighted ridge regression with an unpenalised intercept. */
    static ToDoubleFunction<double[]> ridge(double[][] x, double[] r, double[] w, double lambda) {
        int d = x[0].length + 1;
        double[][] a = new double[d][d];
        double[] b = new double[d];
        double[] row = new double[d];
        double tw = 0;
        for (int i = 0; i < x.length; i++) {
            row[0] = 1;
            System.arraycopy(x[i], 0, row, 1, d - 1);
            for (int j = 0; j < d; j++) {
                b[j] += w[i] * row[j] * r[i];
                for (int k = 0; k < d; k++) {
                    a[j][k] += w[i] * row[j] * row[k];
                }
            }
            tw += w[i];
        }
        for (int j = 1; j < d; j++) {
            a[j][j] += lambda * tw / x.length;
        }
        a[0][0] += 1e-9;
        double[] beta = Recalibrators.solve(a, b);
        return f -> {
            double s = beta[0];
            for (int j = 1; j < beta.length; j++) {
                s += beta[j] * f[j - 1];
            }
            return s;
        };
    }

    /** A weighted least-squares regression tree. */
    static ToDoubleFunction<double[]> tree(double[][] x, double[] r, double[] w, int depth, int minLeaf) {
        int[] idx = new int[x.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        return grow(x, r, w, idx, depth, minLeaf);
    }

    private static ToDoubleFunction<double[]> grow(double[][] x, double[] r, double[] w, int[] idx, int depth,
            int minLeaf) {
        double sw = 0;
        double swr = 0;
        for (int i : idx) {
            sw += w[i];
            swr += w[i] * r[i];
        }
        double leaf = swr / Math.max(sw, 1e-12);
        if (depth == 0 || idx.length < 2 * minLeaf) {
            return f -> leaf;
        }
        int bestF = -1;
        double bestT = 0;
        double bestGain = 1e-12;
        Integer[] o = new Integer[idx.length];
        for (int f = 0; f < x[0].length; f++) {
            for (int k = 0; k < idx.length; k++) {
                o[k] = idx[k];
            }
            int ff = f;
            Arrays.sort(o, (a, b) -> Double.compare(x[a][ff], x[b][ff]));
            double lw = 0;
            double lwr = 0;
            for (int k = 0; k < o.length - 1; k++) {
                lw += w[o[k]];
                lwr += w[o[k]] * r[o[k]];
                if (k + 1 < minLeaf || o.length - k - 1 < minLeaf || x[o[k]][f] == x[o[k + 1]][f]) {
                    continue;
                }
                double rw = sw - lw;
                double rwr = swr - lwr;
                double gain = lwr * lwr / lw + rwr * rwr / rw - swr * swr / sw;
                if (gain > bestGain) {
                    bestGain = gain;
                    bestF = f;
                    bestT = 0.5 * (x[o[k]][f] + x[o[k + 1]][f]);
                }
            }
        }
        if (bestF < 0) {
            return f -> leaf;
        }
        int bf = bestF;
        double bt = bestT;
        int[] left = Arrays.stream(idx).filter(i -> x[i][bf] <= bt).toArray();
        int[] right = Arrays.stream(idx).filter(i -> x[i][bf] > bt).toArray();
        ToDoubleFunction<double[]> l = grow(x, r, w, left, depth - 1, minLeaf);
        ToDoubleFunction<double[]> rr = grow(x, r, w, right, depth - 1, minLeaf);
        return f -> f[bf] <= bt ? l.applyAsDouble(f) : rr.applyAsDouble(f);
    }
}
