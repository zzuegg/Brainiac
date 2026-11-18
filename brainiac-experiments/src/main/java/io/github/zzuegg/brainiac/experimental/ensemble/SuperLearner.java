package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Stacking (Wolpert 1992) and the super learner (van der Laan, Polley &amp; Hubbard 2007): fit a
 * library of base learners in k folds, collect their out-of-fold predictions, and learn how to
 * combine them from those predictions only. The super learner restricts the combination to a
 * convex one, which comes with an oracle inequality: asymptotically it does as well as the best
 * convex combination of the library.
 *
 * <ul>
 *   <li>{@link Meta#DISCRETE}: the "discrete super learner", the single member with the lowest
 *       cross-validated risk;</li>
 *   <li>{@link Meta#UNIFORM}: the plain average;</li>
 *   <li>{@link Meta#CONVEX}: non-negative least squares normalised to the simplex (regression)
 *       or simplex weights minimising the log loss of the probability mixture, by EM
 *       (classification);</li>
 *   <li>{@link Meta#STACKER}: an unconstrained ridge regression with intercept on the members'
 *       predictions (regression) or a multinomial logistic regression on their log-probabilities
 *       (classification).</li>
 * </ul>
 *
 * After the meta-learner is fitted, every member is refitted on all rows.
 */
public final class SuperLearner {

    /** A base learner. */
    @FunctionalInterface
    public interface Learner {

        /**
         * @param x rows
         * @param y targets (regression) or class ids (classification)
         * @return the fitted predictor
         */
        Predictor fit(double[][] x, double[] y);
    }

    /** A fitted base learner. */
    @FunctionalInterface
    public interface Predictor {

        /**
         * @param x a row
         * @return {@code {value}} (regression) or class probabilities (classification)
         */
        double[] predict(double[] x);
    }

    /** The task. */
    public enum Mode {
        /** squared error */
        REGRESSION,
        /** log loss */
        CLASSIFICATION
    }

    /** The meta-learner. */
    public enum Meta {
        /** best single member by CV */
        DISCRETE,
        /** uniform average */
        UNIFORM,
        /** convex weights */
        CONVEX,
        /** unconstrained stacker */
        STACKER
    }

    private final Mode mode;
    private final Meta meta;
    private final List<Predictor> members;
    private final double[] weights;
    private final double[][] stacker;
    private final double[] risk;

    private SuperLearner(Mode mode, Meta meta, List<Predictor> members, double[] weights, double[][] stacker,
            double[] risk) {
        this.mode = mode;
        this.meta = meta;
        this.members = members;
        this.weights = weights;
        this.stacker = stacker;
        this.risk = risk;
    }

    /**
     * Fits the library in folds, the meta-learner on the out-of-fold predictions, and refits the
     * members on all rows.
     *
     * @param mode the task
     * @param meta the meta-learner
     * @param library the base learners
     * @param x rows
     * @param y targets or class ids
     * @param classes the number of classes (ignored for regression)
     * @param folds the number of folds
     * @param seed the fold-assignment seed
     * @return the fitted super learner
     */
    public static SuperLearner fit(Mode mode, Meta meta, List<Learner> library, double[][] x, double[] y, int classes,
            int folds, long seed) {
        double[][][] oof = outOfFold(library, x, y, folds, seed);
        return fromOutOfFold(mode, meta, library, oof, x, y, classes);
    }

    /**
     * Out-of-fold predictions of every member.
     *
     * @param library the base learners
     * @param x rows
     * @param y targets
     * @param folds the number of folds
     * @param seed the seed
     * @return {@code [member][row][output]}
     */
    public static double[][][] outOfFold(List<Learner> library, double[][] x, double[] y, int folds, long seed) {
        int n = x.length;
        int[] fold = new int[n];
        int[] perm = permutation(n, new SplittableRandom(seed));
        for (int i = 0; i < n; i++) {
            fold[perm[i]] = i % folds;
        }
        double[][][] oof = new double[library.size()][n][];
        for (int f = 0; f < folds; f++) {
            List<Integer> tr = new ArrayList<>();
            List<Integer> te = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                (fold[i] == f ? te : tr).add(i);
            }
            double[][] tx = new double[tr.size()][];
            double[] ty = new double[tr.size()];
            for (int i = 0; i < tr.size(); i++) {
                tx[i] = x[tr.get(i)];
                ty[i] = y[tr.get(i)];
            }
            for (int m = 0; m < library.size(); m++) {
                Predictor p = library.get(m).fit(tx, ty);
                for (int i : te) {
                    oof[m][i] = p.predict(x[i]);
                }
            }
        }
        return oof;
    }

    /**
     * Fits the meta-learner on given out-of-fold predictions and refits the members on all rows.
     *
     * @param mode the task
     * @param meta the meta-learner
     * @param library the base learners
     * @param oof {@code [member][row][output]}
     * @param x rows
     * @param y targets
     * @param classes the number of classes
     * @return the fitted super learner
     */
    public static SuperLearner fromOutOfFold(Mode mode, Meta meta, List<Learner> library, double[][][] oof,
            double[][] x, double[] y, int classes) {
        int m = library.size();
        int n = x.length;
        double[] risk = new double[m];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                risk[j] += loss(mode, oof[j][i], y[i]) / n;
            }
        }
        double[] w = new double[m];
        double[][] stack = null;
        switch (meta) {
            case DISCRETE -> {
                int best = 0;
                for (int j = 1; j < m; j++) {
                    best = risk[j] < risk[best] ? j : best;
                }
                w[best] = 1;
            }
            case UNIFORM -> java.util.Arrays.fill(w, 1.0 / m);
            case CONVEX -> {
                if (mode == Mode.REGRESSION) {
                    double[][] z = new double[n][m];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < m; j++) {
                            z[i][j] = oof[j][i][0];
                        }
                    }
                    w = nnls(z, y, 500);
                    double s = 0;
                    for (double v : w) {
                        s += v;
                    }
                    for (int j = 0; j < m; j++) {
                        w[j] = s > 0 ? w[j] / s : 1.0 / m;
                    }
                } else {
                    int[] labels = new int[n];
                    for (int i = 0; i < n; i++) {
                        labels[i] = (int) y[i];
                    }
                    w = simplexLogLoss(oof, labels, 300);
                }
            }
            case STACKER -> stack = mode == Mode.REGRESSION ? ridgeStacker(oof, y, 1e-3)
                    : logisticStacker(oof, y, classes, 1e-3, 400);
        }
        List<Predictor> full = new ArrayList<>();
        for (Learner l : library) {
            full.add(l.fit(x, y));
        }
        return new SuperLearner(mode, meta, full, w, stack, risk);
    }

    private static double loss(Mode mode, double[] p, double y) {
        if (mode == Mode.REGRESSION) {
            double d = p[0] - y;
            return d * d;
        }
        return -Math.log(Math.max(1e-12, p[(int) y]));
    }

    /**
     * Predicts one row.
     *
     * @param x the row
     * @return {@code {value}} or class probabilities
     */
    public double[] predict(double[] x) {
        double[][] p = new double[members.size()][];
        for (int j = 0; j < p.length; j++) {
            p[j] = members.get(j).predict(x);
        }
        if (meta == Meta.STACKER) {
            return mode == Mode.REGRESSION ? new double[] {applyRidge(stacker, p)} : applyLogistic(stacker, p);
        }
        double[] out = new double[p[0].length];
        for (int j = 0; j < p.length; j++) {
            for (int c = 0; c < out.length; c++) {
                out[c] += weights[j] * p[j][c];
            }
        }
        return out;
    }

    /** @return the meta weights of the members (zeros for the stacker) */
    public double[] weights() {
        return weights.clone();
    }

    /** @return each member's cross-validated risk (MSE or log loss) */
    public double[] crossValidatedRisk() {
        return risk.clone();
    }

    // ------------------------------------------------------------------ meta solvers

    /**
     * Non-negative least squares by cyclic coordinate descent.
     *
     * @param z the design {@code [n][m]}
     * @param y the targets
     * @param sweeps coordinate sweeps
     * @return {@code w >= 0} minimising {@code |z w - y|²}
     */
    public static double[] nnls(double[][] z, double[] y, int sweeps) {
        int n = z.length;
        int m = z[0].length;
        double[][] g = new double[m][m];
        double[] b = new double[m];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < m; a++) {
                b[a] += z[i][a] * y[i];
                for (int c = 0; c < m; c++) {
                    g[a][c] += z[i][a] * z[i][c];
                }
            }
        }
        double[] w = new double[m];
        java.util.Arrays.fill(w, 1.0 / m);
        for (int s = 0; s < sweeps; s++) {
            for (int a = 0; a < m; a++) {
                double r = b[a];
                for (int c = 0; c < m; c++) {
                    if (c != a) {
                        r -= g[a][c] * w[c];
                    }
                }
                w[a] = g[a][a] > 0 ? Math.max(0, r / g[a][a]) : 0;
            }
        }
        return w;
    }

    /**
     * Simplex weights that minimise the log loss of the mixture {@code Σ_m w_m p_m}, by the EM
     * update for mixture weights {@code w_m <- w_m mean_i(p_{m,i,y_i} / Σ_j w_j p_{j,i,y_i})},
     * which never increases the loss.
     *
     * @param p {@code [member][sample][class]}
     * @param labels the labels
     * @param iterations EM iterations
     * @return the weights
     */
    public static double[] simplexLogLoss(double[][][] p, int[] labels, int iterations) {
        int m = p.length;
        int n = labels.length;
        double[] w = new double[m];
        java.util.Arrays.fill(w, 1.0 / m);
        for (int it = 0; it < iterations; it++) {
            double[] next = new double[m];
            for (int i = 0; i < n; i++) {
                double mix = 0;
                for (int j = 0; j < m; j++) {
                    mix += w[j] * Math.max(1e-12, p[j][i][labels[i]]);
                }
                for (int j = 0; j < m; j++) {
                    next[j] += w[j] * Math.max(1e-12, p[j][i][labels[i]]) / mix / n;
                }
            }
            w = next;
        }
        return w;
    }

    private static double[][] ridgeStacker(double[][][] oof, double[] y, double ridge) {
        int m = oof.length;
        int n = y.length;
        int d = m + 1;
        double[][] g = new double[d][d];
        double[] b = new double[d];
        for (int i = 0; i < n; i++) {
            double[] f = new double[d];
            f[0] = 1;
            for (int j = 0; j < m; j++) {
                f[j + 1] = oof[j][i][0];
            }
            for (int a = 0; a < d; a++) {
                b[a] += f[a] * y[i];
                for (int c = 0; c < d; c++) {
                    g[a][c] += f[a] * f[c];
                }
            }
        }
        for (int a = 1; a < d; a++) {
            g[a][a] += ridge * n;
        }
        return new double[][] {Recalibrators.solve(g, b)};
    }

    private static double applyRidge(double[][] coef, double[][] p) {
        double s = coef[0][0];
        for (int j = 0; j < p.length; j++) {
            s += coef[0][j + 1] * p[j][0];
        }
        return s;
    }

    private static double[] stackFeatures(double[][] p) {
        int k = p[0].length;
        double[] f = new double[p.length * k + 1];
        for (int j = 0; j < p.length; j++) {
            for (int c = 0; c < k; c++) {
                f[j * k + c] = Math.log(Math.max(1e-6, p[j][c]));
            }
        }
        f[f.length - 1] = 1;
        return f;
    }

    private static double[][] logisticStacker(double[][][] oof, double[] y, int classes, double ridge,
            int iterations) {
        int n = y.length;
        double[][] f = new double[n][];
        double[][] rowP = new double[oof.length][];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < oof.length; j++) {
                rowP[j] = oof[j][i];
            }
            f[i] = stackFeatures(rowP);
        }
        int d = f[0].length;
        double[][] w = new double[classes][d];
        double[][] m1 = new double[classes][d];
        double[][] m2 = new double[classes][d];
        for (int it = 1; it <= iterations; it++) {
            double[][] g = new double[classes][d];
            for (int i = 0; i < n; i++) {
                double[] p = softmax(w, f[i]);
                for (int c = 0; c < classes; c++) {
                    double e = (p[c] - (c == (int) y[i] ? 1 : 0)) / n;
                    for (int a = 0; a < d; a++) {
                        g[c][a] += e * f[i][a];
                    }
                }
            }
            double c1 = 1 - Math.pow(0.9, it);
            double c2 = 1 - Math.pow(0.999, it);
            for (int c = 0; c < classes; c++) {
                for (int a = 0; a < d; a++) {
                    double gr = g[c][a] + ridge * w[c][a];
                    m1[c][a] = 0.9 * m1[c][a] + 0.1 * gr;
                    m2[c][a] = 0.999 * m2[c][a] + 0.001 * gr * gr;
                    w[c][a] -= 0.05 * (m1[c][a] / c1) / (Math.sqrt(m2[c][a] / c2) + 1e-8);
                }
            }
        }
        return w;
    }

    private static double[] applyLogistic(double[][] w, double[][] p) {
        return softmax(w, stackFeatures(p));
    }

    private static double[] softmax(double[][] w, double[] f) {
        double[] z = new double[w.length];
        double max = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < w.length; c++) {
            for (int a = 0; a < f.length; a++) {
                z[c] += w[c][a] * f[a];
            }
            max = Math.max(max, z[c]);
        }
        double s = 0;
        for (int c = 0; c < z.length; c++) {
            z[c] = Math.exp(z[c] - max);
            s += z[c];
        }
        for (int c = 0; c < z.length; c++) {
            z[c] /= s;
        }
        return z;
    }

    static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
