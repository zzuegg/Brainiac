package io.github.zzuegg.brainiac.experimental.causal;

import java.util.SplittableRandom;

/**
 * K-fold cross-fitting (Chernozhukov et al. 2018, <a href="https://arxiv.org/abs/1608.00060">
 * arXiv:1608.00060</a>): a nuisance model is trained on K-1 folds and evaluated only on the
 * held-out fold, so every unit's nuisance prediction comes from a model that never saw it. This is
 * what keeps an overfitting learner (a deep forest, say) from leaking its own noise into the effect
 * estimate; it is shared by the R- and DR-learners, double machine learning, causal forests and
 * any statistic that plugs in fitted values.
 */
public final class CrossFitting {

    private CrossFitting() {
    }

    /**
     * Random fold labels of near-equal size.
     *
     * @param n units
     * @param k folds, at least 2
     * @param seed the seed
     * @return the fold of every unit, in {@code 0..k-1}
     */
    public static int[] folds(int n, int k, long seed) {
        if (k < 2) {
            throw new IllegalArgumentException("need at least 2 folds");
        }
        SplittableRandom r = new SplittableRandom(seed);
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        int[] fold = new int[n];
        for (int i = 0; i < n; i++) {
            fold[perm[i]] = i % k;
        }
        return fold;
    }

    /**
     * Out-of-fold predictions of a regressor.
     *
     * @param x rows
     * @param y targets
     * @param weights sample weights or null
     * @param model the regressor
     * @param fold each row's fold
     * @return for every row, the prediction of the model fitted without its fold
     */
    public static double[] predict(double[][] x, double[] y, double[] weights, Regressor model, int[] fold) {
        int k = 0;
        for (int f : fold) {
            k = Math.max(k, f + 1);
        }
        double[] out = new double[x.length];
        for (int f = 0; f < k; f++) {
            int train = 0;
            for (int g : fold) {
                train += g != f ? 1 : 0;
            }
            double[][] tx = new double[train][];
            double[] ty = new double[train];
            double[] tw = weights == null ? null : new double[train];
            int c = 0;
            for (int i = 0; i < x.length; i++) {
                if (fold[i] != f) {
                    tx[c] = x[i];
                    ty[c] = y[i];
                    if (tw != null) {
                        tw[c] = weights[i];
                    }
                    c++;
                }
            }
            Regressor.Fitted fit = model.fit(tx, ty, tw);
            for (int i = 0; i < x.length; i++) {
                if (fold[i] == f) {
                    out[i] = fit.predict(x[i]);
                }
            }
        }
        return out;
    }

    /**
     * Out-of-fold predictions for the rows of one treatment arm only, evaluated on every row: the
     * cross-fitted outcome model {@code mu_arm(x)}.
     *
     * @param x rows
     * @param t treatments
     * @param y outcomes
     * @param arm the arm whose rows train the model
     * @param model the regressor
     * @param fold each row's fold
     * @return {@code mu_arm(x_i)} from the model fitted without row {@code i}'s fold
     */
    public static double[] predictArm(double[][] x, int[] t, double[] y, int arm, Regressor model, int[] fold) {
        int k = 0;
        for (int f : fold) {
            k = Math.max(k, f + 1);
        }
        double[] out = new double[x.length];
        for (int f = 0; f < k; f++) {
            int train = 0;
            for (int i = 0; i < x.length; i++) {
                train += fold[i] != f && t[i] == arm ? 1 : 0;
            }
            double[][] tx = new double[train][];
            double[] ty = new double[train];
            int c = 0;
            for (int i = 0; i < x.length; i++) {
                if (fold[i] != f && t[i] == arm) {
                    tx[c] = x[i];
                    ty[c++] = y[i];
                }
            }
            Regressor.Fitted fit = model.fit(tx, ty, null);
            for (int i = 0; i < x.length; i++) {
                if (fold[i] == f) {
                    out[i] = fit.predict(x[i]);
                }
            }
        }
        return out;
    }

    /**
     * @param t binary values
     * @return them as doubles
     */
    public static double[] asDouble(int[] t) {
        double[] out = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            out[i] = t[i];
        }
        return out;
    }
}
