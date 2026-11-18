package io.github.zzuegg.brainiac.experimental.causal;

/**
 * Double/debiased machine learning (Chernozhukov et al., Econometrics Journal 2018,
 * <a href="https://arxiv.org/abs/1608.00060">arXiv:1608.00060</a>) for an average treatment effect
 * with flexible nuisance models.
 *
 * <p>Plugging a regularised learner into an effect formula leaves a bias that shrinks slower than
 * {@code 1/sqrt(n)}. DML fixes it with two ingredients: a <em>Neyman-orthogonal score</em>, whose
 * expectation is insensitive to first-order errors in the nuisances, and <em>cross-fitting</em>
 * ({@link CrossFitting}), which evaluates every nuisance on units it was not trained on. The result
 * is a {@code sqrt(n)}-consistent, asymptotically normal estimate with an analytic standard error.
 * <ul>
 *   <li>{@link #partiallyLinear}: {@code y = theta t + g(x) + e}; the estimate is the regression of
 *       the outcome residual {@code y - l(x)} on the treatment residual {@code t - m(x)}, with the
 *       sandwich standard error of the score {@code (y~ - theta t~) t~}.</li>
 *   <li>{@link #interactive}: binary treatment with arbitrary heterogeneity; the AIPW score, whose
 *       mean is the ATE and whose standard deviation gives the standard error.</li>
 * </ul>
 * Setting {@code crossFit = false} fits the nuisances once on all units and reuses them (the
 * plain plug-in AIPW/PLR), the ablation that shows why cross-fitting matters for learners that
 * overfit.
 */
public final class DoubleMl {

    /**
     * An effect estimate with a normal 95 % interval.
     *
     * @param theta the estimate
     * @param se its standard error
     */
    public record Estimate(double theta, double se) {

        /** @return the lower end of the 95 % interval */
        public double lower() {
            return theta - 1.959964 * se;
        }

        /** @return the upper end of the 95 % interval */
        public double upper() {
            return theta + 1.959964 * se;
        }

        /**
         * @param truth the true value
         * @return whether the 95 % interval covers it
         */
        public boolean covers(double truth) {
            return lower() <= truth && truth <= upper();
        }
    }

    private DoubleMl() {
    }

    private static double[] nuisance(double[][] x, double[] y, Regressor model, int[] fold, boolean crossFit) {
        if (crossFit) {
            return CrossFitting.predict(x, y, null, model, fold);
        }
        return model.fit(x, y, null).predict(x);
    }

    /**
     * The partially linear model {@code y = theta t + g(x) + e}, {@code t = m(x) + v}.
     *
     * @param x covariates
     * @param t treatments (binary or continuous)
     * @param y outcomes
     * @param outcomeModel the model of {@code l(x) = E[y | x]}
     * @param treatmentModel the model of {@code m(x) = E[t | x]}
     * @param folds cross-fitting folds
     * @param crossFit whether to cross-fit (false: fit once on everything)
     * @param seed the fold seed
     * @return the estimate of {@code theta}
     */
    public static Estimate partiallyLinear(double[][] x, double[] t, double[] y, Regressor outcomeModel,
            Regressor treatmentModel, int folds, boolean crossFit, long seed) {
        int n = x.length;
        int[] fold = CrossFitting.folds(n, folds, seed);
        double[] l = nuisance(x, y, outcomeModel, fold, crossFit);
        double[] m = nuisance(x, t, treatmentModel, fold, crossFit);
        double sxy = 0;
        double sxx = 0;
        double[] ry = new double[n];
        double[] rt = new double[n];
        for (int i = 0; i < n; i++) {
            ry[i] = y[i] - l[i];
            rt[i] = t[i] - m[i];
            sxy += rt[i] * ry[i];
            sxx += rt[i] * rt[i];
        }
        double theta = sxy / sxx;
        double j = sxx / n;
        double psi2 = 0;
        for (int i = 0; i < n; i++) {
            double psi = (ry[i] - theta * rt[i]) * rt[i];
            psi2 += psi * psi / n;
        }
        return new Estimate(theta, Math.sqrt(psi2 / (j * j) / n));
    }

    /**
     * The interactive model with a binary treatment: the mean of the AIPW score.
     *
     * @param x covariates
     * @param t treatments
     * @param y outcomes
     * @param outcomeModel the per-arm outcome model
     * @param propensityModel the regression of {@code t} on {@code x}
     * @param folds cross-fitting folds
     * @param crossFit whether to cross-fit
     * @param clip propensity clipping
     * @param seed the fold seed
     * @return the ATE estimate
     */
    public static Estimate interactive(double[][] x, int[] t, double[] y, Regressor outcomeModel,
            Regressor propensityModel, int folds, boolean crossFit, double clip, long seed) {
        int n = x.length;
        double[] mu0;
        double[] mu1;
        double[] e;
        if (crossFit) {
            int[] fold = CrossFitting.folds(n, folds, seed);
            mu0 = CrossFitting.predictArm(x, t, y, 0, outcomeModel, fold);
            mu1 = CrossFitting.predictArm(x, t, y, 1, outcomeModel, fold);
            e = CrossFitting.predict(x, CrossFitting.asDouble(t), null, propensityModel, fold);
        } else {
            mu0 = armFit(x, t, y, 0, outcomeModel);
            mu1 = armFit(x, t, y, 1, outcomeModel);
            e = propensityModel.fit(x, CrossFitting.asDouble(t), null).predict(x);
        }
        double[] phi = new double[n];
        double mean = 0;
        for (int i = 0; i < n; i++) {
            double p = Math.clamp(e[i], clip, 1 - clip);
            phi[i] = mu1[i] - mu0[i] + (t[i] == 1 ? (y[i] - mu1[i]) / p : -(y[i] - mu0[i]) / (1 - p));
            mean += phi[i] / n;
        }
        double var = 0;
        for (double v : phi) {
            var += (v - mean) * (v - mean) / (n - 1);
        }
        return new Estimate(mean, Math.sqrt(var / n));
    }

    private static double[] armFit(double[][] x, int[] t, double[] y, int arm, Regressor model) {
        int c = 0;
        for (int v : t) {
            c += v == arm ? 1 : 0;
        }
        double[][] ax = new double[c][];
        double[] ay = new double[c];
        int k = 0;
        for (int i = 0; i < x.length; i++) {
            if (t[i] == arm) {
                ax[k] = x[i];
                ay[k++] = y[i];
            }
        }
        return model.fit(ax, ay, null).predict(x);
    }
}
