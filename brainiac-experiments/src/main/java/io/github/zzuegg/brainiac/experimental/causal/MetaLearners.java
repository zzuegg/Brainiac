package io.github.zzuegg.brainiac.experimental.causal;

/**
 * Meta-learners: recipes that turn any {@link Regressor} into an estimator of the conditional
 * average treatment effect (CATE) {@code tau(x) = E[y1 - y0 | x]} from observational data with a
 * binary treatment.
 * <ul>
 *   <li>{@link Kind#S}: one model of {@code y} on {@code (x, t)}; {@code tau = mu(x, 1) - mu(x, 0)}.
 *       Regularisation tends to shrink the effect toward zero.</li>
 *   <li>{@link Kind#T}: one model per arm; {@code tau = mu1 - mu0}. Each model's error lands in the
 *       effect, which hurts when one arm is small.</li>
 *   <li>{@link Kind#X} (Künzel et al., PNAS 2019, <a href="https://arxiv.org/abs/1706.03461">
 *       arXiv:1706.03461</a>): impute each unit's individual effect from the other arm's model
 *       ({@code y - mu0} for the treated, {@code mu1 - y} for the controls), regress the imputed
 *       effects on {@code x} per arm, and blend the two by the propensity,
 *       {@code tau = e tau0 + (1 - e) tau1}; strong when one arm is much smaller.</li>
 *   <li>{@link Kind#R} (Nie and Wager, Biometrika 2021, <a href="https://arxiv.org/abs/1712.04912">
 *       arXiv:1712.04912</a>): minimise the Robinson residual loss
 *       {@code sum((y - m(x)) - (t - e(x)) tau(x))^2} with cross-fitted {@code m = E[y | x]} and
 *       {@code e}; done as a weighted regression of {@code (y - m) / (t - e)} with weights
 *       {@code (t - e)^2}.</li>
 *   <li>{@link Kind#DR} (Kennedy 2023, <a href="https://arxiv.org/abs/2004.14497">
 *       arXiv:2004.14497</a>): regress the AIPW pseudo-outcome
 *       {@code mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)} on {@code x}, with
 *       cross-fitted nuisances.</li>
 * </ul>
 * The nuisance models and the final effect model are separate {@link Regressor}s; the propensity is
 * a regression of the 0/1 treatment, clipped to {@code [clip, 1 - clip]}.
 */
public final class MetaLearners {

    /** The meta-learner. */
    public enum Kind {
        /** Single model with the treatment as a feature. */
        S,
        /** Two models, one per arm. */
        T,
        /** Cross-imputed effects blended by the propensity. */
        X,
        /** Robinson residual-on-residual loss. */
        R,
        /** Doubly robust pseudo-outcome regression. */
        DR
    }

    private final Regressor outcome;
    private final Regressor effect;
    private final Regressor propensity;
    private int folds = 5;
    private double clip = 0.02;
    private long seed = 1;

    /**
     * Creates the learners.
     *
     * @param outcome the outcome (and conditional-mean) model
     * @param effect the second-stage effect model of X, R and DR
     * @param propensity the regression of the treatment on {@code x}
     */
    public MetaLearners(Regressor outcome, Regressor effect, Regressor propensity) {
        this.outcome = outcome;
        this.effect = effect;
        this.propensity = propensity;
    }

    /**
     * @param k cross-fitting folds (R and DR)
     * @return this
     */
    public MetaLearners folds(int k) {
        this.folds = k;
        return this;
    }

    /**
     * @param c the propensity clipping
     * @return this
     */
    public MetaLearners clip(double c) {
        this.clip = c;
        return this;
    }

    /**
     * @param s the fold seed
     * @return this
     */
    public MetaLearners seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Fits a CATE function.
     *
     * @param kind the meta-learner
     * @param x covariates {@code [n][d]}
     * @param t treatments
     * @param y outcomes
     * @return {@code tau(x)}
     */
    public Regressor.Fitted fit(Kind kind, double[][] x, int[] t, double[] y) {
        return switch (kind) {
            case S -> sLearner(x, t, y);
            case T -> tLearner(x, t, y);
            case X -> xLearner(x, t, y);
            case R -> rLearner(x, t, y);
            case DR -> drLearner(x, t, y);
        };
    }

    private Regressor.Fitted sLearner(double[][] x, int[] t, double[] y) {
        int n = x.length;
        int d = x[0].length;
        double[][] xt = new double[n][];
        for (int i = 0; i < n; i++) {
            xt[i] = java.util.Arrays.copyOf(x[i], d + 1);
            xt[i][d] = t[i];
        }
        Regressor.Fitted mu = outcome.fit(xt, y, null);
        return row -> {
            double[] a = java.util.Arrays.copyOf(row, d + 1);
            double[] b = java.util.Arrays.copyOf(row, d + 1);
            a[d] = 1;
            return mu.predict(a) - mu.predict(b);
        };
    }

    private Regressor.Fitted[] arms(double[][] x, int[] t, double[] y) {
        Regressor.Fitted[] mu = new Regressor.Fitted[2];
        for (int arm = 0; arm < 2; arm++) {
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
            mu[arm] = outcome.fit(ax, ay, null);
        }
        return mu;
    }

    private Regressor.Fitted tLearner(double[][] x, int[] t, double[] y) {
        Regressor.Fitted[] mu = arms(x, t, y);
        return row -> mu[1].predict(row) - mu[0].predict(row);
    }

    private Regressor.Fitted xLearner(double[][] x, int[] t, double[] y) {
        Regressor.Fitted[] mu = arms(x, t, y);
        Regressor.Fitted[] tau = new Regressor.Fitted[2];
        for (int arm = 0; arm < 2; arm++) {
            int c = 0;
            for (int v : t) {
                c += v == arm ? 1 : 0;
            }
            double[][] ax = new double[c][];
            double[] imputed = new double[c];
            int k = 0;
            for (int i = 0; i < x.length; i++) {
                if (t[i] == arm) {
                    ax[k] = x[i];
                    imputed[k++] = arm == 1 ? y[i] - mu[0].predict(x[i]) : mu[1].predict(x[i]) - y[i];
                }
            }
            tau[arm] = effect.fit(ax, imputed, null);
        }
        Regressor.Fitted e = propensity.fit(x, CrossFitting.asDouble(t), null);
        return row -> {
            double p = Math.clamp(e.predict(row), clip, 1 - clip);
            return p * tau[0].predict(row) + (1 - p) * tau[1].predict(row);
        };
    }

    private Regressor.Fitted rLearner(double[][] x, int[] t, double[] y) {
        int[] fold = CrossFitting.folds(x.length, folds, seed);
        double[] m = CrossFitting.predict(x, y, null, outcome, fold);
        double[] e = CrossFitting.predict(x, CrossFitting.asDouble(t), null, propensity, fold);
        double[] pseudo = new double[x.length];
        double[] w = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double res = t[i] - Math.clamp(e[i], clip, 1 - clip);
            pseudo[i] = (y[i] - m[i]) / res;
            w[i] = res * res;
        }
        return effect.fit(x, pseudo, w);
    }

    private Regressor.Fitted drLearner(double[][] x, int[] t, double[] y) {
        double[] phi = aipwScores(x, t, y);
        return effect.fit(x, phi, null);
    }

    /**
     * Cross-fitted AIPW (doubly robust) scores
     * {@code mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)}, whose mean is the AIPW
     * ATE and whose regression on {@code x} is the DR-learner.
     *
     * @param x covariates
     * @param t treatments
     * @param y outcomes
     * @return one score per unit
     */
    public double[] aipwScores(double[][] x, int[] t, double[] y) {
        int[] fold = CrossFitting.folds(x.length, folds, seed);
        double[] mu0 = CrossFitting.predictArm(x, t, y, 0, outcome, fold);
        double[] mu1 = CrossFitting.predictArm(x, t, y, 1, outcome, fold);
        double[] e = CrossFitting.predict(x, CrossFitting.asDouble(t), null, propensity, fold);
        double[] phi = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double p = Math.clamp(e[i], clip, 1 - clip);
            phi[i] = mu1[i] - mu0[i] + (t[i] == 1 ? (y[i] - mu1[i]) / p : -(y[i] - mu0[i]) / (1 - p));
        }
        return phi;
    }

    /**
     * The precision in estimating heterogeneous effects,
     * {@code PEHE = sqrt(mean((tauHat - tau)^2))}.
     *
     * @param estimate estimated effects
     * @param truth true effects
     * @return the PEHE
     */
    public static double pehe(double[] estimate, double[] truth) {
        double s = 0;
        for (int i = 0; i < truth.length; i++) {
            s += (estimate[i] - truth[i]) * (estimate[i] - truth[i]);
        }
        return Math.sqrt(s / truth.length);
    }
}
