package io.github.zzuegg.brainiac.experimental.boost;

import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * NGBoost (Duan et al., "NGBoost: Natural Gradient Boosting for Probabilistic Prediction",
 * ICML 2020, <a href="https://arxiv.org/abs/1910.03225">arXiv:1910.03225</a>): boosting that
 * predicts a whole conditional distribution. Each row carries the distribution's parameters
 * {@code theta = (location, log scale)}; every stage computes the gradient of a proper scoring
 * rule with respect to {@code theta}, premultiplies it by the inverse of the rule's Riemannian
 * metric (the Fisher information for the log score, its CRPS analogue otherwise) to get the
 * <em>natural</em> gradient, fits one regression tree per parameter to it, picks a step by line
 * search on the training score and moves every row's parameters. The natural gradient makes
 * the steps invariant to the parameterisation, so location and scale learn at comparable
 * speeds.
 *
 * <p>Families: Normal and Laplace with the log score or CRPS, LogNormal with the log score. All
 * metrics are closed-form: for the Normal with {@code (mu, log sigma)} the Fisher information
 * is {@code diag(1 / sigma^2, 2)} and the CRPS metric {@code diag(1 / (sqrt(pi) sigma),
 * sigma / (2 sqrt(pi)))}; for the Laplace {@code diag(1 / b^2, 1)} and {@code diag(1 / (2b), b / 4)}.
 * Base learners are CART trees grown by {@link RegressionForest#grow}. Setting
 * {@code natural = false} uses the ordinary gradient (the paper's ablation).
 */
public final class NaturalGradientBoosting {

    /** The predicted distribution family. */
    public enum Family {
        /** Normal on the target. */
        NORMAL,
        /** Normal on the log of a positive target. */
        LOG_NORMAL,
        /** Laplace on the target. */
        LAPLACE
    }

    /** The proper scoring rule boosted on. */
    public enum Score {
        /** Negative log-likelihood. */
        LOG,
        /** Continuous ranked probability score. */
        CRPS
    }

    /**
     * Settings.
     *
     * @param stages boosting stages
     * @param learningRate the shrinkage of each stage
     * @param depth the base trees' depth
     * @param minLeaf the base trees' smallest leaf
     * @param natural natural (true) or ordinary (false) gradient
     * @param subsample the share of rows each stage's trees are fitted on (the paper's minibatch)
     * @param seed seeds the base learners and the row sampling
     */
    public record Settings(int stages, double learningRate, int depth, int minLeaf, boolean natural,
            double subsample, long seed) {

        /** @return 300 stages, learning rate 0.05, depth-3 trees with leaves of 20, half the rows per stage */
        public static Settings defaults() {
            return new Settings(300, 0.05, 3, 20, true, 0.5, 1);
        }

        /**
         * @param on natural or ordinary gradient
         * @return a copy with that choice
         */
        public Settings withNatural(boolean on) {
            return new Settings(stages, learningRate, depth, minLeaf, on, subsample, seed);
        }

        /**
         * @param n stages
         * @return a copy with that many stages
         */
        public Settings withStages(int n) {
            return new Settings(n, learningRate, depth, minLeaf, natural, subsample, seed);
        }
    }

    /**
     * A predicted distribution.
     *
     * @param family its family
     * @param loc the location ({@code mu}, on the log scale for LogNormal)
     * @param logScale the log of the scale ({@code sigma} or {@code b})
     */
    public record Distribution(Family family, double loc, double logScale) {

        /** @return the scale */
        public double scale() {
            return Math.exp(logScale);
        }

        /** @return the mean */
        public double mean() {
            return family == Family.LOG_NORMAL ? Math.exp(loc + scale() * scale() / 2) : loc;
        }

        /**
         * @param p a probability in (0, 1)
         * @return the {@code p}-quantile
         */
        public double quantile(double p) {
            double s = scale();
            return switch (family) {
                case NORMAL -> loc + s * Gaussian.inverseCdf(p);
                case LOG_NORMAL -> Math.exp(loc + s * Gaussian.inverseCdf(p));
                case LAPLACE -> loc - s * Math.signum(p - 0.5) * Math.log(1 - 2 * Math.abs(p - 0.5));
            };
        }

        /**
         * @param y a value
         * @return the negative log density at {@code y}
         */
        public double nll(double y) {
            double s = scale();
            return switch (family) {
                case NORMAL -> {
                    double z = (y - loc) / s;
                    yield logScale + 0.5 * z * z + 0.5 * Math.log(2 * Math.PI);
                }
                case LOG_NORMAL -> {
                    double ly = Math.log(Math.max(y, 1e-300));
                    double z = (ly - loc) / s;
                    yield logScale + 0.5 * z * z + 0.5 * Math.log(2 * Math.PI) + ly;
                }
                case LAPLACE -> Math.log(2 * s) + Math.abs(y - loc) / s;
            };
        }

        /**
         * @param y a value
         * @return the continuous ranked probability score at {@code y}
         */
        public double crps(double y) {
            double s = scale();
            return switch (family) {
                case NORMAL -> {
                    double z = (y - loc) / s;
                    yield s * (z * (2 * Gaussian.cdf(z) - 1) + 2 * Gaussian.pdf(z) - 1 / Math.sqrt(Math.PI));
                }
                case LOG_NORMAL -> {
                    double z = (Math.log(Math.max(y, 1e-300)) - loc) / s;
                    yield y * (2 * Gaussian.cdf(z) - 1) - 2 * Math.exp(loc + s * s / 2)
                            * (Gaussian.cdf(z - s) + Gaussian.cdf(s / Math.sqrt(2)) - 1);
                }
                case LAPLACE -> {
                    double a = Math.abs(y - loc);
                    yield a + s * Math.exp(-a / s) - 0.75 * s;
                }
            };
        }
    }

    private final Family family;
    private final double[] init;
    private final RegressionForest.Tree[][] trees;   // [stage][parameter]
    private final double[] steps;

    private NaturalGradientBoosting(Family family, double[] init, RegressionForest.Tree[][] trees, double[] steps) {
        this.family = family;
        this.init = init;
        this.trees = trees;
        this.steps = steps;
    }

    /**
     * Trains.
     *
     * @param x rows {@code [n][d]}
     * @param y targets (positive for LogNormal)
     * @param family the distribution family
     * @param score the scoring rule
     * @param s the settings
     * @return the model
     */
    public static NaturalGradientBoosting fit(double[][] x, double[] y, Family family, Score score, Settings s) {
        if (family == Family.LOG_NORMAL && score == Score.CRPS) {
            throw new IllegalArgumentException("LogNormal is boosted on the log score only");
        }
        int n = y.length;
        double[] target = family == Family.LOG_NORMAL ? Arrays.stream(y).map(v -> Math.log(Math.max(v, 1e-300))).toArray()
                : y.clone();
        double[] init = marginal(target, family);
        double[][] theta = new double[n][];
        for (int i = 0; i < n; i++) {
            theta[i] = init.clone();
        }
        int[] all = new int[n];
        Arrays.setAll(all, i -> i);
        SplittableRandom random = new SplittableRandom(s.seed());
        RegressionForest.Tree[][] trees = new RegressionForest.Tree[s.stages()][2];
        double[] steps = new double[s.stages()];
        double[] grad = new double[2];
        double[][] direction = new double[2][n];
        for (int t = 0; t < s.stages(); t++) {
            for (int i = 0; i < n; i++) {
                gradient(family, score, theta[i], target[i], grad);
                if (s.natural()) {
                    double[] metric = metric(family, score, theta[i]);
                    direction[0][i] = grad[0] / metric[0];
                    direction[1][i] = grad[1] / metric[1];
                } else {
                    direction[0][i] = grad[0];
                    direction[1][i] = grad[1];
                }
            }
            double[][] fitted = new double[2][n];
            int[] sample = s.subsample() >= 1 ? all : RegressionForest.subsample(n, s.subsample(), false, random)[0];
            for (int p = 0; p < 2; p++) {
                double[] target2 = direction[p];
                trees[t][p] = RegressionForest.grow(x, sample, s.depth(), s.minLeaf(), x[0].length, mse(target2), random,
                        rows -> mean(rows, target2));
                for (int i = 0; i < n; i++) {
                    fitted[p][i] = trees[t][p].predict(x[i]);
                }
            }
            double before = total(family, score, theta, target, fitted, 0);
            double rho = 1;
            while (rho > 1.0 / 1024 && total(family, score, theta, target, fitted, rho) > before) {
                rho /= 2;
            }
            // try growing the step as the paper's line search does
            while (rho < 64 && total(family, score, theta, target, fitted, 2 * rho)
                    < total(family, score, theta, target, fitted, rho)) {
                rho *= 2;
            }
            steps[t] = rho * s.learningRate();
            for (int i = 0; i < n; i++) {
                theta[i][0] -= steps[t] * fitted[0][i];
                theta[i][1] = Math.clamp(theta[i][1] - steps[t] * fitted[1][i], -12, 12);
            }
        }
        return new NaturalGradientBoosting(family, init, trees, steps);
    }

    private static double total(Family family, Score score, double[][] theta, double[] y, double[][] fitted,
            double rho) {
        double sum = 0;
        for (int i = 0; i < y.length; i++) {
            double loc = theta[i][0] - rho * fitted[0][i];
            double ls = Math.clamp(theta[i][1] - rho * fitted[1][i], -12, 12);
            Family base = family == Family.LOG_NORMAL ? Family.NORMAL : family;
            Distribution d = new Distribution(base, loc, ls);
            sum += score == Score.LOG ? d.nll(y[i]) : d.crps(y[i]);
        }
        return sum;
    }

    private static double[] marginal(double[] y, Family family) {
        if (family == Family.LAPLACE) {
            double[] sorted = y.clone();
            Arrays.sort(sorted);
            double med = sorted[sorted.length / 2];
            double mad = Arrays.stream(y).map(v -> Math.abs(v - med)).average().orElse(1);
            return new double[] {med, Math.log(Math.max(mad, 1e-6))};
        }
        double mean = Arrays.stream(y).average().orElse(0);
        double var = Arrays.stream(y).map(v -> (v - mean) * (v - mean)).average().orElse(1);
        return new double[] {mean, 0.5 * Math.log(Math.max(var, 1e-12))};
    }

    /** Gradient of the score with respect to {@code (loc, log scale)}; LogNormal works on log y. */
    private static void gradient(Family family, Score score, double[] theta, double y, double[] out) {
        double s = Math.exp(theta[1]);
        double z = (y - theta[0]) / s;
        if (family == Family.LAPLACE) {
            double a = Math.abs(z);
            if (score == Score.LOG) {
                out[0] = -Math.signum(z) / s;
                out[1] = 1 - a;
            } else {
                out[0] = -Math.signum(z) * (1 - Math.exp(-a));
                out[1] = s * (Math.exp(-a) * (1 + a) - 0.75);
            }
            return;
        }
        if (score == Score.LOG) {
            out[0] = -z / s;
            out[1] = 1 - z * z;
        } else {
            out[0] = -(2 * Gaussian.cdf(z) - 1);
            out[1] = s * (2 * Gaussian.pdf(z) - 1 / Math.sqrt(Math.PI));
        }
    }

    /** The diagonal Riemannian metric of the score at {@code theta}. */
    private static double[] metric(Family family, Score score, double[] theta) {
        double s = Math.exp(theta[1]);
        if (family == Family.LAPLACE) {
            return score == Score.LOG ? new double[] {1 / (s * s), 1} : new double[] {1 / (2 * s), s / 4};
        }
        return score == Score.LOG ? new double[] {1 / (s * s), 2}
                : new double[] {1 / (Math.sqrt(Math.PI) * s), s / (2 * Math.sqrt(Math.PI))};
    }

    private static double mean(int[] rows, double[] v) {
        double s = 0;
        for (int r : rows) {
            s += v[r];
        }
        return rows.length == 0 ? 0 : s / rows.length;
    }

    /** Squared-error split gain with prefix sums, the base learners' criterion. */
    static RegressionForest.SplitScore mse(double[] y) {
        return new RegressionForest.SplitScore() {
            private double[] prefix = new double[0];
            private int[] cached;
            private double total;

            @Override
            public void node(int[] rows) {
                total = 0;
                for (int r : rows) {
                    total += y[r];
                }
                cached = null;
            }

            @Override
            public double gain(int[] rows, int cut) {
                if (rows != cached) {
                    prefix = new double[rows.length + 1];
                    for (int i = 0; i < rows.length; i++) {
                        prefix[i + 1] = prefix[i] + y[rows[i]];
                    }
                    cached = rows;
                }
                double sl = prefix[cut];
                double sr = total - sl;
                int nr = rows.length - cut;
                return sl * sl / cut + sr * sr / nr - total * total / rows.length;
            }
        };
    }

    /**
     * @param x a row
     * @return the predicted distribution
     */
    public Distribution predict(double[] x) {
        double loc = init[0];
        double ls = init[1];
        for (int t = 0; t < trees.length; t++) {
            loc -= steps[t] * trees[t][0].predict(x);
            ls = Math.clamp(ls - steps[t] * trees[t][1].predict(x), -12, 12);
        }
        return new Distribution(family, loc, ls);
    }

    /** @return the number of stages */
    public int size() {
        return trees.length;
    }

    /** Standard normal helpers. */
    static final class Gaussian {
        private Gaussian() {
        }

        static double pdf(double z) {
            return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
        }

        static double cdf(double z) {
            return 0.5 * erfc(-z / Math.sqrt(2));
        }

        /** Complementary error function (Numerical Recipes' Chebyshev fit, relative error below 1.2e-7). */
        static double erfc(double x) {
            double z = Math.abs(x);
            double t = 1 / (1 + 0.5 * z);
            double r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                    + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
                    + t * (-0.82215223 + t * 0.17087277)))))))));
            return x >= 0 ? r : 2 - r;
        }

        /** Acklam's rational approximation of the inverse normal CDF. */
        static double inverseCdf(double p) {
            double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
            double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01};
            double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
            double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                3.754408661907416e+00};
            double q;
            if (p < 0.02425) {
                q = Math.sqrt(-2 * Math.log(p));
                return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
            }
            if (p > 1 - 0.02425) {
                q = Math.sqrt(-2 * Math.log(1 - p));
                return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
            }
            q = p - 0.5;
            double r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }
    }
}
