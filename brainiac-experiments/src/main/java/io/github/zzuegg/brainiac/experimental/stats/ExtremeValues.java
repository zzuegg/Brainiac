package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.tensor.Optimize;

import java.util.Arrays;

/**
 * Extreme value statistics: extrapolating beyond the range of the data.
 * <ul>
 *   <li>Peaks over threshold: by the Pickands-Balkema-de Haan theorem, exceedances over a high
 *       threshold are approximately generalised Pareto, {@code GPD(xi, sigma)}; fitted by
 *       maximum likelihood ({@link #fitGpd}, core {@link Optimize#nelderMead} from the PWM start)
 *       or probability-weighted moments ({@link #fitGpdPwm}, Hosking and Wallis 1987); the
 *       threshold is chosen where the shape estimate stabilises ({@link #stableThreshold}, the
 *       numeric form of a parameter-stability plot); tail quantiles come from
 *       {@link Pot#quantile} with profile-likelihood intervals ({@link Pot#quantileInterval});</li>
 *   <li>the Hill estimator of the tail index {@code gamma = 1/alpha} of heavy tails with an
 *       automatic choice of {@code k} ({@link #hill}, {@link #hillK}) and Weissman's quantile
 *       extrapolation ({@link #hillQuantile});</li>
 *   <li>generalised extreme value fits to block maxima and their return levels
 *       ({@link #fitGev}, {@link Gev#returnLevel}).</li>
 * </ul>
 * References: Pickands (Annals 1975), Hill (Annals 1975), Coles, "An Introduction to
 * Statistical Modeling of Extreme Values" (2001).
 */
public final class ExtremeValues {

    private ExtremeValues() {
    }

    /**
     * A generalised Pareto distribution for exceedances {@code y = x - u > 0}.
     *
     * @param shape {@code xi}
     * @param scale {@code sigma}
     */
    public record Gpd(double shape, double scale) {

        /**
         * The log density of an exceedance.
         *
         * @param y the exceedance
         * @return {@code log g(y)}, negative infinity outside the support
         */
        public double logDensity(double y) {
            if (y < 0 || scale <= 0) {
                return Double.NEGATIVE_INFINITY;
            }
            if (Math.abs(shape) < 1e-9) {
                return -Math.log(scale) - y / scale;
            }
            double z = 1 + shape * y / scale;
            return z <= 0 ? Double.NEGATIVE_INFINITY : -Math.log(scale) - (1 / shape + 1) * Math.log(z);
        }

        /**
         * The survival function.
         *
         * @param y the exceedance
         * @return {@code P(Y > y)}
         */
        public double survival(double y) {
            if (y <= 0) {
                return 1;
            }
            if (Math.abs(shape) < 1e-9) {
                return Math.exp(-y / scale);
            }
            double z = 1 + shape * y / scale;
            return z <= 0 ? 0 : Math.pow(z, -1 / shape);
        }

        /**
         * The quantile of the exceedance distribution.
         *
         * @param p the probability
         * @return {@code G^-1(p)}
         */
        public double quantile(double p) {
            if (Math.abs(shape) < 1e-9) {
                return -scale * Math.log1p(-p);
            }
            return scale / shape * (Math.pow(1 - p, -shape) - 1);
        }

        /**
         * The log likelihood of exceedances.
         *
         * @param y the exceedances
         * @return the log likelihood
         */
        public double logLikelihood(double[] y) {
            double s = 0;
            for (double v : y) {
                s += logDensity(v);
            }
            return s;
        }
    }

    /**
     * Probability-weighted-moment estimate of a GPD.
     *
     * @param y exceedances, positive
     * @return the fit
     */
    public static Gpd fitGpdPwm(double[] y) {
        double[] s = y.clone();
        Arrays.sort(s);
        int n = s.length;
        double a0 = 0;
        double a1 = 0;
        for (int i = 0; i < n; i++) {
            a0 += s[i] / n;
            a1 += s[i] * (n - 1 - i) / (double) (n - 1) / n;
        }
        double k = a0 / (a0 - 2 * a1) - 2;
        double sigma = 2 * a0 * a1 / (a0 - 2 * a1);
        return new Gpd(-k, Math.max(sigma, 1e-12));
    }

    /**
     * Maximum-likelihood GPD by Nelder-Mead on {@code (xi, log sigma)}, started at PWM.
     *
     * @param y exceedances, positive
     * @return the fit
     */
    public static Gpd fitGpd(double[] y) {
        Gpd start = fitGpdPwm(y);
        double xi0 = Math.max(-0.45, Math.min(1.5, start.shape()));
        double[] x0 = {xi0, Math.log(start.scale())};
        Optimize.Minimum m = Optimize.nelderMead(v -> {
            double ll = new Gpd(v[0], Math.exp(v[1])).logLikelihood(y);
            return Double.isFinite(ll) ? -ll : 1e300;
        }, x0, 0.1, 2000, 1e-10);
        return new Gpd(m.x()[0], Math.exp(m.x()[1]));
    }

    /**
     * A peaks-over-threshold tail model.
     *
     * @param threshold {@code u}
     * @param gpd the exceedance distribution
     * @param exceedances the exceedances used
     * @param n the total sample size
     */
    public record Pot(double threshold, Gpd gpd, double[] exceedances, int n) {

        /**
         * The rate of exceeding the threshold.
         *
         * @return {@code N_u / n}
         */
        public double rate() {
            return exceedances.length / (double) n;
        }

        /**
         * A quantile of the original distribution beyond the threshold.
         *
         * @param p the probability, above {@code 1 - rate()}
         * @return {@code x_p}
         */
        public double quantile(double p) {
            return threshold + gpd.quantile(1 - (1 - p) / rate());
        }

        /**
         * The profile-likelihood interval for {@link #quantile}: the GPD scale is written in
         * terms of the quantile and the shape, the shape is maximised out on a grid, and the
         * interval holds the quantiles whose deviance is below {@code chi2_1(level)}.
         *
         * @param p the probability
         * @param critical the chi-squared critical value, 3.84 for 95%
         * @return {@code {low, high}}
         */
        public double[] quantileInterval(double p, double critical) {
            double zeta = (1 - p) / rate();
            double best = gpd.logLikelihood(exceedances);
            double q0 = quantile(p) - threshold;
            double lo = q0;
            double hi = q0;
            double[] factors = new double[120];
            for (int i = 0; i < factors.length; i++) {
                factors[i] = Math.exp(-2.5 + 5.0 * i / (factors.length - 1));
            }
            for (double f : factors) {
                double q = q0 * f;
                double prof = Double.NEGATIVE_INFINITY;
                for (int s = 0; s <= 120; s++) {
                    double xi = -0.5 + 1.7 * s / 120;
                    double denom = Math.abs(xi) < 1e-9 ? -Math.log(zeta) : (Math.pow(zeta, -xi) - 1) / xi;
                    double sigma = q / denom;
                    if (sigma > 0) {
                        prof = Math.max(prof, new Gpd(xi, sigma).logLikelihood(exceedances));
                    }
                }
                if (2 * (best - prof) <= critical) {
                    lo = Math.min(lo, q);
                    hi = Math.max(hi, q);
                }
            }
            return new double[] {threshold + lo, threshold + hi};
        }
    }

    /**
     * Fits a POT model at a given threshold quantile.
     *
     * @param data the sample
     * @param thresholdQuantile for example 0.95
     * @param mle maximum likelihood, else PWM
     * @return the tail model
     */
    public static Pot pot(double[] data, double thresholdQuantile, boolean mle) {
        double[] s = data.clone();
        Arrays.sort(s);
        double u = s[(int) Math.floor(thresholdQuantile * (s.length - 1))];
        double[] y = Arrays.stream(s).filter(v -> v > u).map(v -> v - u).toArray();
        return new Pot(u, mle ? fitGpd(y) : fitGpdPwm(y), y, s.length);
    }

    /**
     * Chooses a threshold quantile where the MLE shape stops drifting: the lowest of the
     * candidate quantiles 0.80, 0.82, ..., 0.98 whose shape estimate stays within
     * {@code tolerance} of the estimates at all higher candidates.
     *
     * @param data the sample
     * @param tolerance for example 0.15
     * @return the threshold quantile
     */
    public static double stableThreshold(double[] data, double tolerance) {
        int c = 10;
        double[] q = new double[c];
        double[] xi = new double[c];
        for (int i = 0; i < c; i++) {
            q[i] = 0.80 + 0.02 * i;
            xi[i] = pot(data, q[i], true).gpd().shape();
        }
        for (int i = 0; i < c; i++) {
            boolean stable = true;
            for (int j = i + 1; j < c; j++) {
                stable &= Math.abs(xi[j] - xi[i]) <= tolerance;
            }
            if (stable) {
                return q[i];
            }
        }
        return q[c - 1];
    }

    /**
     * The Hill estimator {@code gamma = mean(log X_(n-i+1)) - log X_(n-k)} over the top
     * {@code k} order statistics.
     *
     * @param data the sample (the positive part is used)
     * @param k the number of top order statistics
     * @return {@code gamma = 1 / alpha}
     */
    public static double hill(double[] data, int k) {
        double[] s = data.clone();
        Arrays.sort(s);
        int n = s.length;
        double ref = Math.log(Math.max(s[n - k - 1], 1e-300));
        double g = 0;
        for (int i = 0; i < k; i++) {
            g += Math.log(Math.max(s[n - 1 - i], 1e-300)) - ref;
        }
        return g / k;
    }

    /**
     * Chooses {@code k} for the Hill estimator where the Hill plot is flattest: the centre of
     * the window of 20% of the candidates with the smallest standard deviation, over
     * {@code k} from {@code sqrt(n)} to {@code n / 5}.
     *
     * @param data the sample
     * @return {@code k}
     */
    public static int hillK(double[] data) {
        int n = data.length;
        int lo = Math.max(10, (int) Math.sqrt(n));
        int hi = Math.max(lo + 5, n / 5);
        double[] g = new double[hi - lo + 1];
        for (int k = lo; k <= hi; k++) {
            g[k - lo] = hill(data, k);
        }
        int w = Math.max(3, g.length / 5);
        double best = Double.POSITIVE_INFINITY;
        int arg = lo;
        for (int s = 0; s + w <= g.length; s++) {
            double m = 0;
            for (int i = s; i < s + w; i++) {
                m += g[i] / w;
            }
            double v = 0;
            for (int i = s; i < s + w; i++) {
                v += (g[i] - m) * (g[i] - m);
            }
            if (v < best) {
                best = v;
                arg = lo + s + w / 2;
            }
        }
        return arg;
    }

    /**
     * Weissman's quantile extrapolation {@code X_(n-k) (k / (n (1 - p)))^gamma}.
     *
     * @param data the sample
     * @param k the number of top order statistics
     * @param p the probability
     * @return {@code x_p}
     */
    public static double hillQuantile(double[] data, int k, double p) {
        double[] s = data.clone();
        Arrays.sort(s);
        int n = s.length;
        return s[n - k - 1] * Math.pow(k / (n * (1 - p)), hill(data, k));
    }

    /**
     * A generalised extreme value distribution.
     *
     * @param location {@code mu}
     * @param scale {@code sigma}
     * @param shape {@code xi}
     */
    public record Gev(double location, double scale, double shape) {

        /**
         * The log density.
         *
         * @param x the value
         * @return {@code log f(x)}
         */
        public double logDensity(double x) {
            double z = (x - location) / scale;
            if (Math.abs(shape) < 1e-9) {
                return -Math.log(scale) - z - Math.exp(-z);
            }
            double t = 1 + shape * z;
            if (t <= 0) {
                return Double.NEGATIVE_INFINITY;
            }
            return -Math.log(scale) - (1 / shape + 1) * Math.log(t) - Math.pow(t, -1 / shape);
        }

        /**
         * The return level exceeded once every {@code period} blocks on average.
         *
         * @param period the return period in blocks
         * @return the level
         */
        public double returnLevel(double period) {
            double y = -Math.log(1 - 1 / period);
            if (Math.abs(shape) < 1e-9) {
                return location - scale * Math.log(y);
            }
            return location + scale / shape * (Math.pow(y, -shape) - 1);
        }
    }

    /**
     * Maximum-likelihood GEV fit to block maxima, started from Gumbel moments.
     *
     * @param maxima the block maxima
     * @return the fit
     */
    public static Gev fitGev(double[] maxima) {
        double m = Arrays.stream(maxima).average().orElse(0);
        double v = Arrays.stream(maxima).map(x -> (x - m) * (x - m)).average().orElse(1);
        double sigma0 = Math.sqrt(6 * v) / Math.PI;
        double[] x0 = {m - 0.5772 * sigma0, Math.log(sigma0), 0.1};
        Optimize.Minimum min = Optimize.nelderMead(p -> {
            Gev g = new Gev(p[0], Math.exp(p[1]), p[2]);
            double ll = 0;
            for (double x : maxima) {
                ll += g.logDensity(x);
            }
            return Double.isFinite(ll) ? -ll : 1e300;
        }, x0, 0.1, 4000, 1e-10);
        return new Gev(min.x()[0], Math.exp(min.x()[1]), min.x()[2]);
    }
}
