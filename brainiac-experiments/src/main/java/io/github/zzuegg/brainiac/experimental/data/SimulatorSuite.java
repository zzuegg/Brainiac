package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.sbi.SliceSampler;
import io.github.zzuegg.brainiac.tensor.Special;

import java.util.SplittableRandom;

/**
 * sbibm-style toy simulators for simulation-based inference (Lueckmann et al., "Benchmarking
 * simulation-based inference", AISTATS 2021, <a href="https://arxiv.org/abs/2101.04653">
 * arXiv:2101.04653</a>) that complement the Gaussian-linear, two-moons and Lotka-Volterra tasks
 * of {@code experiments.probabilistic.SimulatorInference}:
 * <ul>
 *   <li>{@link #slcp}: "simple likelihood, complex posterior", a 5-parameter Gaussian whose
 *       mean and covariance depend nonlinearly on the parameters, observed through four i.i.d.
 *       draws, optionally padded with uninformative distractor coordinates; its posterior has
 *       four symmetric modes;</li>
 *   <li>{@link #gaussianMixture}: {@code x ~ 0.5 N(theta, I) + 0.5 N(theta, 0.01 I)};</li>
 *   <li>{@link #bernoulliGlm}: a logistic GLM with raw binary responses or their sufficient
 *       statistics;</li>
 *   <li>{@link #ricker}: a chaotic Ricker population model observed through Poisson counts,
 *       with Wood's (2010) hand summaries in {@link #rickerSummaries};</li>
 *   <li>{@link #sir} and {@link #seir}: stochastic epidemics observed as binomial prevalence
 *       samples;</li>
 *   <li>{@link #gaussianLinear}: a d-dimensional linear-Gaussian task with a closed-form
 *       posterior, and its heavy-tailed misspecified twin;</li>
 * </ul>
 * Tasks with a tractable likelihood expose it, and {@link Problem#reference} draws reference
 * posterior samples (closed form, or slice sampling with the exact likelihood, using the
 * known symmetries for SLCP). {@link #misspecified} pairs a simulator with a different "real"
 * data process for robustness studies. {@link DesignProblem}s ({@link #locationFinding},
 * {@link #deathProcess}, {@link #ces}, {@link #linearGaussianDesign}) serve experimental design.
 */
public final class SimulatorSuite {

    private SimulatorSuite() {
    }

    /** A tractable log likelihood. */
    @FunctionalInterface
    public interface Likelihood {
        /**
         * The log likelihood.
         *
         * @param theta the parameters
         * @param x the data
         * @return {@code log p(x | theta)}
         */
        double logLikelihood(double[] theta, double[] x);
    }

    /** Draws reference posterior samples. */
    @FunctionalInterface
    public interface Reference {
        /**
         * Draws posterior samples.
         *
         * @param observed the observation
         * @param trueTheta the parameters that generated it (a good MCMC start)
         * @param n the number of samples
         * @param random the random source
         * @return {@code [n][d]}
         */
        double[][] sample(double[] observed, double[] trueTheta, int n, SplittableRandom random);
    }

    /**
     * A task.
     *
     * @param name the name
     * @param dimension the parameter dimension
     * @param prior the prior
     * @param simulator the simulator
     * @param likelihood the likelihood, or {@code null} when intractable
     * @param reference the reference posterior sampler, or {@code null}
     */
    public record Problem(String name, int dimension, Prior prior, Simulator simulator, Likelihood likelihood,
            Reference reference) {

        /**
         * A ground-truth parameter and its simulated observation.
         *
         * @param seed the seed
         * @return {@code {theta, x}}
         */
        public double[][] observation(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            double[] t = prior.sample(r);
            return new double[][] {t, simulator.simulate(t, r)};
        }
    }

    /**
     * A misspecified task: inference uses {@code model}, the observations come from {@code real}.
     *
     * @param model the assumed simulator
     * @param real the true data process, same parameters
     */
    public record Misspecified(Problem model, Simulator real) {

        /**
         * A calibration set of real (theta, x) pairs.
         *
         * @param n the number of pairs
         * @param seed the seed
         * @return {@code {thetas [n][d], xs [n][dx]}}
         */
        public double[][][] calibration(int n, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            double[][] t = new double[n][];
            double[][] x = new double[n][];
            for (int i = 0; i < n; i++) {
                t[i] = model.prior().sample(r);
                x[i] = real.simulate(t[i], r);
            }
            return new double[][][] {t, x};
        }
    }

    /**
     * A box-uniform prior.
     *
     * @param low lower bounds
     * @param high upper bounds
     * @return the prior
     */
    public static Prior uniform(double[] low, double[] high) {
        return new Prior() {
            @Override
            public double[] sample(SplittableRandom random) {
                double[] t = new double[low.length];
                for (int i = 0; i < t.length; i++) {
                    t[i] = low[i] + (high[i] - low[i]) * random.nextDouble();
                }
                return t;
            }

            @Override
            public double logDensity(double[] theta) {
                double s = 0;
                for (int i = 0; i < theta.length; i++) {
                    if (theta[i] < low[i] || theta[i] > high[i]) {
                        return Double.NEGATIVE_INFINITY;
                    }
                    s -= Math.log(high[i] - low[i]);
                }
                return s;
            }
        };
    }

    /**
     * An isotropic Gaussian prior (normalised).
     *
     * @param mean the mean
     * @param sd the standard deviation
     * @return the prior
     */
    public static Prior gaussian(double[] mean, double sd) {
        return new Prior() {
            @Override
            public double[] sample(SplittableRandom random) {
                double[] t = new double[mean.length];
                for (int i = 0; i < t.length; i++) {
                    t[i] = mean[i] + sd * random.nextGaussian();
                }
                return t;
            }

            @Override
            public double logDensity(double[] theta) {
                double s = 0;
                for (int i = 0; i < theta.length; i++) {
                    double z = (theta[i] - mean[i]) / sd;
                    s += -0.5 * z * z - Math.log(sd) - 0.5 * Math.log(2 * Math.PI);
                }
                return s;
            }
        };
    }

    /**
     * Reference samples by slice sampling the exact posterior from the true parameters.
     *
     * @param prior the prior
     * @param likelihood the likelihood
     * @return the reference sampler
     */
    public static Reference sliceReference(Prior prior, Likelihood likelihood) {
        return (x, t, n, r) -> {
            double[][] chain = SliceSampler.chain(th -> {
                double lp = prior.logDensity(th);
                return Double.isFinite(lp) ? lp + likelihood.logLikelihood(th, x) : lp;
            }, t, 200, 2 * n, 0.5, r);
            double[][] out = new double[n][];
            for (int i = 0; i < n; i++) {
                out[i] = chain[2 * i];
            }
            return out;
        };
    }

    // ------------------------------------------------------------------ SLCP

    private static double[] slcpMoments(double[] t) {
        double s1 = t[2] * t[2];
        double s2 = t[3] * t[3];
        double rho = Math.tanh(t[4]);
        return new double[] {t[0], t[1], s1, s2, rho};
    }

    /**
     * SLCP: {@code theta ~ U(-3, 3)^5}, four i.i.d. 2-D Gaussian draws with mean
     * {@code (theta1, theta2)}, standard deviations {@code theta3^2, theta4^2} and correlation
     * {@code tanh(theta5)}, followed by {@code distractors} uninformative Student-t coordinates.
     *
     * @param distractors the number of distractor coordinates, 0 for plain SLCP
     * @return the task
     */
    public static Problem slcp(int distractors) {
        double[] lo = {-3, -3, -3, -3, -3};
        double[] hi = {3, 3, 3, 3, 3};
        Prior prior = uniform(lo, hi);
        Simulator sim = (t, r) -> {
            double[] m = slcpMoments(t);
            double[] x = new double[8 + distractors];
            for (int k = 0; k < 4; k++) {
                double z1 = r.nextGaussian();
                double z2 = r.nextGaussian();
                x[2 * k] = m[0] + m[2] * z1;
                x[2 * k + 1] = m[1] + m[3] * (m[4] * z1 + Math.sqrt(1 - m[4] * m[4]) * z2);
            }
            for (int k = 0; k < distractors; k++) {
                double chi = 0;
                for (int j = 0; j < 3; j++) {
                    double g = r.nextGaussian();
                    chi += g * g;
                }
                x[8 + k] = (k % 3 - 1) * 2 + r.nextGaussian() / Math.sqrt(chi / 3);
            }
            return x;
        };
        Likelihood lik = (t, x) -> {
            double[] m = slcpMoments(t);
            double s1 = Math.max(m[2], 1e-6);
            double s2 = Math.max(m[3], 1e-6);
            double rho = m[4];
            double one = Math.max(1 - rho * rho, 1e-12);
            double ll = 0;
            for (int k = 0; k < 4; k++) {
                double a = (x[2 * k] - m[0]) / s1;
                double b = (x[2 * k + 1] - m[1]) / s2;
                ll += -(a * a - 2 * rho * a * b + b * b) / (2 * one) - Math.log(2 * Math.PI * s1 * s2 * Math.sqrt(one));
            }
            return ll;
        };
        Reference slice = sliceReference(prior, lik);
        Reference ref = (x, t, n, r) -> {
            double[][] s = slice.sample(x, t, n, r);
            for (double[] v : s) {   // the likelihood depends on theta3 and theta4 only through their squares
                if (r.nextBoolean()) {
                    v[2] = -v[2];
                }
                if (r.nextBoolean()) {
                    v[3] = -v[3];
                }
            }
            return s;
        };
        return new Problem(distractors == 0 ? "SLCP" : "SLCP+distractors", 5, prior, sim, lik, ref);
    }

    // ------------------------------------------------------------------ Gaussian mixture

    /**
     * The Gaussian mixture task: {@code theta ~ U(-10, 10)^2},
     * {@code x ~ 0.5 N(theta, I) + 0.5 N(theta, 0.01 I)}; its posterior is the same mixture
     * centred on {@code x} (truncated to the prior box).
     *
     * @return the task
     */
    public static Problem gaussianMixture() {
        Prior prior = uniform(new double[] {-10, -10}, new double[] {10, 10});
        Simulator sim = (t, r) -> {
            double s = r.nextBoolean() ? 1 : 0.1;
            return new double[] {t[0] + s * r.nextGaussian(), t[1] + s * r.nextGaussian()};
        };
        Likelihood lik = (t, x) -> {
            double d2 = (x[0] - t[0]) * (x[0] - t[0]) + (x[1] - t[1]) * (x[1] - t[1]);
            double a = Math.log(0.5) - d2 / 2 - Math.log(2 * Math.PI);
            double b = Math.log(0.5) - d2 / 0.02 - Math.log(2 * Math.PI * 0.01);
            return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
        };
        Reference ref = (x, t, n, r) -> {
            double[][] out = new double[n][];
            int k = 0;
            while (k < n) {
                double s = r.nextBoolean() ? 1 : 0.1;
                double[] v = {x[0] + s * r.nextGaussian(), x[1] + s * r.nextGaussian()};
                if (Math.abs(v[0]) <= 10 && Math.abs(v[1]) <= 10) {
                    out[k++] = v;
                }
            }
            return out;
        };
        return new Problem("Gaussian mixture", 2, prior, sim, lik, ref);
    }

    // ------------------------------------------------------------------ Bernoulli GLM

    /**
     * A Bernoulli GLM: {@code theta ~ N(0, I_4)} (intercept and three slopes), 60 trials with
     * fixed standard normal covariates, {@code y_i ~ Bernoulli(sigmoid(v_i . theta))}.
     *
     * @param summarised return the sufficient statistics {@code V^T y} (4 numbers) instead of
     *     the raw 60 responses
     * @return the task
     */
    public static Problem bernoulliGlm(boolean summarised) {
        int trials = 60;
        SplittableRandom design = new SplittableRandom(11);
        double[][] v = new double[trials][4];
        for (double[] row : v) {
            row[0] = 1;
            for (int j = 1; j < 4; j++) {
                row[j] = design.nextGaussian();
            }
        }
        Prior prior = gaussian(new double[4], 1);
        Simulator sim = (t, r) -> {
            double[] y = new double[trials];
            for (int i = 0; i < trials; i++) {
                double eta = 0;
                for (int j = 0; j < 4; j++) {
                    eta += v[i][j] * t[j];
                }
                y[i] = r.nextDouble() < 1 / (1 + Math.exp(-eta)) ? 1 : 0;
            }
            if (!summarised) {
                return y;
            }
            double[] s = new double[4];
            for (int i = 0; i < trials; i++) {
                for (int j = 0; j < 4; j++) {
                    s[j] += v[i][j] * y[i];
                }
            }
            return s;
        };
        Likelihood lik;
        if (summarised) {
            // the summary is sufficient: log p(y | theta) = theta . V^T y - sum log(1 + e^eta)
            lik = (t, s) -> {
                double ll = 0;
                for (int j = 0; j < 4; j++) {
                    ll += t[j] * s[j];
                }
                for (int i = 0; i < trials; i++) {
                    double eta = 0;
                    for (int j = 0; j < 4; j++) {
                        eta += v[i][j] * t[j];
                    }
                    ll -= Math.max(eta, 0) + Math.log1p(Math.exp(-Math.abs(eta)));
                }
                return ll;
            };
        } else {
            lik = (t, y) -> {
                double ll = 0;
                for (int i = 0; i < trials; i++) {
                    double eta = 0;
                    for (int j = 0; j < 4; j++) {
                        eta += v[i][j] * t[j];
                    }
                    ll += y[i] * eta - Math.max(eta, 0) - Math.log1p(Math.exp(-Math.abs(eta)));
                }
                return ll;
            };
        }
        return new Problem(summarised ? "Bernoulli GLM (summary)" : "Bernoulli GLM (raw)", 4, prior, sim, lik,
                sliceReference(prior, lik));
    }

    // ------------------------------------------------------------------ Ricker

    /**
     * The Ricker model (Wood, Nature 2010): {@code N_{t+1} = r N_t exp(-N_t + e_t)},
     * {@code e_t ~ N(0, sigma^2)}, {@code y_t ~ Poisson(phi N_t)} for 50 steps after 20 burn-in
     * steps; parameters {@code (log r, sigma, phi)} with a uniform prior on
     * {@code [3, 5] x [0.1, 0.6] x [5, 15]}. Returns the raw counts.
     *
     * @param extraProcess when true, adds an unmodelled immigration process (the misspecified
     *     "real" world): {@code N_t += 0.4} with probability 0.2 each step
     * @return the task (likelihood intractable)
     */
    public static Problem ricker(boolean extraProcess) {
        Prior prior = uniform(new double[] {3, 0.1, 5}, new double[] {5, 0.6, 15});
        Simulator sim = (t, r) -> {
            double n = 1;
            double rate = Math.exp(t[0]);
            double[] y = new double[50];
            for (int s = 0; s < 70; s++) {
                n = rate * n * Math.exp(-n + t[1] * r.nextGaussian());
                if (extraProcess && r.nextDouble() < 0.2) {
                    n += 0.4;
                }
                if (s >= 20) {
                    y[s - 20] = poisson(t[2] * n, r);
                }
            }
            return y;
        };
        return new Problem(extraProcess ? "Ricker+immigration" : "Ricker", 3, prior, sim, null, null);
    }

    /**
     * Wood's hand-crafted Ricker summaries: mean, number of zeros, autocovariances at lags 1-2,
     * mean of the square root, and quartiles of the counts.
     *
     * @param y the raw counts
     * @return 8 summaries
     */
    public static double[] rickerSummaries(double[] y) {
        int n = y.length;
        double mean = 0;
        double zeros = 0;
        double sq = 0;
        for (double v : y) {
            mean += v / n;
            zeros += v == 0 ? 1 : 0;
            sq += Math.sqrt(v) / n;
        }
        double a1 = 0;
        double a2 = 0;
        for (int i = 0; i + 1 < n; i++) {
            a1 += (y[i] - mean) * (y[i + 1] - mean) / n;
            if (i + 2 < n) {
                a2 += (y[i] - mean) * (y[i + 2] - mean) / n;
            }
        }
        double[] sorted = y.clone();
        java.util.Arrays.sort(sorted);
        return new double[] {mean, zeros, a1 / 10, a2 / 10, sq, sorted[n / 4], sorted[n / 2], sorted[3 * n / 4]};
    }

    /**
     * A Poisson draw (inversion for small means, a rounded normal for large ones).
     *
     * @param mean the mean
     * @param r the random source
     * @return the count
     */
    public static double poisson(double mean, SplittableRandom r) {
        if (mean > 50) {
            return Math.max(0, Math.round(mean + Math.sqrt(mean) * r.nextGaussian()));
        }
        double l = Math.exp(-mean);
        double p = r.nextDouble();
        int k = 0;
        double term = l;
        double cdf = l;
        while (p > cdf && k < 1000) {
            k++;
            term *= mean / k;
            cdf += term;
        }
        return k;
    }

    // ------------------------------------------------------------------ epidemics

    private static double binomial(double n, double p, SplittableRandom r) {
        if (n <= 0 || p <= 0) {
            return 0;
        }
        if (n > 40) {
            double m = n * p;
            double v = n * p * (1 - p);
            return Math.max(0, Math.min(n, Math.round(m + Math.sqrt(v) * r.nextGaussian())));
        }
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (r.nextDouble() < p) {
                k++;
            }
        }
        return k;
    }

    private static double[] epidemic(double beta, double gamma, double sigma, int delay, SplittableRandom r) {
        double pop = 1000;
        double s = pop - 10;
        double e = 0;
        double i = sigma > 0 ? 0 : 10;
        if (sigma > 0) {
            e = 10;
        }
        double[] prevalence = new double[161];
        prevalence[0] = i;
        for (int day = 1; day <= 160; day++) {
            double newInf = binomial(s, 1 - Math.exp(-beta * i / pop), r);
            double newRec = binomial(i, 1 - Math.exp(-gamma), r);
            if (sigma > 0) {
                double newSym = binomial(e, 1 - Math.exp(-sigma), r);
                e += newInf - newSym;
                i += newSym - newRec;
            } else {
                i += newInf - newRec;
            }
            s -= newInf;
            prevalence[day] = i;
        }
        double[] x = new double[10];
        for (int k = 0; k < 10; k++) {
            int day = Math.max(0, 16 * (k + 1) - delay);
            x[k] = binomial(1000, prevalence[day] / pop, r) / 1000.0;
        }
        return x;
    }

    /**
     * A stochastic SIR epidemic (population 1000, 10 initially infected, 160 days, binomial
     * transitions), observed as 10 binomial prevalence samples of size 1000 every 16 days.
     * Parameters {@code (beta, gamma)} with log-normal priors around {@code (0.4, 0.125)},
     * truncated to positive values by working on {@code theta = log(rate)}.
     *
     * @param reportingDelay an unmodelled delay in days of the observations (0 for the model)
     * @return the task, parameterised by the log rates
     */
    public static Problem sir(int reportingDelay) {
        Prior prior = new Prior() {
            @Override
            public double[] sample(SplittableRandom random) {
                return new double[] {Math.log(0.4) + 0.5 * random.nextGaussian(),
                    Math.log(0.125) + 0.2 * random.nextGaussian()};
            }

            @Override
            public double logDensity(double[] t) {
                double a = (t[0] - Math.log(0.4)) / 0.5;
                double b = (t[1] - Math.log(0.125)) / 0.2;
                return -0.5 * (a * a + b * b);
            }
        };
        Simulator sim = (t, r) -> epidemic(Math.exp(t[0]), Math.exp(t[1]), 0, reportingDelay, r);
        return new Problem(reportingDelay == 0 ? "SIR" : "SIR+delay", 2, prior, sim, null, null);
    }

    /**
     * The SEIR twin of {@link #sir}: an exposed compartment with rate 0.2 per day, same prior
     * and observation model (for model comparison).
     *
     * @return the task
     */
    public static Problem seir() {
        Problem sir = sir(0);
        Simulator sim = (t, r) -> epidemic(Math.exp(t[0]), Math.exp(t[1]), 0.2, 0, r);
        return new Problem("SEIR", 2, sir.prior(), sim, null, null);
    }

    // ------------------------------------------------------------------ Gaussian linear

    /**
     * A linear-Gaussian task: {@code theta ~ N(0, I_d)}, {@code x = theta + N(0, s^2 I)} with
     * {@code s = 0.5}, and a closed-form Gaussian posterior.
     *
     * @param d the dimension
     * @return the task
     */
    public static Problem gaussianLinear(int d) {
        double s = 0.5;
        Prior prior = gaussian(new double[d], 1);
        Simulator sim = (t, r) -> {
            double[] x = new double[d];
            for (int j = 0; j < d; j++) {
                x[j] = t[j] + s * r.nextGaussian();
            }
            return x;
        };
        Likelihood lik = (t, x) -> {
            double ll = 0;
            for (int j = 0; j < d; j++) {
                double z = (x[j] - t[j]) / s;
                ll += -0.5 * z * z - Math.log(s) - 0.5 * Math.log(2 * Math.PI);
            }
            return ll;
        };
        double prec = 1 + 1 / (s * s);
        Reference ref = (x, t, n, r) -> {
            double[][] out = new double[n][d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    out[i][j] = x[j] / (s * s) / prec + r.nextGaussian() / Math.sqrt(prec);
                }
            }
            return out;
        };
        return new Problem("Gaussian linear d=" + d, d, prior, sim, lik, ref);
    }

    /**
     * The analytic posterior mean and standard deviation of {@link #gaussianLinear} for one
     * coordinate.
     *
     * @param x the observed coordinate
     * @return {@code {mean, sd}}
     */
    public static double[] gaussianLinearPosterior(double x) {
        double prec = 1 + 1 / 0.25;
        return new double[] {x / 0.25 / prec, 1 / Math.sqrt(prec)};
    }

    // ------------------------------------------------------------------ misspecification

    /**
     * Misspecified pairs.
     *
     * @param name {@code "gaussian-heavy-tails"} (real noise Student-t with 2 degrees of freedom,
     *     scaled 3x, plus a 1.0 bias on the first coordinate), {@code "sir-delay"} (real
     *     observations 12 days late) or {@code "ricker-immigration"} (an extra immigration
     *     process)
     * @return the pair
     */
    public static Misspecified misspecified(String name) {
        return switch (name) {
            case "gaussian-heavy-tails" -> {
                Problem p = gaussianLinear(2);
                Simulator real = (t, r) -> {
                    double[] x = new double[2];
                    for (int j = 0; j < 2; j++) {
                        double chi = 0;
                        for (int k = 0; k < 2; k++) {
                            double g = r.nextGaussian();
                            chi += g * g;
                        }
                        x[j] = t[j] + (j == 0 ? 1.0 : 0) + 1.5 * r.nextGaussian() / Math.sqrt(chi / 2);
                    }
                    return x;
                };
                yield new Misspecified(p, real);
            }
            case "sir-delay" -> new Misspecified(sir(0), sir(12).simulator());
            case "ricker-immigration" -> new Misspecified(ricker(false), ricker(true).simulator());
            default -> throw new IllegalArgumentException("unknown misspecified task " + name);
        };
    }

    /**
     * The log density of a Student-t location-scale distribution.
     *
     * @param x the value
     * @param mu the location
     * @param scale the scale
     * @param nu the degrees of freedom
     * @return the log density
     */
    public static double studentTLogDensity(double x, double mu, double scale, double nu) {
        double z = (x - mu) / scale;
        return Special.lgamma((nu + 1) / 2) - Special.lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - Math.log(scale)
                - (nu + 1) / 2 * Math.log1p(z * z / nu);
    }

    // ------------------------------------------------------------------ design problems

    /** A Bayesian experimental design problem with a scalar outcome and a tractable likelihood. */
    public interface DesignProblem {
        /**
         * The name.
         *
         * @return the name
         */
        String name();

        /**
         * The parameter dimension.
         *
         * @return the dimension
         */
        int parameterDim();

        /**
         * The lower corner of the design box.
         *
         * @return the bounds
         */
        double[] designLow();

        /**
         * The upper corner of the design box.
         *
         * @return the bounds
         */
        double[] designHigh();

        /**
         * Draws parameters from the prior.
         *
         * @param random the random source
         * @return {@code theta}
         */
        double[] samplePrior(SplittableRandom random);

        /**
         * Simulates an outcome.
         *
         * @param theta the parameters
         * @param design the design
         * @param random the random source
         * @return the outcome
         */
        double simulate(double[] theta, double[] design, SplittableRandom random);

        /**
         * The log likelihood of an outcome.
         *
         * @param y the outcome
         * @param theta the parameters
         * @param design the design
         * @return {@code log p(y | theta, design)}
         */
        double logLikelihood(double y, double[] theta, double[] design);
    }

    private record SimpleDesignProblem(String name, int parameterDim, double[] designLow, double[] designHigh,
            java.util.function.Function<SplittableRandom, double[]> prior, DesignSim sim, DesignLik lik)
            implements DesignProblem {
        @Override
        public double[] samplePrior(SplittableRandom random) {
            return prior.apply(random);
        }

        @Override
        public double simulate(double[] theta, double[] design, SplittableRandom random) {
            return sim.simulate(theta, design, random);
        }

        @Override
        public double logLikelihood(double y, double[] theta, double[] design) {
            return lik.logLikelihood(y, theta, design);
        }
    }

    @FunctionalInterface
    private interface DesignSim {
        double simulate(double[] theta, double[] design, SplittableRandom random);
    }

    @FunctionalInterface
    private interface DesignLik {
        double logLikelihood(double y, double[] theta, double[] design);
    }

    /** Location finding's observation noise. */
    public static final double LOCATION_NOISE = 0.5;

    /**
     * The mean signal of location finding: {@code log(0.1 + sum_k 1 / (1e-4 + |theta_k - xi|^2))}.
     *
     * @param theta the source positions {@code [2K]}
     * @param design the 2-D measurement point
     * @return the mean of the log intensity
     */
    public static double locationMean(double[] theta, double[] design) {
        double s = 0.1;
        for (int k = 0; k < theta.length / 2; k++) {
            double dx = theta[2 * k] - design[0];
            double dy = theta[2 * k + 1] - design[1];
            s += 1 / (1e-4 + dx * dx + dy * dy);
        }
        return Math.log(s);
    }

    /**
     * Location finding (Foster et al. 2021): {@code K} sources {@code theta_k ~ N(0, I_2)} emit a
     * signal whose log total intensity at a 2-D design {@code xi} in {@code [-4, 4]^2} is observed
     * with Gaussian noise of standard deviation {@link #LOCATION_NOISE}.
     *
     * @param sources {@code K}
     * @return the problem
     */
    public static DesignProblem locationFinding(int sources) {
        return new SimpleDesignProblem("location finding K=" + sources, 2 * sources, new double[] {-4, -4},
                new double[] {4, 4}, r -> {
                    double[] t = new double[2 * sources];
                    for (int i = 0; i < t.length; i++) {
                        t[i] = r.nextGaussian();
                    }
                    return t;
                }, (t, xi, r) -> locationMean(t, xi) + LOCATION_NOISE * r.nextGaussian(),
                (y, t, xi) -> {
                    double z = (y - locationMean(t, xi)) / LOCATION_NOISE;
                    return -0.5 * z * z - Math.log(LOCATION_NOISE) - 0.5 * Math.log(2 * Math.PI);
                });
    }

    /**
     * The death process (Cook, Gibson and Gilligan 2008): 50 individuals, each infected by time
     * {@code xi} in {@code [0.01, 4]} with probability {@code 1 - exp(-b xi)}; infection rate
     * {@code log b ~ N(0, 0.5^2)}; the outcome is the number infected.
     *
     * @return the problem, parameterised by {@code b}
     */
    public static DesignProblem deathProcess() {
        return new SimpleDesignProblem("death process", 1, new double[] {0.01}, new double[] {4},
                r -> new double[] {Math.exp(0.5 * r.nextGaussian())},
                (t, xi, r) -> binomial(50, 1 - Math.exp(-t[0] * xi[0]), r),
                (y, t, xi) -> {
                    double p = Math.min(1 - 1e-12, Math.max(1e-12, 1 - Math.exp(-t[0] * xi[0])));
                    return Special.lgamma(51) - Special.lgamma(y + 1) - Special.lgamma(51 - y) + y * Math.log(p)
                            + (50 - y) * Math.log1p(-p);
                });
    }

    private static double cesUtility(double[] t, double[] x, int offset) {
        double rho = Math.max(1e-3, t[0]);
        double s = 0;
        for (int i = 0; i < 3; i++) {
            s += t[1 + i] * Math.pow(Math.max(x[offset + i], 1e-9), rho);
        }
        return Math.pow(s, 1 / rho);
    }

    private static double[] cesMoments(double[] t, double[] xi) {
        double u = Math.exp(t[4]);
        double dist = 0;
        for (int i = 0; i < 3; i++) {
            dist += (xi[i] - xi[3 + i]) * (xi[i] - xi[3 + i]);
        }
        return new double[] {u * (cesUtility(t, xi, 0) - cesUtility(t, xi, 3)), 0.005 * u * (1 + Math.sqrt(dist))};
    }

    /**
     * Constant-elasticity-of-substitution preference elicitation (Foster et al. 2020): a design
     * is two baskets of three goods in {@code [0, 100]^6}; parameters {@code (rho, alpha_1..3,
     * log u)} with {@code rho ~ U(0, 1)}, {@code alpha ~ Dirichlet(1, 1, 1)},
     * {@code log u ~ N(1, 3^2)}; the outcome is a censored logistic-normal preference in
     * {@code [0.01, 0.99]}.
     *
     * @return the problem
     */
    public static DesignProblem ces() {
        double eps = 0.01;
        double lo = Math.log(eps / (1 - eps));
        return new SimpleDesignProblem("CES", 5, new double[6], new double[] {100, 100, 100, 100, 100, 100}, r -> {
            double[] g = new double[3];
            double tot = 0;
            for (int i = 0; i < 3; i++) {
                g[i] = -Math.log(1 - r.nextDouble());
                tot += g[i];
            }
            return new double[] {r.nextDouble(), g[0] / tot, g[1] / tot, g[2] / tot, 1 + 3 * r.nextGaussian()};
        }, (t, xi, r) -> {
            double[] m = cesMoments(t, xi);
            double eta = m[0] + m[1] * r.nextGaussian();
            return Math.min(1 - eps, Math.max(eps, 1 / (1 + Math.exp(-eta))));
        }, (y, t, xi) -> {
            double[] m = cesMoments(t, xi);
            double sd = Math.max(m[1], 1e-9);
            if (y <= eps + 1e-12) {
                return logNormalCdf((lo - m[0]) / sd);
            }
            if (y >= 1 - eps - 1e-12) {
                return logNormalCdf((m[0] + lo) / sd);
            }
            double z = (Math.log(y / (1 - y)) - m[0]) / sd;
            return -0.5 * z * z - Math.log(sd) - 0.5 * Math.log(2 * Math.PI) - Math.log(y * (1 - y));
        });
    }

    private static double logNormalCdf(double z) {
        return Math.log(Math.max(0.5 * Special.erfc(-z / Math.sqrt(2)), 1e-300));
    }

    /**
     * A linear-Gaussian design with a closed-form expected information gain: {@code theta ~ N(0,
     * diag(1, 4))}, {@code y = xi . theta + N(0, 1)}, {@code xi in [-1, 1]^2}.
     *
     * @return the problem
     */
    public static DesignProblem linearGaussianDesign() {
        return new SimpleDesignProblem("linear Gaussian", 2, new double[] {-1, -1}, new double[] {1, 1},
                r -> new double[] {r.nextGaussian(), 2 * r.nextGaussian()},
                (t, xi, r) -> xi[0] * t[0] + xi[1] * t[1] + r.nextGaussian(),
                (y, t, xi) -> {
                    double z = y - xi[0] * t[0] - xi[1] * t[1];
                    return -0.5 * z * z - 0.5 * Math.log(2 * Math.PI);
                });
    }

    /**
     * The exact expected information gain of {@link #linearGaussianDesign}:
     * {@code 0.5 log(1 + xi^T Sigma xi)}.
     *
     * @param xi the design
     * @return the EIG in nats
     */
    public static double linearGaussianEig(double[] xi) {
        return 0.5 * Math.log(1 + xi[0] * xi[0] + 4 * xi[1] * xi[1]);
    }
}
