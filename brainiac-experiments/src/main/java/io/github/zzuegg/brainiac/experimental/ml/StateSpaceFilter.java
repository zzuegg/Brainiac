package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * Bayesian filtering in state-space models {@code x_t = f(x_t-1) + w}, {@code y_t = h(x_t) + v}:
 * <ul>
 *   <li>{@link #kalman}: the Kalman filter for linear-Gaussian models, with the
 *       Rauch-Tung-Striebel smoother (its {@code smooth} flag) and the exact log-likelihood
 *       from the innovations (Welch and Bishop,
 *       <a href="https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf">An introduction to
 *       the Kalman filter</a>);</li>
 *   <li>{@link #extendedKalman}: the extended Kalman filter, which linearises {@code f} and
 *       {@code h} around the current estimate with their Jacobians;</li>
 *   <li>{@link #particle}: the bootstrap particle filter (Gordon, Salmond and Smith 1993,
 *       <a href="https://ieeexplore.ieee.org/document/210672">Novel approach to
 *       nonlinear/non-Gaussian Bayesian state estimation</a>): particles propagated through the
 *       dynamics, weighted by the observation likelihood and systematically resampled when the
 *       effective sample size falls below half; it represents multimodal posteriors that the
 *       Gaussian filters cannot.</li>
 * </ul>
 */
public final class StateSpaceFilter {

    private StateSpaceFilter() {
    }

    /**
     * Filtered (and possibly smoothed) state estimates.
     *
     * @param means the posterior means {@code [T][n]}
     * @param covariances the posterior covariances {@code [T][n][n]}
     * @param logLikelihood {@code log p(y_1..T)} (or its particle estimate)
     */
    public record Estimate(double[][] means, double[][][] covariances, double logLikelihood) {

        /**
         * The variance of one state component over time.
         *
         * @param component the component
         * @return {@code [T]}
         */
        public double[] variance(int component) {
            double[] v = new double[means.length];
            for (int t = 0; t < v.length; t++) {
                v[t] = covariances[t][component][component];
            }
            return v;
        }

        /**
         * The mean of one state component over time.
         *
         * @param component the component
         * @return {@code [T]}
         */
        public double[] mean(int component) {
            double[] m = new double[means.length];
            for (int t = 0; t < m.length; t++) {
                m[t] = means[t][component];
            }
            return m;
        }
    }

    /**
     * A linear-Gaussian model.
     *
     * @param f the transition matrix {@code [n][n]}
     * @param q the process noise covariance {@code [n][n]}
     * @param h the observation matrix {@code [m][n]}
     * @param r the observation noise covariance {@code [m][m]}
     * @param x0 the prior mean {@code [n]}
     * @param p0 the prior covariance {@code [n][n]}
     */
    public record LinearModel(double[][] f, double[][] q, double[][] h, double[][] r, double[] x0, double[][] p0) {
    }

    /** A differentiable map with its Jacobian, for the extended Kalman filter. */
    public interface Differentiable {

        /**
         * Evaluates the map.
         *
         * @param x the state
         * @return the image
         */
        double[] apply(double[] x);

        /**
         * The Jacobian at a point.
         *
         * @param x the state
         * @return {@code d apply / d x}
         */
        double[][] jacobian(double[] x);
    }

    private static double[] mv(double[][] a, double[] x) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = MlMatrices.dot(a[i], x);
        }
        return y;
    }

    private static double[][] add(double[][] a, double[][] b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    /** {@code a b a^T}. */
    private static double[][] sandwich(double[][] a, double[][] b) {
        return MlMatrices.multiply(MlMatrices.multiply(a, b), MlMatrices.transpose(a));
    }

    private static double[][] symmetric(double[][] a) {
        double[][] s = new double[a.length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                s[i][j] = 0.5 * (a[i][j] + a[j][i]);
            }
        }
        return s;
    }

    /**
     * One measurement update; returns {updated mean, updated covariance, log-likelihood term}.
     */
    private static Object[] update(double[] mean, double[][] cov, double[] y, double[] predictedY, double[][] hj,
            double[][] r) {
        int m = y.length;
        double[][] s = add(sandwich(hj, cov), r);
        double[][] sChol = MlMatrices.cholesky(s);
        double[] innov = new double[m];
        for (int i = 0; i < m; i++) {
            innov[i] = y[i] - predictedY[i];
        }
        // K = P H^T S^-1
        double[][] pht = MlMatrices.multiply(cov, MlMatrices.transpose(hj));          // [n][m]
        double[][] kT = MlMatrices.choleskySolve(sChol, MlMatrices.transpose(pht));   // [m][n]
        double[][] k = MlMatrices.transpose(kT);
        double[] next = mean.clone();
        double[] kv = mv(k, innov);
        for (int i = 0; i < next.length; i++) {
            next[i] += kv[i];
        }
        // Joseph-free form: P - K S K^T
        double[][] ksk = sandwich(k, s);
        double[][] p = new double[cov.length][cov.length];
        for (int i = 0; i < p.length; i++) {
            for (int j = 0; j < p.length; j++) {
                p[i][j] = cov[i][j] - ksk[i][j];
            }
        }
        double[] z = MlMatrices.forwardSolve(sChol, innov);
        double ll = -0.5 * (m * Math.log(2 * Math.PI) + MlMatrices.logDet(sChol) + MlMatrices.dot(z, z));
        return new Object[] {next, symmetric(p), ll};
    }

    /**
     * The Kalman filter, and optionally the RTS smoother.
     *
     * @param model the linear-Gaussian model
     * @param ys observations {@code [T][m]}
     * @param smooth whether to run the backward smoothing pass
     * @return filtered (or smoothed) estimates and the log-likelihood
     */
    public static Estimate kalman(LinearModel model, double[][] ys, boolean smooth) {
        int steps = ys.length;
        int n = model.x0().length;
        double[][] means = new double[steps][];
        double[][][] covs = new double[steps][][];
        double[][] predMeans = new double[steps][];
        double[][][] predCovs = new double[steps][][];
        double[] mean = model.x0();
        double[][] cov = model.p0();
        double ll = 0;
        for (int t = 0; t < steps; t++) {
            if (t > 0) {
                mean = mv(model.f(), mean);
                cov = add(sandwich(model.f(), cov), model.q());
            }
            predMeans[t] = mean;
            predCovs[t] = cov;
            Object[] u = update(mean, cov, ys[t], mv(model.h(), mean), model.h(), model.r());
            mean = (double[]) u[0];
            cov = (double[][]) u[1];
            ll += (double) u[2];
            means[t] = mean;
            covs[t] = cov;
        }
        if (smooth) {
            for (int t = steps - 2; t >= 0; t--) {
                // C = P_t F^T Ppred_t+1^-1
                double[][] pc = MlMatrices.cholesky(predCovs[t + 1]);
                double[][] ct = MlMatrices.choleskySolve(pc, MlMatrices.multiply(model.f(), covs[t]));  // C^T
                double[][] c = MlMatrices.transpose(ct);
                double[] dm = new double[n];
                for (int i = 0; i < n; i++) {
                    dm[i] = means[t + 1][i] - predMeans[t + 1][i];
                }
                double[] cdm = mv(c, dm);
                double[] sm = means[t].clone();
                for (int i = 0; i < n; i++) {
                    sm[i] += cdm[i];
                }
                double[][] dp = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        dp[i][j] = covs[t + 1][i][j] - predCovs[t + 1][i][j];
                    }
                }
                means[t] = sm;
                covs[t] = symmetric(add(covs[t], sandwich(c, dp)));
            }
        }
        return new Estimate(means, covs, ll);
    }

    /**
     * The extended Kalman filter.
     *
     * @param f the transition, with its Jacobian
     * @param q the process noise covariance
     * @param h the observation map, with its Jacobian
     * @param r the observation noise covariance
     * @param x0 the prior mean
     * @param p0 the prior covariance
     * @param ys observations {@code [T][m]}
     * @return the filtered estimates and the approximate log-likelihood
     */
    public static Estimate extendedKalman(Differentiable f, double[][] q, Differentiable h, double[][] r, double[] x0,
            double[][] p0, double[][] ys) {
        int steps = ys.length;
        double[][] means = new double[steps][];
        double[][][] covs = new double[steps][][];
        double[] mean = x0;
        double[][] cov = p0;
        double ll = 0;
        for (int t = 0; t < steps; t++) {
            if (t > 0) {
                double[][] fj = f.jacobian(mean);
                mean = f.apply(mean);
                cov = add(sandwich(fj, cov), q);
            }
            Object[] u = update(mean, cov, ys[t], h.apply(mean), h.jacobian(mean), r);
            mean = (double[]) u[0];
            cov = (double[][]) u[1];
            ll += (double) u[2];
            means[t] = mean;
            covs[t] = cov;
        }
        return new Estimate(means, covs, ll);
    }

    /** Draws a state from the dynamics. */
    @FunctionalInterface
    public interface Transition {

        /**
         * Samples the next state.
         *
         * @param x the current state
         * @param random the source
         * @return the next state
         */
        double[] sample(double[] x, SplittableRandom random);
    }

    /** The observation log-likelihood. */
    @FunctionalInterface
    public interface Likelihood {

        /**
         * Scores an observation.
         *
         * @param y the observation
         * @param x a state
         * @return {@code log p(y | x)}
         */
        double log(double[] y, double[] x);
    }

    /**
     * The bootstrap particle filter.
     *
     * @param initial draws a particle from the prior
     * @param transition samples the dynamics
     * @param likelihood the observation log-likelihood
     * @param ys observations {@code [T][m]}
     * @param particles the number of particles
     * @param seed the seed
     * @return the weighted means and covariances, and the log-likelihood estimate
     */
    public static Estimate particle(java.util.function.Function<SplittableRandom, double[]> initial,
            Transition transition, Likelihood likelihood, double[][] ys, int particles, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        double[][] x = new double[particles][];
        for (int i = 0; i < particles; i++) {
            x[i] = initial.apply(random);
        }
        int n = x[0].length;
        double[] logW = new double[particles];
        double[][] means = new double[ys.length][];
        double[][][] covs = new double[ys.length][][];
        double ll = 0;
        for (int t = 0; t < ys.length; t++) {
            if (t > 0) {
                for (int i = 0; i < particles; i++) {
                    x[i] = transition.sample(x[i], random);
                }
            }
            double max = Double.NEGATIVE_INFINITY;
            double[] inc = new double[particles];
            for (int i = 0; i < particles; i++) {
                inc[i] = likelihood.log(ys[t], x[i]);
                logW[i] += inc[i];
                max = Math.max(max, logW[i]);
            }
            // log p(y_t | y_<t) = log sum_i w_i(t-1) p(y_t | x_i), with normalised previous weights
            double prevNorm = 0;
            double num = 0;
            for (int i = 0; i < particles; i++) {
                double prev = logW[i] - inc[i];
                prevNorm += Math.exp(prev - max);
                num += Math.exp(logW[i] - max);
            }
            ll += Math.log(num) - Math.log(prevNorm);
            double[] w = new double[particles];
            double sum = 0;
            for (int i = 0; i < particles; i++) {
                w[i] = Math.exp(logW[i] - max);
                sum += w[i];
            }
            double ess = 0;
            for (int i = 0; i < particles; i++) {
                w[i] /= sum;
                ess += w[i] * w[i];
            }
            ess = 1 / ess;
            double[] m = new double[n];
            for (int i = 0; i < particles; i++) {
                for (int c = 0; c < n; c++) {
                    m[c] += w[i] * x[i][c];
                }
            }
            double[][] cov = new double[n][n];
            for (int i = 0; i < particles; i++) {
                for (int a = 0; a < n; a++) {
                    for (int b = 0; b < n; b++) {
                        cov[a][b] += w[i] * (x[i][a] - m[a]) * (x[i][b] - m[b]);
                    }
                }
            }
            means[t] = m;
            covs[t] = cov;
            if (ess < particles / 2.0) {
                double[][] next = new double[particles][];
                double u = random.nextDouble() / particles;
                double c = w[0];
                int j = 0;
                for (int i = 0; i < particles; i++) {
                    double target = u + i / (double) particles;
                    while (target > c && j < particles - 1) {
                        j++;
                        c += w[j];
                    }
                    next[i] = x[j].clone();
                }
                x = next;
                java.util.Arrays.fill(logW, 0);
            } else {
                for (int i = 0; i < particles; i++) {
                    logW[i] = Math.log(Math.max(w[i], 1e-300));
                }
            }
        }
        return new Estimate(means, covs, ll);
    }
}
