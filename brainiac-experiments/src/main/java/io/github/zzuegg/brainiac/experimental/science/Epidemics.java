package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.tensor.Distributions;
import io.github.zzuegg.brainiac.tensor.Optimize;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Epidemic inference from case counts.
 *
 * <ul>
 *   <li><b>Compartmental fits</b>: SIR and SEIR ordinary differential equations (RK4) fitted to
 *       daily counts by a Poisson or negative-binomial likelihood with core
 *       {@link Optimize#nelderMead} on log parameters, with a profile-likelihood interval for
 *       {@code R0 = beta / gamma} (Kermack and McKendrick, 1927).</li>
 *   <li><b>Time-varying reproduction number</b> from the renewal equation
 *       {@code I_t ~ Poisson(R_t Lambda_t)}, {@code Lambda_t = sum_s w_s I_(t-s)}:
 *       {@link RtMethod#EPIESTIM}, the conjugate gamma posterior over a sliding window (Cori et
 *       al., AJE 2013); {@link RtMethod#RW_KALMAN}, a random walk on {@code log R_t} with a
 *       linearised (log-count) Gaussian observation, Kalman filter and RTS smoother; and
 *       {@link RtMethod#RW_PARTICLE}, the same random walk with the exact Poisson likelihood in a
 *       bootstrap particle filter, smoothed along the surviving genealogies.</li>
 *   <li><b>Nowcasting</b> reporting delays by a multiplicative chain-ladder on the reporting
 *       triangle: delay proportions from complete rows scale up the incomplete recent rows
 *       (Abbott et al., Wellcome Open Research 2020, use a Bayesian version).</li>
 * </ul>
 */
public final class Epidemics {

    /** How {@code R_t} is estimated. */
    public enum RtMethod {
        /** Cori et al.'s sliding-window gamma posterior. */
        EPIESTIM,
        /** Random walk on log R_t, linearised Kalman smoother. */
        RW_KALMAN,
        /** Random walk on log R_t, bootstrap particle smoother. */
        RW_PARTICLE
    }

    /**
     * An {@code R_t} estimate.
     *
     * @param mean posterior mean per day (NaN where undefined)
     * @param lower 5% quantile
     * @param upper 95% quantile
     */
    public record RtEstimate(double[] mean, double[] lower, double[] upper) {
    }

    /**
     * A compartmental fit.
     *
     * @param beta the transmission rate
     * @param gamma the recovery rate
     * @param initial the initially infectious fraction times the population
     * @param ascertainment the fitted reporting fraction
     * @param nll the negative log-likelihood
     * @param r0 {@code beta / gamma}
     * @param r0Interval the 95% profile-likelihood interval of {@code R0}
     */
    public record Fit(double beta, double gamma, double initial, double ascertainment, double nll, double r0,
            double[] r0Interval) {
    }

    private Epidemics() {
    }

    // ------------------------------------------------------------------ compartmental models

    /**
     * Daily new cases (E to I flow, or S to I for SIR when {@code sigma <= 0}) of a deterministic
     * SEIR model, integrated by RK4 with ten steps a day.
     *
     * @param population the population
     * @param beta transmission rate (per day)
     * @param sigma latent rate (per day); {@code <= 0} for SIR
     * @param gamma recovery rate (per day)
     * @param initial initially infectious individuals
     * @param days days
     * @return daily new cases
     */
    public static double[] seirIncidence(double population, double beta, double sigma, double gamma,
            double initial, int days) {
        boolean sir = sigma <= 0;
        double[] y = {1 - initial / population, 0, initial / population, 0};
        double[] out = new double[days];
        double h = 0.1;
        for (int d = 0; d < days; d++) {
            double cumBefore = y[2] + y[3];
            for (int k = 0; k < 10; k++) {
                y = rk4(y, h, beta, sigma, gamma, sir);
            }
            // cumulative cases: everyone who has become infectious, I + R
            out[d] = Math.max(0, (y[2] + y[3] - cumBefore) * population);
        }
        return out;
    }

    private static double[] deriv(double[] y, double beta, double sigma, double gamma, boolean sir) {
        double inf = beta * y[0] * y[2];
        if (sir) {
            return new double[] {-inf, 0, inf - gamma * y[2], gamma * y[2]};
        }
        return new double[] {-inf, inf - sigma * y[1], sigma * y[1] - gamma * y[2], gamma * y[2]};
    }

    private static double[] rk4(double[] y, double h, double beta, double sigma, double gamma, boolean sir) {
        double[] k1 = deriv(y, beta, sigma, gamma, sir);
        double[] k2 = deriv(axpy(y, k1, h / 2), beta, sigma, gamma, sir);
        double[] k3 = deriv(axpy(y, k2, h / 2), beta, sigma, gamma, sir);
        double[] k4 = deriv(axpy(y, k3, h), beta, sigma, gamma, sir);
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        }
        return out;
    }

    private static double[] axpy(double[] y, double[] k, double h) {
        double[] o = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            o[i] = y[i] + h * k[i];
        }
        return o;
    }

    /**
     * The negative log-likelihood of counts given expected counts.
     *
     * @param counts observed counts
     * @param mean expected counts
     * @param dispersion the NB dispersion, or {@code <= 0} for Poisson
     * @return the NLL (up to terms constant in {@code mean})
     */
    public static double nll(int[] counts, double[] mean, double dispersion) {
        double s = 0;
        for (int t = 0; t < counts.length; t++) {
            double m = Math.max(mean[t], 1e-9);
            if (dispersion <= 0) {
                s -= counts[t] * Math.log(m) - m;
            } else {
                double k = dispersion;
                s -= counts[t] * Math.log(m / (m + k)) + k * Math.log(k / (m + k));
            }
        }
        return s;
    }

    /**
     * Fits an SEIR model to daily case counts with the latent and recovery rates known (the
     * usual practice, since the counts alone barely separate {@code beta} from {@code gamma}):
     * {@code beta}, the initial infectious and the ascertainment fraction, on log/logit scales.
     *
     * @param cases daily reported cases
     * @param population the population
     * @param sigma the latent rate ({@code <= 0} for SIR)
     * @param gamma the recovery rate
     * @param dispersion NB dispersion ({@code <= 0} for Poisson)
     * @param withProfile whether to compute the R0 profile interval
     * @return the fit
     */
    public static Fit fitSeir(int[] cases, double population, double sigma, double gamma, double dispersion,
            boolean withProfile) {
        int days = cases.length;
        java.util.function.ToDoubleFunction<double[]> f = x -> {
            double[] mu = seirIncidence(population, Math.exp(x[0]), sigma, gamma, Math.exp(x[1]), days);
            double rho = 1 / (1 + Math.exp(-x[2]));
            for (int t = 0; t < days; t++) {
                mu[t] *= rho;
            }
            return nll(cases, mu, dispersion);
        };
        double[] best = null;
        double bestV = Double.POSITIVE_INFINITY;
        for (double r0 : new double[] {1.3, 2.0, 3.0}) {
            double[] x0 = {Math.log(r0 * gamma), Math.log(Math.max(1, cases[0] + 1)), 0.0};
            Optimize.Minimum m = Optimize.nelderMead(f, x0, 0.3, 1500, 1e-10);
            if (m.value() < bestV) {
                bestV = m.value();
                best = m.x();
            }
        }
        double[] x = best;
        double beta = Math.exp(x[0]);
        double[] interval = {Double.NaN, Double.NaN};
        if (withProfile) {
            double cut = bestV + Distributions.chiSquaredQuantile(0.95, 1) / 2;
            interval[0] = profileBound(f, x, cut, -1) / gamma;
            interval[1] = profileBound(f, x, cut, 1) / gamma;
        }
        return new Fit(beta, gamma, Math.exp(x[1]), 1 / (1 + Math.exp(-x[2])), bestV, beta / gamma, interval);
    }

    private static double profileBound(java.util.function.ToDoubleFunction<double[]> f, double[] x, double cut,
            int dir) {
        double logBeta = x[0];
        double step = 0.005;
        double[] rest = {x[1], x[2]};
        for (int k = 0; k < 80; k++) {
            double next = logBeta + dir * step;
            double fixed = next;
            Optimize.Minimum m = Optimize.nelderMead(y -> f.applyAsDouble(new double[] {fixed, y[0], y[1]}), rest,
                    0.05, 300, 1e-9);
            if (m.value() > cut) {
                return Math.exp(logBeta);
            }
            logBeta = next;
            rest = m.x();
            step *= 1.15;
        }
        return Math.exp(logBeta);
    }

    // ------------------------------------------------------------------ R_t

    /**
     * The total infectiousness {@code Lambda_t = sum_s w_s I_(t-s)}.
     *
     * @param incidence daily incidence
     * @param w serial-interval weights (index = lag)
     * @return {@code Lambda}
     */
    public static double[] infectiousness(int[] incidence, double[] w) {
        double[] l = new double[incidence.length];
        for (int t = 0; t < l.length; t++) {
            for (int s = 1; s < w.length && s <= t; s++) {
                l[t] += w[s] * incidence[t - s];
            }
        }
        return l;
    }

    /**
     * Estimates {@code R_t}.
     *
     * @param method the method
     * @param incidence daily incidence (by onset)
     * @param w serial-interval weights
     * @param window EpiEstim's window, or the random walk's step SD times 100 for the filters
     * @param seed the particle filter's seed
     * @return the estimate
     */
    public static RtEstimate rt(RtMethod method, int[] incidence, double[] w, int window, long seed) {
        return switch (method) {
            case EPIESTIM -> epiEstim(incidence, w, window, 1, 5);
            case RW_KALMAN -> randomWalkKalman(incidence, w, window / 100.0);
            case RW_PARTICLE -> randomWalkParticle(incidence, w, window / 100.0, 2000, seed);
        };
    }

    /**
     * Cori et al.'s estimator: over the window ending at {@code t}, the posterior of {@code R}
     * is {@code Gamma(a + sum I, 1/b + sum Lambda)}.
     *
     * @param incidence daily incidence
     * @param w serial-interval weights
     * @param window the window length in days
     * @param a the prior shape
     * @param b the prior scale
     * @return the estimate
     */
    public static RtEstimate epiEstim(int[] incidence, double[] w, int window, double a, double b) {
        double[] lam = infectiousness(incidence, w);
        int n = incidence.length;
        double[] mean = new double[n];
        double[] lo = new double[n];
        double[] hi = new double[n];
        Arrays.fill(mean, Double.NaN);
        Arrays.fill(lo, Double.NaN);
        Arrays.fill(hi, Double.NaN);
        for (int t = window; t < n; t++) {
            double si = 0;
            double sl = 0;
            for (int k = t - window + 1; k <= t; k++) {
                si += incidence[k];
                sl += lam[k];
            }
            double shape = a + si;
            double rate = 1 / b + sl;
            mean[t] = shape / rate;
            lo[t] = Distributions.chiSquaredQuantile(0.05, 2 * shape) / (2 * rate);
            hi[t] = Distributions.chiSquaredQuantile(0.95, 2 * shape) / (2 * rate);
        }
        return new RtEstimate(mean, lo, hi);
    }

    /**
     * A random walk on {@code x_t = log R_t} with step SD {@code q}, observed through the
     * linearisation {@code log((I_t + 0.5) / Lambda_t) ~ N(x_t, 1 / (I_t + 0.5))}, Kalman-filtered
     * and RTS-smoothed.
     *
     * @param incidence daily incidence
     * @param w serial-interval weights
     * @param q the random-walk step SD
     * @return the smoothed estimate
     */
    public static RtEstimate randomWalkKalman(int[] incidence, double[] w, double q) {
        double[] lam = infectiousness(incidence, w);
        int n = incidence.length;
        double[] mf = new double[n];
        double[] pf = new double[n];
        double m = 0;
        double p = 1;
        for (int t = 0; t < n; t++) {
            double pp = p + q * q;
            if (lam[t] > 0.5) {
                double obs = Math.log((incidence[t] + 0.5) / lam[t]);
                double r = 1 / (incidence[t] + 0.5);
                double k = pp / (pp + r);
                m = m + k * (obs - m);
                p = (1 - k) * pp;
            } else {
                p = pp;
            }
            mf[t] = m;
            pf[t] = p;
        }
        double[] ms = mf.clone();
        double[] ps = pf.clone();
        for (int t = n - 2; t >= 0; t--) {
            double pp = pf[t] + q * q;
            double g = pf[t] / pp;
            ms[t] = mf[t] + g * (ms[t + 1] - mf[t]);
            ps[t] = pf[t] + g * g * (ps[t + 1] - pp);
        }
        double[] mean = new double[n];
        double[] lo = new double[n];
        double[] hi = new double[n];
        for (int t = 0; t < n; t++) {
            double sd = Math.sqrt(Math.max(ps[t], 0));
            mean[t] = Math.exp(ms[t] + ps[t] / 2);
            lo[t] = Math.exp(ms[t] - 1.645 * sd);
            hi[t] = Math.exp(ms[t] + 1.645 * sd);
        }
        return new RtEstimate(mean, lo, hi);
    }

    /**
     * A bootstrap particle filter for the log-{@code R_t} random walk with the exact Poisson
     * renewal likelihood, smoothed by tracing the final particles' genealogies.
     *
     * @param incidence daily incidence
     * @param w serial-interval weights
     * @param q the random-walk step SD
     * @param particles the particle count
     * @param seed the seed
     * @return the smoothed estimate
     */
    public static RtEstimate randomWalkParticle(int[] incidence, double[] w, double q, int particles, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] lam = infectiousness(incidence, w);
        int n = incidence.length;
        double[][] x = new double[n][particles];
        int[][] parent = new int[n][particles];
        double[] cur = new double[particles];
        for (int i = 0; i < particles; i++) {
            cur[i] = Math.log(0.5 + 2.5 * r.nextDouble());
        }
        double[] lw = new double[particles];
        for (int t = 0; t < n; t++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < particles; i++) {
                cur[i] += q * r.nextGaussian();
                double mu = Math.exp(cur[i]) * lam[t];
                lw[i] = lam[t] > 0.5 ? incidence[t] * Math.log(Math.max(mu, 1e-12)) - mu : 0;
                max = Math.max(max, lw[i]);
            }
            double sum = 0;
            double[] wgt = new double[particles];
            for (int i = 0; i < particles; i++) {
                wgt[i] = Math.exp(lw[i] - max);
                sum += wgt[i];
            }
            // systematic resampling
            double u = r.nextDouble() / particles;
            double acc = wgt[0] / sum;
            int j = 0;
            double[] next = new double[particles];
            for (int i = 0; i < particles; i++) {
                double target = u + (double) i / particles;
                while (acc < target && j < particles - 1) {
                    j++;
                    acc += wgt[j] / sum;
                }
                next[i] = cur[j];
                parent[t][i] = j;
            }
            cur = next;
            x[t] = cur.clone();
        }
        // genealogy tracing
        double[] mean = new double[n];
        double[] lo = new double[n];
        double[] hi = new double[n];
        int[] idx = new int[particles];
        for (int i = 0; i < particles; i++) {
            idx[i] = i;
        }
        for (int t = n - 1; t >= 0; t--) {
            double[] v = new double[particles];
            for (int i = 0; i < particles; i++) {
                v[i] = Math.exp(x[t][idx[i]]);
            }
            Arrays.sort(v);
            double s = 0;
            for (double e : v) {
                s += e;
            }
            mean[t] = s / particles;
            lo[t] = v[(int) (0.05 * particles)];
            hi[t] = v[(int) (0.95 * particles) - 1];
            for (int i = 0; i < particles; i++) {
                idx[i] = parent[t][idx[i]];
            }
        }
        return new RtEstimate(mean, lo, hi);
    }

    // ------------------------------------------------------------------ nowcasting

    /**
     * Nowcasts counts by onset day from a reporting triangle by the chain-ladder method.
     *
     * @param triangle {@code [onset day][delay]} reported counts
     * @param now the current day (only {@code day + delay <= now} is known)
     * @param maxDelay the longest delay
     * @return the corrected counts for days {@code 0..now}
     */
    public static double[] nowcast(int[][] triangle, int now, int maxDelay) {
        double[] f = new double[maxDelay + 1];
        double total = 0;
        for (int t = 0; t + maxDelay <= now; t++) {
            for (int d = 0; d <= maxDelay; d++) {
                f[d] += triangle[t][d];
                total += triangle[t][d];
            }
        }
        double[] cdf = new double[maxDelay + 1];
        double acc = 0;
        for (int d = 0; d <= maxDelay; d++) {
            acc += f[d] / Math.max(total, 1);
            cdf[d] = acc;
        }
        double[] out = new double[now + 1];
        for (int t = 0; t <= now; t++) {
            int known = Math.min(maxDelay, now - t);
            double obs = 0;
            for (int d = 0; d <= known; d++) {
                obs += triangle[t][d];
            }
            out[t] = cdf[known] > 0 ? obs / cdf[known] : obs;
        }
        return out;
    }

    /**
     * Forecasts incidence by the renewal equation at a fixed {@code R}.
     *
     * @param history daily incidence so far
     * @param w serial-interval weights
     * @param r the reproduction number held fixed
     * @param horizon days ahead
     * @return the expected incidence over the horizon
     */
    public static double[] renewalForecast(double[] history, double[] w, double r, int horizon) {
        double[] all = Arrays.copyOf(history, history.length + horizon);
        for (int t = history.length; t < all.length; t++) {
            double l = 0;
            for (int s = 1; s < w.length && s <= t; s++) {
                l += w[s] * all[t - s];
            }
            all[t] = r * l;
        }
        return Arrays.copyOfRange(all, history.length, all.length);
    }
}
