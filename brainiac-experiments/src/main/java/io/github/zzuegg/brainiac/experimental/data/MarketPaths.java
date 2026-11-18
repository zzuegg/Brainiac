package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * Simulated markets with known dynamics: geometric Brownian motion, Heston stochastic
 * volatility, GARCH / GJR-GARCH returns with Student-t innovations, regime-switching returns
 * with jumps (each keeping the true conditional variance), multi-asset price relatives with no
 * edge, mean reversion or momentum, and private values for auctions.
 */
public final class MarketPaths {

    private MarketPaths() {
    }

    /**
     * Price paths and, for stochastic-volatility models, the variance paths.
     *
     * @param prices {@code [path][step]}, {@code steps + 1} columns
     * @param variances {@code [path][step]} instantaneous variance, or {@code null}
     */
    public record Paths(double[][] prices, double[][] variances) {
    }

    /**
     * Geometric Brownian motion, exact discretisation.
     *
     * @param paths paths
     * @param steps steps
     * @param s0 the initial price
     * @param mu the drift
     * @param sigma the volatility
     * @param dt the step in years
     * @param rnd the random source
     * @return the paths
     */
    public static Paths gbm(int paths, int steps, double s0, double mu, double sigma, double dt,
            SplittableRandom rnd) {
        double[][] s = new double[paths][steps + 1];
        for (int p = 0; p < paths; p++) {
            s[p][0] = s0;
            for (int t = 1; t <= steps; t++) {
                s[p][t] = s[p][t - 1] * Math.exp((mu - 0.5 * sigma * sigma) * dt
                        + sigma * Math.sqrt(dt) * ChannelCodes.gaussian(rnd));
            }
        }
        return new Paths(s, null);
    }

    /**
     * Heston stochastic volatility by full-truncation Euler.
     *
     * @param paths paths
     * @param steps steps
     * @param s0 the initial price
     * @param v0 the initial variance
     * @param kappa the mean-reversion speed
     * @param theta the long-run variance
     * @param xi the volatility of variance
     * @param rho the price-variance correlation
     * @param dt the step in years
     * @param rnd the random source
     * @return prices and variances
     */
    public static Paths heston(int paths, int steps, double s0, double v0, double kappa, double theta, double xi,
            double rho, double dt, SplittableRandom rnd) {
        double[][] s = new double[paths][steps + 1];
        double[][] v = new double[paths][steps + 1];
        for (int p = 0; p < paths; p++) {
            s[p][0] = s0;
            v[p][0] = v0;
            for (int t = 1; t <= steps; t++) {
                double z1 = ChannelCodes.gaussian(rnd);
                double z2 = rho * z1 + Math.sqrt(1 - rho * rho) * ChannelCodes.gaussian(rnd);
                double vp = Math.max(v[p][t - 1], 0);
                s[p][t] = s[p][t - 1] * Math.exp(-0.5 * vp * dt + Math.sqrt(vp * dt) * z1);
                v[p][t] = Math.max(v[p][t - 1] + kappa * (theta - vp) * dt + xi * Math.sqrt(vp * dt) * z2, 0);
            }
        }
        return new Paths(s, v);
    }

    /**
     * A return series with its true conditional variance.
     *
     * @param returns the returns
     * @param variance the conditional variance of each return given the past
     */
    public record Returns(double[] returns, double[] variance) {
    }

    /**
     * GJR-GARCH(1,1) returns {@code r_t = sigma_t z_t},
     * {@code sigma^2_t = omega + (alpha + gamma 1[r_(t-1) < 0]) r^2_(t-1) + beta sigma^2_(t-1)},
     * with unit-variance Student-t innovations ({@code nu = infinity} for Gaussian); plain GARCH
     * at {@code gamma = 0}.
     *
     * @param days the length
     * @param omega the constant
     * @param alpha the ARCH coefficient
     * @param gamma the leverage coefficient
     * @param beta the GARCH coefficient
     * @param nu the degrees of freedom
     * @param rnd the random source
     * @return returns and variances
     */
    public static Returns garch(int days, double omega, double alpha, double gamma, double beta, double nu,
            SplittableRandom rnd) {
        double[] r = new double[days];
        double[] v = new double[days];
        double var = omega / Math.max(1 - alpha - gamma / 2 - beta, 1e-3);
        double prev = 0;
        for (int burn = 0; burn < 500 + days; burn++) {
            var = burn == 0 ? var : omega + (alpha + (prev < 0 ? gamma : 0)) * prev * prev + beta * var;
            double ret = Math.sqrt(var) * studentT(nu, rnd);
            if (burn >= 500) {
                r[burn - 500] = ret;
                v[burn - 500] = var;
            }
            prev = ret;
        }
        return new Returns(r, v);
    }

    /**
     * Daily log returns of a Heston process (variance {@code v_t dt} known).
     *
     * @param days the length
     * @param rnd the random source
     * @return returns and conditional variances
     */
    public static Returns hestonReturns(int days, SplittableRandom rnd) {
        Paths p = heston(1, days, 1, 0.04, 3, 0.04, 0.6, -0.7, 1 / 252.0, rnd);
        double[] r = new double[days];
        double[] v = new double[days];
        for (int t = 0; t < days; t++) {
            r[t] = Math.log(p.prices()[0][t + 1] / p.prices()[0][t]);
            v[t] = Math.max(p.variances()[0][t], 0) / 252.0;
        }
        return new Returns(r, v);
    }

    /**
     * Two-regime Markov-switching returns (calm 1% and turbulent 3% daily vol, switching
     * probabilities 0.01 and 0.04) with Poisson jumps (rate 0.01, size N(0, 5%)). The variance
     * kept is the regime's diffusive variance plus the expected jump variance.
     *
     * @param days the length
     * @param rnd the random source
     * @return returns and conditional variances given the regime
     */
    public static Returns regimeSwitching(int days, SplittableRandom rnd) {
        double[] r = new double[days];
        double[] v = new double[days];
        int regime = 0;
        double[] vol = {0.01, 0.03};
        double[] leave = {0.01, 0.04};
        for (int t = 0; t < days; t++) {
            if (rnd.nextDouble() < leave[regime]) {
                regime = 1 - regime;
            }
            double jump = rnd.nextDouble() < 0.01 ? 0.05 * ChannelCodes.gaussian(rnd) : 0;
            r[t] = vol[regime] * ChannelCodes.gaussian(rnd) + jump;
            v[t] = vol[regime] * vol[regime] + 0.01 * 0.05 * 0.05;
        }
        return new Returns(r, v);
    }

    /**
     * A unit-variance Student-t draw (Gaussian when {@code nu} is infinite).
     *
     * @param nu degrees of freedom, above 2
     * @param rnd the random source
     * @return the draw
     */
    public static double studentT(double nu, SplittableRandom rnd) {
        double z = ChannelCodes.gaussian(rnd);
        if (Double.isInfinite(nu)) {
            return z;
        }
        double chi = 0;
        int whole = (int) Math.floor(nu);
        for (int i = 0; i < whole; i++) {
            double g = ChannelCodes.gaussian(rnd);
            chi += g * g;
        }
        return z / Math.sqrt(chi / whole) * Math.sqrt((whole - 2.0) / whole);
    }

    /** The kind of multi-asset market. */
    public enum Regime {
        /** independent GBMs with equal drift: no edge for any strategy. */
        NO_EDGE,
        /** log prices revert to a common slowly moving mean (OLMAR's regime). */
        MEAN_REVERTING,
        /** returns are positively autocorrelated (trends persist). */
        MOMENTUM
    }

    /**
     * Price relatives {@code x_t[i] = p_t[i] / p_(t-1)[i]} of {@code assets} assets.
     *
     * @param periods periods
     * @param assets assets
     * @param regime the dynamics
     * @param rnd the random source
     * @return {@code [period][asset]}
     */
    public static double[][] priceRelatives(int periods, int assets, Regime regime, SplittableRandom rnd) {
        double[][] x = new double[periods][assets];
        double[] logP = new double[assets];
        double[] lastRet = new double[assets];
        double[] vol = new double[assets];
        for (int i = 0; i < assets; i++) {
            vol[i] = 0.01 + 0.02 * rnd.nextDouble();
        }
        for (int t = 0; t < periods; t++) {
            for (int i = 0; i < assets; i++) {
                double ret = switch (regime) {
                    case NO_EDGE -> 0.0002 + vol[i] * ChannelCodes.gaussian(rnd);
                    case MEAN_REVERTING -> -0.3 * logP[i] + vol[i] * ChannelCodes.gaussian(rnd);
                    case MOMENTUM -> 0.4 * lastRet[i] + vol[i] * ChannelCodes.gaussian(rnd);
                };
                logP[i] += ret;
                lastRet[i] = ret;
                x[t][i] = Math.exp(ret);
            }
        }
        return x;
    }

    /**
     * Independent uniform private values on {@code [0, 1]}.
     *
     * @param samples auctions
     * @param bidders bidders
     * @param items items
     * @param rnd the random source
     * @return {@code [sample][bidder][item]}
     */
    public static double[][][] privateValues(int samples, int bidders, int items, SplittableRandom rnd) {
        double[][][] v = new double[samples][bidders][items];
        for (double[][] s : v) {
            for (double[] b : s) {
                for (int j = 0; j < items; j++) {
                    b[j] = rnd.nextDouble();
                }
            }
        }
        return v;
    }
}
