package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;
import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Generalised linear models for counts and zero-heavy amounts (Nelder and Wedderburn, JRSS-A
 * 1972), all with a log link {@code log mu = log(exposure) + b0 + x b} (the exposure enters as
 * an offset), fitted by iteratively reweighted least squares with an optional ridge.
 *
 * <ul>
 *   <li>{@link Family#POISSON}: {@code var = mu}.</li>
 *   <li>{@link Family#NEGATIVE_BINOMIAL}: NB2, {@code var = mu + mu^2 / k}, alternating IRLS
 *       for the mean with a moment estimate of {@code k}.</li>
 *   <li>{@link Family#ZERO_INFLATED_POISSON} (Lambert, Technometrics 1992): a structural zero
 *       with logistic probability {@code pi(x)} mixed with a Poisson count, fitted by EM.</li>
 *   <li>{@link Family#TWEEDIE} (Jorgensen 1987): the compound Poisson-gamma family with power
 *       {@code 1 < p < 2} ({@code var = phi mu^p}), which has exact zeros plus positive
 *       continuous amounts (insurance claims); {@code phi} from the Pearson statistic.</li>
 * </ul>
 */
public final class CountGlm {

    /** The family. */
    public enum Family {
        /** Poisson. */
        POISSON,
        /** Negative binomial (NB2). */
        NEGATIVE_BINOMIAL,
        /** Zero-inflated Poisson. */
        ZERO_INFLATED_POISSON,
        /** Tweedie with power 1.5. */
        TWEEDIE
    }

    /** The Tweedie power used by {@link Family#TWEEDIE}. */
    public static final double TWEEDIE_POWER = 1.5;

    private final Family family;
    private final double[] beta;
    private final double[] zeroBeta;
    private final double dispersion;

    private CountGlm(Family family, double[] beta, double[] zeroBeta, double dispersion) {
        this.family = family;
        this.beta = beta;
        this.zeroBeta = zeroBeta;
        this.dispersion = dispersion;
    }

    /**
     * Fits a count GLM.
     *
     * @param family the family
     * @param x the design {@code [n][p]}
     * @param exposure exposures (offset {@code log exposure}), or {@code null} for ones
     * @param y the outcomes
     * @param ridge a ridge on the slopes
     * @return the model
     */
    public static CountGlm fit(Family family, double[][] x, double[] exposure, double[] y, double ridge) {
        int n = y.length;
        double[] offset = new double[n];
        for (int i = 0; i < n; i++) {
            offset[i] = exposure == null ? 0 : Math.log(exposure[i]);
        }
        double[] ones = new double[n];
        Arrays.fill(ones, 1);
        return switch (family) {
            case POISSON -> new CountGlm(family, logLink(x, y, offset, ones, ridge, 0, Double.POSITIVE_INFINITY),
                    null, 1);
            case TWEEDIE -> {
                double[] b = logLink(x, y, offset, ones, ridge, TWEEDIE_POWER, Double.POSITIVE_INFINITY);
                double pearson = 0;
                for (int i = 0; i < n; i++) {
                    double mu = Math.exp(offset[i] + eta(b, x[i]));
                    pearson += (y[i] - mu) * (y[i] - mu) / Math.pow(mu, TWEEDIE_POWER);
                }
                yield new CountGlm(family, b, null, pearson / Math.max(1, n - b.length));
            }
            case NEGATIVE_BINOMIAL -> {
                double k = 10;
                double[] b = logLink(x, y, offset, ones, ridge, 0, Double.POSITIVE_INFINITY);
                for (int it = 0; it < 20; it++) {
                    double num = 0;
                    double den = 0;
                    for (int i = 0; i < n; i++) {
                        double mu = Math.exp(offset[i] + eta(b, x[i]));
                        num += mu * mu;
                        den += (y[i] - mu) * (y[i] - mu) - mu;
                    }
                    double nk = den <= 0 ? 1e6 : Math.min(1e6, Math.max(0.05, num / den));
                    b = logLink(x, y, offset, ones, ridge, 0, nk);
                    if (Math.abs(Math.log(nk / k)) < 1e-4) {
                        k = nk;
                        break;
                    }
                    k = nk;
                }
                yield new CountGlm(family, b, null, k);
            }
            case ZERO_INFLATED_POISSON -> zip(x, y, offset, ridge);
        };
    }

    private static CountGlm zip(double[][] x, double[] y, double[] offset, double ridge) {
        int n = y.length;
        int p = x[0].length;
        double[] w = new double[n];
        Arrays.fill(w, 1);
        double[] b = logLink(x, y, offset, w, ridge, 0, Double.POSITIVE_INFINITY);
        double[] g = new double[p + 1];
        double[] zeta = new double[n];
        for (int it = 0; it < 100; it++) {
            for (int i = 0; i < n; i++) {
                if (y[i] > 0) {
                    zeta[i] = 0;
                } else {
                    double pi = sigmoid(eta(g, x[i]));
                    double mu = Math.exp(offset[i] + eta(b, x[i]));
                    zeta[i] = pi / (pi + (1 - pi) * Math.exp(-mu));
                }
                w[i] = 1 - zeta[i];
            }
            double[] nb = logLink(x, y, offset, w, ridge, 0, Double.POSITIVE_INFINITY);
            double[] ng = logistic(x, zeta, g, ridge + 1e-4);
            double change = 0;
            for (int j = 0; j <= p; j++) {
                change = Math.max(change, Math.max(Math.abs(nb[j] - b[j]), Math.abs(ng[j] - g[j])));
            }
            b = nb;
            g = ng;
            if (change < 1e-6) {
                break;
            }
        }
        return new CountGlm(Family.ZERO_INFLATED_POISSON, b, g, 1);
    }

    /**
     * IRLS for a log-link GLM with variance {@code mu^power} (Poisson/Tweedie) or
     * {@code mu + mu^2 / k} (negative binomial when {@code k} is finite), with prior weights.
     */
    private static double[] logLink(double[][] x, double[] y, double[] offset, double[] prior, double ridge,
            double power, double k) {
        int n = y.length;
        int p = x[0].length;
        double[] b = new double[p + 1];
        double my = 0;
        double ws = 0;
        for (int i = 0; i < n; i++) {
            my += prior[i] * y[i] / Math.exp(offset[i]);
            ws += prior[i];
        }
        b[0] = Math.log(Math.max(1e-3, my / ws));
        for (int it = 0; it < 100; it++) {
            double[][] gram = new double[p + 1][p + 1];
            double[] rhs = new double[p + 1];
            double[] z = new double[p + 1];
            for (int i = 0; i < n; i++) {
                double e = eta(b, x[i]);
                double mu = Math.exp(Math.min(30, offset[i] + e));
                double var = Double.isInfinite(k) ? Math.pow(mu, power == 0 ? 1 : power) : mu + mu * mu / k;
                double wi = prior[i] * mu * mu / Math.max(var, 1e-12);
                double zi = e + (y[i] - mu) / mu;
                z[0] = 1;
                System.arraycopy(x[i], 0, z, 1, p);
                for (int a = 0; a <= p; a++) {
                    rhs[a] += wi * z[a] * zi;
                    for (int c = 0; c <= p; c++) {
                        gram[a][c] += wi * z[a] * z[c];
                    }
                }
            }
            for (int a = 1; a <= p; a++) {
                gram[a][a] += ridge + 1e-9;
            }
            gram[0][0] += 1e-9;
            double[] nb = Recalibrators.solve(gram, rhs);
            double change = 0;
            for (int a = 0; a <= p; a++) {
                change = Math.max(change, Math.abs(nb[a] - b[a]));
            }
            b = nb;
            if (change < 1e-8) {
                break;
            }
        }
        return b;
    }

    private static double[] logistic(double[][] x, double[] target, double[] start, double ridge) {
        int n = target.length;
        int p = x[0].length;
        double[] g = start.clone();
        for (int it = 0; it < 25; it++) {
            double[][] gram = new double[p + 1][p + 1];
            double[] rhs = new double[p + 1];
            double[] z = new double[p + 1];
            for (int i = 0; i < n; i++) {
                double e = eta(g, x[i]);
                double pr = sigmoid(e);
                double wi = Math.max(pr * (1 - pr), 1e-6);
                double zi = e + (target[i] - pr) / wi;
                z[0] = 1;
                System.arraycopy(x[i], 0, z, 1, p);
                for (int a = 0; a <= p; a++) {
                    rhs[a] += wi * z[a] * zi;
                    for (int c = 0; c <= p; c++) {
                        gram[a][c] += wi * z[a] * z[c];
                    }
                }
            }
            for (int a = 0; a <= p; a++) {
                gram[a][a] += ridge;
            }
            double[] ng = Recalibrators.solve(gram, rhs);
            double change = 0;
            for (int a = 0; a <= p; a++) {
                change = Math.max(change, Math.abs(ng[a] - g[a]));
            }
            g = ng;
            if (change < 1e-7) {
                break;
            }
        }
        return g;
    }

    private static double eta(double[] b, double[] row) {
        double s = b[0];
        for (int j = 0; j < row.length; j++) {
            s += b[j + 1] * row[j];
        }
        return s;
    }

    private static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    /**
     * The mean of the count part (before zero inflation).
     *
     * @param row the features
     * @param exposure the exposure
     * @return {@code mu}
     */
    public double countMean(double[] row, double exposure) {
        return exposure * Math.exp(eta(beta, row));
    }

    /**
     * The mean outcome.
     *
     * @param row the features
     * @param exposure the exposure
     * @return {@code E[y]}
     */
    public double mean(double[] row, double exposure) {
        double mu = countMean(row, exposure);
        return family == Family.ZERO_INFLATED_POISSON ? (1 - sigmoid(eta(zeroBeta, row))) * mu : mu;
    }

    /**
     * The predicted probability of an exact zero.
     *
     * @param row the features
     * @param exposure the exposure
     * @return {@code P(y = 0)}
     */
    public double zeroProbability(double[] row, double exposure) {
        double mu = countMean(row, exposure);
        return switch (family) {
            case POISSON -> Math.exp(-mu);
            case NEGATIVE_BINOMIAL -> Math.pow(dispersion / (dispersion + mu), dispersion);
            case ZERO_INFLATED_POISSON -> {
                double pi = sigmoid(eta(zeroBeta, row));
                yield pi + (1 - pi) * Math.exp(-mu);
            }
            case TWEEDIE -> Math.exp(-Math.pow(mu, 2 - TWEEDIE_POWER) / (dispersion * (2 - TWEEDIE_POWER)));
        };
    }

    /**
     * Draws from the fitted predictive distribution.
     *
     * @param row the features
     * @param exposure the exposure
     * @param r the random source
     * @return a draw
     */
    public double sample(double[] row, double exposure, SplittableRandom r) {
        double mu = countMean(row, exposure);
        return switch (family) {
            case POISSON -> SparseDesigns.poisson(mu, r);
            case NEGATIVE_BINOMIAL -> SparseDesigns.poisson(mu * SparseDesigns.gamma(dispersion, r) / dispersion, r);
            case ZERO_INFLATED_POISSON -> r.nextDouble() < sigmoid(eta(zeroBeta, row)) ? 0
                    : SparseDesigns.poisson(mu, r);
            case TWEEDIE -> SparseDesigns.tweedie(mu, dispersion, TWEEDIE_POWER, r);
        };
    }

    /**
     * A central predictive interval by Monte Carlo.
     *
     * @param row the features
     * @param exposure the exposure
     * @param level the coverage, for example 0.9
     * @param draws Monte Carlo draws
     * @param r the random source
     * @return {@code {lower, upper}}
     */
    public double[] interval(double[] row, double exposure, double level, int draws, SplittableRandom r) {
        double[] d = new double[draws];
        for (int k = 0; k < draws; k++) {
            d[k] = sample(row, exposure, r);
        }
        Arrays.sort(d);
        double a = (1 - level) / 2;
        return new double[] {d[(int) Math.floor(a * (draws - 1))], d[(int) Math.ceil((1 - a) * (draws - 1))]};
    }

    /**
     * The fitted dispersion: {@code k} for the negative binomial, {@code phi} for the Tweedie, 1
     * otherwise.
     *
     * @return the dispersion
     */
    public double dispersion() {
        return dispersion;
    }

    /**
     * The count-part coefficients, intercept first.
     *
     * @return a copy
     */
    public double[] coefficients() {
        return beta.clone();
    }

    /**
     * The mean Poisson deviance.
     *
     * @param y outcomes
     * @param mu predicted means
     * @return {@code 2 mean(y log(y / mu) - (y - mu))}
     */
    public static double poissonDeviance(double[] y, double[] mu) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double m = Math.max(mu[i], 1e-9);
            s += 2 * ((y[i] > 0 ? y[i] * Math.log(y[i] / m) : 0) - (y[i] - m));
        }
        return s / y.length;
    }

    /**
     * The mean Tweedie deviance with power {@code p} in {@code (1, 2)}.
     *
     * @param y outcomes
     * @param mu predicted means
     * @param p the power
     * @return the deviance
     */
    public static double tweedieDeviance(double[] y, double[] mu, double p) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double m = Math.max(mu[i], 1e-9);
            s += 2 * (Math.pow(y[i], 2 - p) / ((1 - p) * (2 - p)) - y[i] * Math.pow(m, 1 - p) / (1 - p)
                    + Math.pow(m, 2 - p) / (2 - p));
        }
        return s / y.length;
    }
}
