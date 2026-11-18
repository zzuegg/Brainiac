package io.github.zzuegg.brainiac.experimental.bandit;

import java.util.SplittableRandom;

/**
 * Divergences, confidence bounds and samplers shared by the bandit algorithms: Bernoulli and
 * Gaussian Kullback-Leibler divergences, the KL upper and lower confidence bounds by bisection,
 * and Gamma/Beta sampling (Marsaglia-Tsang).
 */
public final class BanditMath {

    private BanditMath() {
    }

    /** The reward family whose divergence the KL-based rules use. */
    public enum Family {
        /** rewards in [0, 1], Bernoulli divergence */
        BERNOULLI,
        /** Gaussian rewards with unit variance (scale with {@link #klGaussian}) */
        GAUSSIAN
    }

    /**
     * @param p the first mean in [0, 1]
     * @param q the second mean in [0, 1]
     * @return {@code KL(Ber(p) || Ber(q))}
     */
    public static double klBernoulli(double p, double q) {
        double eps = 1e-12;
        p = Math.min(Math.max(p, eps), 1 - eps);
        q = Math.min(Math.max(q, eps), 1 - eps);
        return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
    }

    /**
     * @param p the first mean
     * @param q the second mean
     * @param sigma the common standard deviation
     * @return {@code KL(N(p, s^2) || N(q, s^2))}
     */
    public static double klGaussian(double p, double q, double sigma) {
        return (p - q) * (p - q) / (2 * sigma * sigma);
    }

    /**
     * @param family the family
     * @param sigma the Gaussian standard deviation (ignored for Bernoulli)
     * @param p the first mean
     * @param q the second mean
     * @return the divergence
     */
    public static double kl(Family family, double sigma, double p, double q) {
        return family == Family.BERNOULLI ? klBernoulli(p, q) : klGaussian(p, q, sigma);
    }

    /**
     * The KL-UCB index: the largest {@code q >= mean} with {@code n KL(mean, q) <= level}.
     *
     * @param family the family
     * @param sigma the Gaussian standard deviation
     * @param mean the empirical mean
     * @param n the pulls
     * @param level the exploration level, for example {@code log t}
     * @return the upper bound
     */
    public static double klUpper(Family family, double sigma, double mean, int n, double level) {
        if (family == Family.GAUSSIAN) {
            return mean + sigma * Math.sqrt(2 * Math.max(level, 0) / n);
        }
        double lo = Math.min(Math.max(mean, 0), 1);
        double hi = 1;
        for (int i = 0; i < 40; i++) {
            double mid = (lo + hi) / 2;
            if (n * klBernoulli(mean, mid) > level) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return lo;
    }

    /**
     * The KL lower confidence bound: the smallest {@code q <= mean} with
     * {@code n KL(mean, q) <= level}.
     *
     * @param family the family
     * @param sigma the Gaussian standard deviation
     * @param mean the empirical mean
     * @param n the pulls
     * @param level the level
     * @return the lower bound
     */
    public static double klLower(Family family, double sigma, double mean, int n, double level) {
        if (family == Family.GAUSSIAN) {
            return mean - sigma * Math.sqrt(2 * Math.max(level, 0) / n);
        }
        double lo = 0;
        double hi = Math.min(Math.max(mean, 0), 1);
        for (int i = 0; i < 40; i++) {
            double mid = (lo + hi) / 2;
            if (n * klBernoulli(mean, mid) > level) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return hi;
    }

    /**
     * @param shape the shape, positive
     * @param random the randomness
     * @return a Gamma(shape, 1) draw
     */
    public static double gamma(double shape, SplittableRandom random) {
        if (shape < 1) {
            return gamma(shape + 1, random) * Math.pow(random.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = random.nextGaussian();
            double v = 1 + c * x;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = random.nextDouble();
            if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
                return d * v;
            }
        }
    }

    /**
     * @param a the first shape
     * @param b the second shape
     * @param random the randomness
     * @return a Beta(a, b) draw
     */
    public static double beta(double a, double b, SplittableRandom random) {
        double x = gamma(a, random);
        double y = gamma(b, random);
        return x / (x + y);
    }

    static int argmax(double[] v, SplittableRandom tieBreak) {
        int best = 0;
        int ties = 1;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
                ties = 1;
            } else if (v[i] == v[best] && tieBreak != null && tieBreak.nextInt(++ties) == 0) {
                best = i;
            }
        }
        return best;
    }
}
