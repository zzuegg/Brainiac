package io.github.zzuegg.brainiac.tensor;

import java.util.function.DoubleUnaryOperator;
import java.util.random.RandomGenerator;

/**
 * Distribution functions for tests and intervals: Student's t, chi-squared, Fisher's F and
 * Weibull, as SciPy's {@code cdf}, {@code sf} (survival, {@code 1 - cdf}) and {@code ppf}
 * (quantile). The standard normal is in {@link Special#erfc} and {@link Special#normalQuantile}.
 */
public final class Distributions {

    private Distributions() {
    }

    // ------------------------------------------------------------------ Student's t

    /**
     * @param t the value
     * @param df the degrees of freedom, positive
     * @return {@code P(T <= t)}
     */
    public static double studentTCdf(double t, double df) {
        positive(df, "degrees of freedom");
        double tail = 0.5 * Special.betaI(df / 2, 0.5, df / (df + t * t));
        return t >= 0 ? 1 - tail : tail;
    }

    /**
     * The two-sided p-value of a t statistic: {@code P(|T| >= |t|)}.
     *
     * @param t the statistic
     * @param df the degrees of freedom
     * @return the p-value
     */
    public static double studentTTwoSided(double t, double df) {
        positive(df, "degrees of freedom");
        return Special.betaI(df / 2, 0.5, df / (df + t * t));
    }

    /**
     * @param p a probability in (0, 1)
     * @param df the degrees of freedom
     * @return the {@code t} with {@code P(T <= t) = p}
     */
    public static double studentTQuantile(double p, double df) {
        probability(p);
        if (p == 0.5) {
            return 0;
        }
        return invert(t -> studentTCdf(t, df), p, Special.normalQuantile(p), Double.NEGATIVE_INFINITY);
    }

    // ------------------------------------------------------------------ chi-squared

    /**
     * @param x the value
     * @param k the degrees of freedom, positive
     * @return {@code P(X <= x)}
     */
    public static double chiSquaredCdf(double x, double k) {
        positive(k, "degrees of freedom");
        return x <= 0 ? 0 : Special.gammaP(k / 2, x / 2);
    }

    /**
     * @param x the value
     * @param k the degrees of freedom
     * @return {@code P(X >= x)}, accurate where it is tiny
     */
    public static double chiSquaredSurvival(double x, double k) {
        positive(k, "degrees of freedom");
        return x <= 0 ? 1 : Special.gammaQ(k / 2, x / 2);
    }

    /**
     * @param p a probability in (0, 1)
     * @param k the degrees of freedom
     * @return the {@code x} with {@code P(X <= x) = p}
     */
    public static double chiSquaredQuantile(double p, double k) {
        probability(p);
        return invert(x -> chiSquaredCdf(x, k), p, k, 0);
    }

    // ------------------------------------------------------------------ F

    /**
     * @param x the value
     * @param d1 the numerator's degrees of freedom
     * @param d2 the denominator's degrees of freedom
     * @return {@code P(F <= x)}
     */
    public static double fCdf(double x, double d1, double d2) {
        positive(d1, "degrees of freedom");
        positive(d2, "degrees of freedom");
        return x <= 0 ? 0 : Special.betaI(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2));
    }

    /**
     * @param x the value
     * @param d1 the numerator's degrees of freedom
     * @param d2 the denominator's degrees of freedom
     * @return {@code P(F >= x)}, as for an ANOVA or regression F test
     */
    public static double fSurvival(double x, double d1, double d2) {
        positive(d1, "degrees of freedom");
        positive(d2, "degrees of freedom");
        return x <= 0 ? 1 : Special.betaI(d2 / 2, d1 / 2, d2 / (d1 * x + d2));
    }

    /**
     * @param p a probability in (0, 1)
     * @param d1 the numerator's degrees of freedom
     * @param d2 the denominator's degrees of freedom
     * @return the {@code x} with {@code P(F <= x) = p}
     */
    public static double fQuantile(double p, double d1, double d2) {
        probability(p);
        return invert(x -> fCdf(x, d1, d2), p, 1, 0);
    }

    // ------------------------------------------------------------------ Weibull

    /**
     * @param x the value
     * @param shape {@code k}, positive
     * @param scale {@code lambda}, positive
     * @return the density {@code (k / lambda) (x / lambda)^(k - 1) exp(-(x / lambda)^k)}
     */
    public static double weibullPdf(double x, double shape, double scale) {
        positive(shape, "shape");
        positive(scale, "scale");
        if (x < 0) {
            return 0;
        }
        double z = x / scale;
        return shape / scale * Math.pow(z, shape - 1) * Math.exp(-Math.pow(z, shape));
    }

    /**
     * @param x the value
     * @param shape {@code k}
     * @param scale {@code lambda}
     * @return {@code 1 - exp(-(x / lambda)^k)}
     */
    public static double weibullCdf(double x, double shape, double scale) {
        positive(shape, "shape");
        positive(scale, "scale");
        return x <= 0 ? 0 : -Math.expm1(-Math.pow(x / scale, shape));
    }

    /**
     * @param p a probability in [0, 1)
     * @param shape {@code k}
     * @param scale {@code lambda}
     * @return {@code lambda (-ln(1 - p))^(1 / k)}
     */
    public static double weibullQuantile(double p, double shape, double scale) {
        positive(shape, "shape");
        positive(scale, "scale");
        if (!(p >= 0 && p < 1)) {
            throw new IllegalArgumentException("a probability in [0, 1), not " + p);
        }
        return scale * Math.pow(-Math.log1p(-p), 1 / shape);
    }

    /**
     * Draws from a Weibull distribution.
     *
     * @param shape {@code k}
     * @param scale {@code lambda}
     * @param random the random source
     * @return a sample
     */
    public static double weibullSample(double shape, double scale, RandomGenerator random) {
        return weibullQuantile(random.nextDouble(), shape, scale);
    }

    // ------------------------------------------------------------------ helpers

    // the x with cdf(x) = p for an increasing cdf: bracket from a guess, then bisect
    private static double invert(DoubleUnaryOperator cdf, double p, double guess, double lowest) {
        double lo = guess;
        double hi = guess;
        double step = Math.max(1, Math.abs(guess));
        while (cdf.applyAsDouble(lo) > p) {
            lo = Math.max(lo - step, lowest == 0 ? lo / 2 : Double.NEGATIVE_INFINITY);
            step *= 2;
            if (lowest == 0 && lo < 1e-300) {
                return 0;
            }
        }
        step = Math.max(1, Math.abs(guess));
        while (cdf.applyAsDouble(hi) < p) {
            hi += step;
            step *= 2;
        }
        for (int i = 0; i < 200 && hi - lo > 1e-14 * Math.max(1, Math.abs(hi)); i++) {
            double mid = 0.5 * (lo + hi);
            if (cdf.applyAsDouble(mid) < p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }

    private static void positive(double v, String what) {
        if (!(v > 0)) {
            throw new IllegalArgumentException("the " + what + " must be positive, not " + v);
        }
    }

    private static void probability(double p) {
        if (!(p > 0 && p < 1)) {
            throw new IllegalArgumentException("a probability in (0, 1), not " + p);
        }
    }
}
