package io.github.zzuegg.brainiac.experimental.ml;

/**
 * The standard normal distribution's tails and quantile function in double precision, for
 * privacy accounting (log tail probabilities far out in the tail) and certification
 * ({@code Φ⁻¹} of a confidence bound): {@link #logSf} stays finite where {@code 1 - Φ(x)}
 * underflows, and {@link #quantile} refines Acklam's rational approximation by Newton steps.
 */
public final class NormalTails {

    private NormalTails() {
    }

    /**
     * The complementary error function, {@code erfc(z)}, by the Chebyshev fit of Numerical
     * Recipes (relative error below {@code 1.2e-7}).
     *
     * @param z the argument
     * @return {@code erfc(z)}
     */
    public static double erfc(double z) {
        return Math.exp(logErfc(z));
    }

    /**
     * {@code log erfc(z)} for {@code z >= 0}; for negative {@code z} the log of
     * {@code 2 - erfc(-z)}.
     *
     * @param z the argument
     * @return the log
     */
    public static double logErfc(double z) {
        if (z < 0) {
            return Math.log(2 - Math.exp(logErfc(-z)));
        }
        double t = 1 / (1 + 0.5 * z);
        double poly = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
                + t * (-0.82215223 + t * 0.17087277))))))));
        return Math.log(t) + poly;
    }

    /**
     * @param x the argument
     * @return {@code Φ(x)}
     */
    public static double cdf(double x) {
        return 0.5 * erfc(-x / Math.sqrt(2));
    }

    /**
     * @param x the argument
     * @return {@code log(1 - Φ(x))}, accurate far into the upper tail
     */
    public static double logSf(double x) {
        return Math.log(0.5) + logErfc(x / Math.sqrt(2));
    }

    /**
     * @param x the argument
     * @return the standard normal log density
     */
    public static double logPdf(double x) {
        return -0.5 * x * x - 0.5 * Math.log(2 * Math.PI);
    }

    /**
     * The quantile function {@code Φ⁻¹(p)}: Acklam's approximation refined by two Newton
     * steps.
     *
     * @param p the probability in {@code (0, 1)}
     * @return the quantile
     */
    public static double quantile(double p) {
        if (!(p > 0 && p < 1)) {
            if (p == 0) {
                return Double.NEGATIVE_INFINITY;
            }
            if (p == 1) {
                return Double.POSITIVE_INFINITY;
            }
            throw new IllegalArgumentException("p must be in [0, 1]: " + p);
        }
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
        double x;
        if (p < 0.02425) {
            double q = Math.sqrt(-2 * Math.log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        } else if (p <= 1 - 0.02425) {
            double q = p - 0.5;
            double r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        } else {
            double q = Math.sqrt(-2 * Math.log(1 - p));
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        for (int i = 0; i < 2; i++) {
            double e = cdf(x) - p;
            x -= e / Math.exp(logPdf(x));
        }
        return x;
    }
}
