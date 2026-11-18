package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

/**
 * Tail probabilities of the reference distributions that classical tests need: Student's t, the
 * F distribution and the chi-square distribution, through the regularised incomplete beta
 * function (continued fraction of Numerical Recipes, section 6.4) and the regularised incomplete
 * gamma function of {@link KernelTests#regularizedGammaP}.
 */
public final class Distributions {

    private Distributions() {
    }

    /**
     * The natural log of the gamma function (Lanczos approximation).
     *
     * @param x a positive argument
     * @return {@code log Gamma(x)}
     */
    public static double logGamma(double x) {
        double[] c = {76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
            0.1208650973866179e-2, -0.5395239384953e-5};
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015;
        for (double v : c) {
            ser += v / ++y;
        }
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    /**
     * The regularised incomplete beta function {@code I_x(a, b)}.
     *
     * @param x the argument in {@code [0, 1]}
     * @param a the first shape
     * @param b the second shape
     * @return {@code I_x(a, b)}
     */
    public static double incompleteBeta(double x, double a, double b) {
        if (x <= 0) {
            return 0;
        }
        if (x >= 1) {
            return 1;
        }
        double front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x)
                + b * Math.log(1 - x));
        if (x < (a + 1) / (a + b + 2)) {
            return front * betaFraction(x, a, b) / a;
        }
        return 1 - front * betaFraction(1 - x, b, a) / b;
    }

    private static double betaFraction(double x, double a, double b) {
        double tiny = 1e-300;
        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        d = Math.abs(d) < tiny ? tiny : d;
        d = 1 / d;
        double h = d;
        for (int m = 1; m <= 300; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            d = Math.abs(d) < tiny ? tiny : d;
            c = 1 + aa / c;
            c = Math.abs(c) < tiny ? tiny : c;
            d = 1 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            d = Math.abs(d) < tiny ? tiny : d;
            c = 1 + aa / c;
            c = Math.abs(c) < tiny ? tiny : c;
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) < 3e-14) {
                break;
            }
        }
        return h;
    }

    /**
     * The two-sided p-value of a Student-t statistic.
     *
     * @param t the statistic
     * @param df degrees of freedom
     * @return {@code P(|T| >= |t|)}
     */
    public static double studentTwoSided(double t, double df) {
        if (df > 1e6) {
            return 2 * (1 - NormalTails.cdf(Math.abs(t)));
        }
        return incompleteBeta(df / (df + t * t), df / 2, 0.5);
    }

    /**
     * The upper tail of the F distribution.
     *
     * @param f the statistic
     * @param d1 numerator degrees of freedom
     * @param d2 denominator degrees of freedom
     * @return {@code P(F >= f)}
     */
    public static double fUpper(double f, double d1, double d2) {
        if (f <= 0) {
            return 1;
        }
        return incompleteBeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
    }

    /**
     * The upper tail of the chi-square distribution.
     *
     * @param x the statistic
     * @param df degrees of freedom
     * @return {@code P(X >= x)}
     */
    public static double chiSquareUpper(double x, double df) {
        if (x <= 0) {
            return 1;
        }
        return 1 - KernelTests.regularizedGammaP(df / 2, x / 2);
    }

    /**
     * The two-sided p-value of a standard normal statistic.
     *
     * @param z the statistic
     * @return {@code P(|Z| >= |z|)}
     */
    public static double normalTwoSided(double z) {
        return Math.min(1, 2 * (1 - NormalTails.cdf(Math.abs(z))));
    }
}
