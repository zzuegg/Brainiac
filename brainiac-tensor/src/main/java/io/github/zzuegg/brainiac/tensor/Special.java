package io.github.zzuegg.brainiac.tensor;

/**
 * Special functions of each value: the log gamma function and its derivatives, as Beta,
 * Dirichlet, Gamma and Student-t log densities and variational inference need them. They
 * are computed in double precision on the host and returned on the input's device.
 */
public final class Special {

    private static final double[] LANCZOS = {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
        1.5056327351493116e-7};

    private Special() {
    }

    /**
     * {@code ln |Gamma(x)|}, as {@code torch.lgamma}.
     *
     * @param x the values
     * @return the log gamma of each, infinite at 0 and the negative integers
     */
    public static Tensor lgamma(Tensor x) {
        return map(x, Special::lgamma);
    }

    /**
     * The digamma function {@code d/dx ln Gamma(x)}, as {@code torch.digamma}.
     *
     * @param x the values
     * @return the digamma of each, NaN at 0 and the negative integers
     */
    public static Tensor digamma(Tensor x) {
        return map(x, Special::digamma);
    }

    /**
     * The trigamma function, the derivative of {@link #digamma(Tensor)}.
     *
     * @param x the values
     * @return the trigamma of each
     */
    public static Tensor trigamma(Tensor x) {
        return map(x, Special::trigamma);
    }

    /**
     * The complementary error function {@code 1 - erf(x)}, accurate in the tail where
     * {@code 1 - erf(x)} would round to 0.
     *
     * @param x the values
     * @return {@code erfc} of each
     */
    public static Tensor erfc(Tensor x) {
        return map(x, Special::erfc);
    }

    /**
     * The quantile function of the standard normal distribution (probit, {@code ndtri}).
     *
     * @param p probabilities in {@code (0, 1)}
     * @return {@code z} with {@code P(Z <= z) = p}; infinite at 0 and 1, NaN outside
     */
    public static Tensor normalQuantile(Tensor p) {
        return map(p, Special::normalQuantile);
    }

    private static Tensor map(Tensor x, java.util.function.DoubleUnaryOperator f) {
        double[] d = x.toDoubleArray();
        for (int i = 0; i < d.length; i++) {
            d[i] = f.applyAsDouble(d[i]);
        }
        return Tensor.of(x.shape(), d, x.dtype(), x.device());
    }

    /**
     * {@code ln |Gamma(x)|} of one value.
     *
     * @param x the argument
     * @return the log gamma
     */
    public static double lgamma(double x) {
        if (Double.isNaN(x)) {
            return x;
        }
        if (x <= 0 && x == Math.rint(x)) {
            return Double.POSITIVE_INFINITY;
        }
        if (x < 0.5) {
            // reflection: Gamma(x) Gamma(1 - x) = pi / sin(pi x)
            return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
        }
        double y = x - 1;
        double a = LANCZOS[0];
        double t = y + 7.5;
        for (int i = 1; i < LANCZOS.length; i++) {
            a += LANCZOS[i] / (y + i);
        }
        return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
    }

    /**
     * The digamma function of one value.
     *
     * @param x the argument
     * @return the digamma
     */
    public static double digamma(double x) {
        if (Double.isNaN(x) || x <= 0 && x == Math.rint(x)) {
            return Double.NaN;
        }
        if (x < 0) {
            // reflection: psi(1 - x) - psi(x) = pi cot(pi x)
            return digamma(1 - x) - Math.PI / Math.tan(Math.PI * x);
        }
        double r = 0;
        while (x < 12) {
            r -= 1 / x;
            x += 1;
        }
        double f = 1 / (x * x);
        return r + Math.log(x) - 0.5 / x
                - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))));
    }

    /**
     * The trigamma function of one value.
     *
     * @param x the argument
     * @return the trigamma
     */
    public static double trigamma(double x) {
        if (Double.isNaN(x) || x <= 0 && x == Math.rint(x)) {
            return Double.NaN;
        }
        if (x < 0) {
            // reflection: psi1(1 - x) + psi1(x) = pi^2 / sin^2(pi x)
            double s = Math.sin(Math.PI * x);
            return -trigamma(1 - x) + Math.PI * Math.PI / (s * s);
        }
        double r = 0;
        while (x < 12) {
            r += 1 / (x * x);
            x += 1;
        }
        double f = 1 / (x * x);
        return r + 1 / x + f / 2
                + f / x * (1.0 / 6 - f * (1.0 / 30 - f * (1.0 / 42 - f / 30)));
    }

    /**
     * The complementary error function of one value (W. J. Cody's rational approximations,
     * relative error below 1e-15 in double).
     *
     * @param x the argument
     * @return {@code 1 - erf(x)}
     */
    public static double erfc(double x) {
        if (Double.isNaN(x)) {
            return x;
        }
        double z = Math.abs(x);
        double r;
        if (z < 0.5) {
            // erf by its Taylor series
            double t = z * z;
            double term = z;
            double sum = z;
            for (int n = 1; n < 40; n++) {
                term *= -t / n;
                double next = term / (2 * n + 1);
                sum += next;
                if (Math.abs(next) < 1e-17 * Math.abs(sum)) {
                    break;
                }
            }
            double erf = 2 / Math.sqrt(Math.PI) * sum;
            return x >= 0 ? 1 - erf : 1 + erf;
        }
        // continued fraction for the tail (Lentz), converging fast for z >= 0.5
        double tiny = 1e-300;
        double b = 2 * z * z + 1;
        double f = b;
        double c = b;
        double d = 0;
        for (int n = 1; n < 500; n++) {
            double a = -(2.0 * n - 1) * (2.0 * n);
            b += 4;
            d = b + a * d;
            d = Math.abs(d) < tiny ? tiny : d;
            c = b + a / c;
            c = Math.abs(c) < tiny ? tiny : c;
            d = 1 / d;
            double delta = c * d;
            f *= delta;
            if (Math.abs(delta - 1) < 1e-16) {
                break;
            }
        }
        r = 2 * z / Math.sqrt(Math.PI) * Math.exp(-z * z) / f;
        return x >= 0 ? r : 2 - r;
    }

    /**
     * The standard normal quantile of one probability (Acklam's approximation refined by
     * one Halley step, near full double precision).
     *
     * @param p the probability
     * @return the quantile
     */
    public static double normalQuantile(double p) {
        if (Double.isNaN(p) || p < 0 || p > 1) {
            return Double.NaN;
        }
        if (p == 0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (p == 1) {
            return Double.POSITIVE_INFINITY;
        }
        if (p > 0.5) {
            // 1 - p is exact here, and the lower tail keeps full relative precision
            return -normalQuantile(1 - p);
        }
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
            6.680131188771972e+01, -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
            -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
            3.754408661907416e+00};
        double low = 0.02425;
        double x;
        if (p < low) {
            double q = Math.sqrt(-2 * Math.log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        } else if (p <= 1 - low) {
            double q = p - 0.5;
            double r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        } else {
            double q = Math.sqrt(-2 * Math.log(1 - p));
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        // Halley refinement against the exact CDF 0.5 erfc(-x / sqrt 2)
        for (int i = 0; i < 2; i++) {
            double e = 0.5 * erfc(-x / Math.sqrt(2)) - p;
            double u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
            x = x - u / (1 + x * u / 2);
        }
        return x;
    }

    /**
     * The regularized lower incomplete gamma function {@code P(a, x)}, the CDF of a gamma
     * distribution with shape {@code a} and scale 1.
     *
     * @param a the shape, positive
     * @param x the argument, at least 0
     * @return {@code P(a, x)} in [0, 1]
     */
    public static double gammaP(double a, double x) {
        if (!(a > 0) || !(x >= 0)) {
            throw new IllegalArgumentException("gammaP needs a > 0 and x >= 0, got " + a + ", " + x);
        }
        if (x == 0) {
            return 0;
        }
        return x < a + 1 ? gammaSeries(a, x) : 1 - gammaFraction(a, x);
    }

    /**
     * The regularized upper incomplete gamma function {@code Q(a, x) = 1 - P(a, x)}, kept
     * accurate where it is tiny.
     *
     * @param a the shape, positive
     * @param x the argument, at least 0
     * @return {@code Q(a, x)} in [0, 1]
     */
    public static double gammaQ(double a, double x) {
        if (!(a > 0) || !(x >= 0)) {
            throw new IllegalArgumentException("gammaQ needs a > 0 and x >= 0, got " + a + ", " + x);
        }
        if (x == 0) {
            return 1;
        }
        return x < a + 1 ? 1 - gammaSeries(a, x) : gammaFraction(a, x);
    }

    private static double gammaSeries(double a, double x) {
        double term = 1 / a;
        double sum = term;
        for (int n = 1; n < 10_000; n++) {
            term *= x / (a + n);
            sum += term;
            if (Math.abs(term) < Math.abs(sum) * 1e-16) {
                break;
            }
        }
        return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
    }

    // Q(a, x) by Lentz's continued fraction
    private static double gammaFraction(double a, double x) {
        double tiny = 1e-300;
        double b = x + 1 - a;
        double c = 1 / tiny;
        double d = 1 / b;
        double h = d;
        for (int i = 1; i < 10_000; i++) {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            d = Math.abs(d) < tiny ? tiny : d;
            c = b + an / c;
            c = Math.abs(c) < tiny ? tiny : c;
            d = 1 / d;
            double delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1) < 1e-16) {
                break;
            }
        }
        return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
    }

    /**
     * The regularized incomplete beta function {@code I_x(a, b)}, the CDF of a beta
     * distribution.
     *
     * @param a the first shape, positive
     * @param b the second shape, positive
     * @param x the argument, in [0, 1]
     * @return {@code I_x(a, b)} in [0, 1]
     */
    public static double betaI(double a, double b, double x) {
        if (!(a > 0) || !(b > 0) || !(x >= 0 && x <= 1)) {
            throw new IllegalArgumentException("betaI needs a, b > 0 and x in [0, 1], got " + a + ", " + b + ", " + x);
        }
        if (x == 0 || x == 1) {
            return x;
        }
        double front = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log1p(-x));
        // the continued fraction converges fast below the mean; use the symmetry above it
        return x < (a + 1) / (a + b + 2) ? front * betaFraction(a, b, x) / a
                : 1 - front * betaFraction(b, a, 1 - x) / b;
    }

    private static double betaFraction(double a, double b, double x) {
        double tiny = 1e-300;
        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        d = Math.abs(d) < tiny ? tiny : d;
        d = 1 / d;
        double h = d;
        for (int m = 1; m < 10_000; m++) {
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
            double delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1) < 1e-16) {
                break;
            }
        }
        return h;
    }
}
