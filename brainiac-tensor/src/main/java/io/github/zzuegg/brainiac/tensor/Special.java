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
}
