package io.github.zzuegg.brainiac.experimental.imaging;

import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Rules for choosing a regularisation strength {@code lambda} without ground truth, for any
 * {@code (y, lambda) -> x} reconstruction with a linear forward model {@code A} and white noise
 * of level {@code sigma}:
 * <ul>
 *   <li>{@link Rule#DISCREPANCY} (Morozov 1966): the residual {@code |A x - y|^2 / m} closest to
 *       {@code sigma^2};</li>
 *   <li>{@link Rule#L_CURVE} (Hansen, SIAM Rev. 1992): the point of maximum curvature of
 *       {@code (log |A x - y|, log R(x))} over the grid;</li>
 *   <li>{@link Rule#GCV} (Golub, Heath &amp; Wahba 1979): the minimum of
 *       {@code m |A x - y|^2 / (m - tr H)^2}, {@code H} the influence map {@code y -> A x(y)},
 *       with a one-probe Hutchinson trace;</li>
 *   <li>{@link Rule#MC_SURE} (Ramani, Blu &amp; Unser, IEEE TIP 2008): the minimum of the
 *       Stein unbiased estimate of the prediction risk
 *       {@code |A x - y|^2 / m - sigma^2 + 2 sigma^2 div(A x)(y) / m}, the divergence estimated by
 *       one random-probe finite difference. For denoising ({@code A = I}) this is the image MSE;
 *       for deblurring it is the risk in the measurement domain.</li>
 * </ul>
 * GCV and MC-SURE share the same probe and need one extra reconstruction per grid point; the
 * noise level enters only at {@link Sweep#choose}, so a misestimated {@code sigma} costs nothing
 * to evaluate.
 */
public final class ParameterChoice {

    /** The selection rules. */
    public enum Rule {
        /** The discrepancy principle. */
        DISCREPANCY,
        /** The L-curve corner (maximum curvature). */
        L_CURVE,
        /** Generalised cross-validation with a Hutchinson trace. */
        GCV,
        /** Monte-Carlo SURE. */
        MC_SURE
    }

    /** A reconstruction for a given strength. */
    @FunctionalInterface
    public interface Reconstruction {
        /**
         * Reconstructs.
         *
         * @param y the measurements
         * @param lambda the strength
         * @return the estimate
         */
        double[] apply(double[] y, double lambda);
    }

    /**
     * A selection.
     *
     * @param rule the rule
     * @param index the chosen grid index
     * @param lambda the chosen strength
     * @param scores the rule's score per grid point (lower is better, except the L-curve's
     *     curvature, stored negated)
     */
    public record Choice(Rule rule, int index, double lambda, double[] scores) {
    }

    /**
     * The quantities of one grid sweep.
     *
     * @param lambdas the grid
     * @param estimates the reconstruction per grid point
     * @param residuals {@code |A x - y|^2} per grid point
     * @param regularisers {@code R(x)} per grid point
     * @param divergences the divergence of {@code y -> A x(y)} per grid point (the trace of the
     *     influence map)
     * @param m the number of measurements
     */
    public record Sweep(double[] lambdas, double[][] estimates, double[] residuals, double[] regularisers,
            double[] divergences, int m) {

        /**
         * Applies one rule.
         *
         * @param rule the rule
         * @param sigma the assumed noise level
         * @return the choice
         */
        public Choice choose(Rule rule, double sigma) {
            int n = lambdas.length;
            double[] s = new double[n];
            double s2 = sigma * sigma;
            switch (rule) {
                case DISCREPANCY -> {
                    for (int i = 0; i < n; i++) {
                        s[i] = Math.abs(Math.log(residuals[i] / m / s2));
                    }
                }
                case GCV -> {
                    for (int i = 0; i < n; i++) {
                        double d = Math.max(m - divergences[i], 1e-9);
                        s[i] = m * residuals[i] / (d * d);
                    }
                }
                case MC_SURE -> {
                    for (int i = 0; i < n; i++) {
                        s[i] = residuals[i] / m - s2 + 2 * s2 * divergences[i] / m;
                    }
                }
                case L_CURVE -> {
                    double[] u = new double[n];
                    double[] v = new double[n];
                    for (int i = 0; i < n; i++) {
                        u[i] = Math.log(Math.max(residuals[i], 1e-300)) / 2;
                        v[i] = Math.log(Math.max(regularisers[i], 1e-300));
                    }
                    s[0] = Double.MAX_VALUE;
                    s[n - 1] = Double.MAX_VALUE;
                    for (int i = 1; i < n - 1; i++) {
                        double t0 = Math.log(lambdas[i - 1]);
                        double t1 = Math.log(lambdas[i]);
                        double t2 = Math.log(lambdas[i + 1]);
                        double du = (u[i + 1] - u[i - 1]) / (t2 - t0);
                        double dv = (v[i + 1] - v[i - 1]) / (t2 - t0);
                        double h1 = t1 - t0;
                        double h2 = t2 - t1;
                        double ddu = 2 * (h1 * u[i + 1] - (h1 + h2) * u[i] + h2 * u[i - 1]) / (h1 * h2 * (h1 + h2));
                        double ddv = 2 * (h1 * v[i + 1] - (h1 + h2) * v[i] + h2 * v[i - 1]) / (h1 * h2 * (h1 + h2));
                        double k = (du * ddv - ddu * dv) / Math.pow(du * du + dv * dv + 1e-300, 1.5);
                        s[i] = -k;
                    }
                }
                default -> throw new IllegalStateException();
            }
            int best = 0;
            for (int i = 1; i < n; i++) {
                if (s[i] < s[best]) {
                    best = i;
                }
            }
            return new Choice(rule, best, lambdas[best], s);
        }
    }

    private ParameterChoice() {
    }

    /**
     * A log-spaced grid.
     *
     * @param lo the smallest value
     * @param hi the largest value
     * @param n the number of points
     * @return the grid
     */
    public static double[] logGrid(double lo, double hi, int n) {
        double[] g = new double[n];
        for (int i = 0; i < n; i++) {
            g[i] = lo * Math.pow(hi / lo, i / (double) (n - 1));
        }
        return g;
    }

    /**
     * Runs the reconstruction over a grid, with one Rademacher probe per grid point for the
     * divergence.
     *
     * @param f the reconstruction
     * @param a the forward model ({@code null} for the identity)
     * @param y the measurements
     * @param lambdas the grid, increasing
     * @param regulariser {@code R(x)} for the L-curve
     * @param epsilon the finite-difference step of the probe
     * @param seed the probe's seed
     * @return the sweep
     */
    public static Sweep sweep(Reconstruction f, LinearOperator a, double[] y, double[] lambdas,
            ToDoubleFunction<double[]> regulariser, double epsilon, long seed) {
        int m = y.length;
        SplittableRandom r = new SplittableRandom(seed);
        double[] b = new double[m];
        for (int i = 0; i < m; i++) {
            b[i] = r.nextBoolean() ? 1 : -1;
        }
        double[] yp = ImageMath.axpy(y, epsilon, b);
        int n = lambdas.length;
        double[][] est = new double[n][];
        double[] res = new double[n];
        double[] reg = new double[n];
        double[] div = new double[n];
        for (int i = 0; i < n; i++) {
            est[i] = f.apply(y, lambdas[i]);
            double[] ax = a == null ? est[i] : a.apply(est[i]);
            double[] axp = a == null ? f.apply(yp, lambdas[i]) : a.apply(f.apply(yp, lambdas[i]));
            double rr = 0;
            double d = 0;
            for (int j = 0; j < m; j++) {
                double e = ax[j] - y[j];
                rr += e * e;
                d += b[j] * (axp[j] - ax[j]);
            }
            res[i] = rr;
            reg[i] = regulariser.applyAsDouble(est[i]);
            div[i] = d / epsilon;
        }
        return new Sweep(lambdas.clone(), est, res, reg, div, m);
    }

    /**
     * Tikhonov deblurring with a gradient penalty, in closed form by FFT (circular boundaries):
     * {@code argmin |k * x - y|^2 + lambda |grad x|^2}.
     *
     * @param y the blurred image
     * @param side its side
     * @param kernel the kernel
     * @param k the kernel's side
     * @param lambda the strength
     * @return the estimate
     */
    public static double[] tikhonovDeblur(double[] y, int side, double[] kernel, int k, double lambda) {
        double[][] h = ImageMath.transfer(kernel, k, side);
        double[] re = y.clone();
        double[] im = new double[y.length];
        ImageMath.fft2(re, im, side, side, false);
        double[] d2 = gradientSpectrum(side);
        for (int i = 0; i < re.length; i++) {
            double hr = h[0][i];
            double hi = h[1][i];
            double den = hr * hr + hi * hi + lambda * d2[i] + 1e-12;
            double nr = hr * re[i] + hi * im[i];
            double ni = hr * im[i] - hi * re[i];
            re[i] = nr / den;
            im[i] = ni / den;
        }
        ImageMath.fft2(re, im, side, side, true);
        return re;
    }

    /**
     * The log marginal likelihood (evidence) of the Tikhonov model read as a Gaussian prior
     * {@code x ~ exp(-lambda |grad x|^2 / (2 sigma^2))}, per non-constant frequency: the
     * empirical-Bayes reference for {@link #tikhonovDeblur}.
     *
     * @param y the blurred image
     * @param side its side
     * @param kernel the kernel
     * @param k its side
     * @param sigma the noise level
     * @param lambda the strength
     * @return the log evidence up to a constant
     */
    public static double tikhonovEvidence(double[] y, int side, double[] kernel, int k, double sigma,
            double lambda) {
        double[][] h = ImageMath.transfer(kernel, k, side);
        double[] re = y.clone();
        double[] im = new double[y.length];
        ImageMath.fft2(re, im, side, side, false);
        double[] d2 = gradientSpectrum(side);
        int n = y.length;
        double s2 = sigma * sigma;
        double ll = 0;
        for (int i = 1; i < n; i++) {
            double gain = h[0][i] * h[0][i] + h[1][i] * h[1][i];
            double v = s2 + s2 * gain / (lambda * d2[i]);
            double p = (re[i] * re[i] + im[i] * im[i]) / n;
            ll -= p / v + Math.log(v);
        }
        return ll / 2;
    }

    private static double[] gradientSpectrum(int side) {
        double[] d2 = new double[side * side];
        for (int u = 0; u < side; u++) {
            for (int v = 0; v < side; v++) {
                d2[u * side + v] = 4 - 2 * Math.cos(2 * Math.PI * u / side) - 2 * Math.cos(2 * Math.PI * v / side);
            }
        }
        return d2;
    }
}
