package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.tensor.Distributions;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Optimize;

import java.util.function.ToDoubleFunction;

/**
 * Capture–recapture estimators of population size and survival when detection is imperfect
 * (Seber, "The estimation of animal abundance", 1982).
 *
 * <ul>
 *   <li>{@link Estimator#LINCOLN_PETERSEN}: two occasions, {@code N = n1 n2 / m2}.</li>
 *   <li>{@link Estimator#CHAPMAN}: its nearly unbiased correction
 *       {@code N = (n1 + 1)(n2 + 1) / (m2 + 1) - 1} with Seber's variance.</li>
 *   <li>{@link Estimator#SCHNABEL}: several occasions, {@code N = sum C_t M_t / (sum R_t + 1)},
 *       with a Poisson interval on the recaptures.</li>
 *   <li>{@link Estimator#CJS}: the Cormack–Jolly–Seber model for open populations, constant
 *       survival {@code phi} and recapture {@code p} by maximum likelihood (core
 *       {@link Optimize#nelderMead} on logits), Wald intervals from the numerical Hessian.</li>
 * </ul>
 */
public final class CaptureRecapture {

    /** The estimator. */
    public enum Estimator {
        /** Lincoln–Petersen. */
        LINCOLN_PETERSEN,
        /** Chapman's bias-corrected Lincoln–Petersen. */
        CHAPMAN,
        /** Schnabel's multi-occasion estimator. */
        SCHNABEL,
        /** Cormack–Jolly–Seber survival (returns phi). */
        CJS
    }

    /**
     * An estimate with a 95% interval.
     *
     * @param value the estimate ({@code N}, or {@code phi} for CJS)
     * @param lower the lower bound
     * @param upper the upper bound
     * @param other a second parameter (CJS: {@code p}), else NaN
     */
    public record Estimate(double value, double lower, double upper, double other) {

        /**
         * @param truth the true value
         * @return whether the interval covers it
         */
        public boolean covers(double truth) {
            return truth >= lower && truth <= upper;
        }
    }

    private CaptureRecapture() {
    }

    /**
     * Applies an estimator.
     *
     * @param estimator the estimator
     * @param histories capture histories {@code [animal][occasion]} of the animals ever caught
     * @return the estimate
     */
    public static Estimate estimate(Estimator estimator, int[][] histories) {
        return switch (estimator) {
            case LINCOLN_PETERSEN, CHAPMAN -> twoOccasion(histories, estimator == Estimator.CHAPMAN);
            case SCHNABEL -> schnabel(histories);
            case CJS -> cjs(histories);
        };
    }

    private static Estimate twoOccasion(int[][] h, boolean chapman) {
        double n1 = 0;
        double n2 = 0;
        double m2 = 0;
        for (int[] row : h) {
            n1 += row[0];
            n2 += row[1];
            m2 += row[0] * row[1];
        }
        double n;
        double var;
        if (chapman) {
            n = (n1 + 1) * (n2 + 1) / (m2 + 1) - 1;
            var = (n1 + 1) * (n2 + 1) * (n1 - m2) * (n2 - m2) / ((m2 + 1) * (m2 + 1) * (m2 + 2));
        } else {
            n = m2 > 0 ? n1 * n2 / m2 : Double.POSITIVE_INFINITY;
            var = m2 > 0 ? n1 * n2 * (n1 - m2) * (n2 - m2) / (m2 * m2 * m2) : Double.POSITIVE_INFINITY;
        }
        double sd = Math.sqrt(var);
        return new Estimate(n, n - 1.96 * sd, n + 1.96 * sd, Double.NaN);
    }

    private static Estimate schnabel(int[][] h) {
        int k = h[0].length;
        double cm = 0;
        double rsum = 0;
        for (int t = 0; t < k; t++) {
            double c = 0;
            double rec = 0;
            double marked = 0;
            for (int[] row : h) {
                boolean before = false;
                for (int s = 0; s < t; s++) {
                    before |= row[s] == 1;
                }
                marked += before ? 1 : 0;
                c += row[t];
                rec += row[t] == 1 && before ? 1 : 0;
            }
            cm += c * marked;
            rsum += rec;
        }
        double n = cm / (rsum + 1);
        // exact Poisson interval for the recaptures
        double lo = rsum == 0 ? 0 : Distributions.chiSquaredQuantile(0.025, 2 * rsum) / 2;
        double hi = Distributions.chiSquaredQuantile(0.975, 2 * (rsum + 1)) / 2;
        return new Estimate(n, cm / (hi + 1), cm / Math.max(lo + 1, 1e-9), Double.NaN);
    }

    /**
     * The CJS negative log-likelihood for constant survival and recapture probabilities.
     *
     * @param h histories (first capture = release)
     * @param phi survival per interval
     * @param p recapture probability
     * @return the NLL
     */
    public static double cjsNll(int[][] h, double phi, double p) {
        int k = h[0].length;
        double[] chi = new double[k];
        chi[k - 1] = 1;
        for (int t = k - 2; t >= 0; t--) {
            chi[t] = (1 - phi) + phi * (1 - p) * chi[t + 1];
        }
        double ll = 0;
        for (int[] row : h) {
            int f = -1;
            int l = -1;
            for (int t = 0; t < k; t++) {
                if (row[t] == 1) {
                    if (f < 0) {
                        f = t;
                    }
                    l = t;
                }
            }
            if (f < 0) {
                continue;
            }
            for (int t = f; t < l; t++) {
                ll += Math.log(phi) + (row[t + 1] == 1 ? Math.log(p) : Math.log(1 - p));
            }
            ll += Math.log(chi[l]);
        }
        return -ll;
    }

    private static Estimate cjs(int[][] h) {
        ToDoubleFunction<double[]> f = x -> cjsNll(h, sig(x[0]), sig(x[1]));
        Optimize.Minimum m = Optimize.nelderMead(f, new double[] {0.5, 0}, 0.5, 2000, 1e-12);
        double[][] hess = hessian(f, m.x(), 1e-4);
        double[][] cov = Matrices.inverse(hess);
        double se = Math.sqrt(Math.max(cov[0][0], 0));
        return new Estimate(sig(m.x()[0]), sig(m.x()[0] - 1.96 * se), sig(m.x()[0] + 1.96 * se), sig(m.x()[1]));
    }

    static double sig(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    /**
     * A central-difference Hessian.
     *
     * @param f the function
     * @param x the point
     * @param h the step
     * @return the Hessian
     */
    public static double[][] hessian(ToDoubleFunction<double[]> f, double[] x, double h) {
        int n = x.length;
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double[] a = x.clone();
                double[] b = x.clone();
                double[] c = x.clone();
                double[] d = x.clone();
                a[i] += h;
                a[j] += h;
                b[i] += h;
                b[j] -= h;
                c[i] -= h;
                c[j] += h;
                d[i] -= h;
                d[j] -= h;
                double v = (f.applyAsDouble(a) - f.applyAsDouble(b) - f.applyAsDouble(c) + f.applyAsDouble(d))
                        / (4 * h * h);
                out[i][j] = v;
                out[j][i] = v;
            }
        }
        return out;
    }
}
