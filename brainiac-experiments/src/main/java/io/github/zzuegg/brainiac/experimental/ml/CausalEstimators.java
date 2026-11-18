package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Average treatment effect (ATE) estimators for observational data with measured
 * confounders {@code x}, a binary treatment {@code t} and an outcome {@code y}.
 *
 * <ul>
 *   <li>{@link #naive}: the difference in means, biased whenever the confounders drive both
 *       treatment and outcome.</li>
 *   <li>{@link #ipw}: inverse propensity weighting, each arm reweighted by
 *       {@code 1 / e(x)} or {@code 1 / (1 - e(x))} (the normalized, Hajek form); consistent when
 *       the propensity model {@code e} is right.</li>
 *   <li>{@link #aipw}: augmented IPW, the doubly robust estimator
 *       {@code mean(mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e))}: consistent when
 *       either the propensity or the outcome model is right, and efficient when both are.</li>
 * </ul>
 *
 * <p>{@link #propensity} fits a logistic regression and {@link #outcomes} one ridge
 * regression per arm (a linear T-learner), the plain models to plug in; any others can be
 * passed instead. References: Robins, Rotnitzky and Zhao (1994) for AIPW; Glynn and Quinn
 * (2010) for a practical account.
 */
public final class CausalEstimators {

    private CausalEstimators() {
    }

    /**
     * The difference in mean outcome between the treated and the untreated.
     *
     * @param t treatments, 0 or 1
     * @param y outcomes
     * @return the naive estimate
     */
    public static double naive(int[] t, double[] y) {
        double s1 = 0;
        double s0 = 0;
        int n1 = 0;
        for (int i = 0; i < t.length; i++) {
            if (t[i] == 1) {
                s1 += y[i];
                n1++;
            } else {
                s0 += y[i];
            }
        }
        return s1 / n1 - s0 / (t.length - n1);
    }

    /**
     * Normalized inverse-propensity-weighted ATE.
     *
     * @param t treatments
     * @param y outcomes
     * @param e propensities {@code P(t = 1 | x)}, clipped to {@code [clip, 1 - clip]}
     * @param clip the clipping, such as 0.01
     * @return the estimate
     */
    public static double ipw(int[] t, double[] y, double[] e, double clip) {
        double s1 = 0;
        double w1 = 0;
        double s0 = 0;
        double w0 = 0;
        for (int i = 0; i < t.length; i++) {
            double p = Math.clamp(e[i], clip, 1 - clip);
            if (t[i] == 1) {
                s1 += y[i] / p;
                w1 += 1 / p;
            } else {
                s0 += y[i] / (1 - p);
                w0 += 1 / (1 - p);
            }
        }
        return s1 / w1 - s0 / w0;
    }

    /**
     * The doubly robust AIPW estimate.
     *
     * @param t treatments
     * @param y outcomes
     * @param e propensities, clipped to {@code [clip, 1 - clip]}
     * @param mu0 predicted untreated outcomes
     * @param mu1 predicted treated outcomes
     * @param clip the clipping
     * @return the estimate
     */
    public static double aipw(int[] t, double[] y, double[] e, double[] mu0, double[] mu1, double clip) {
        double s = 0;
        for (int i = 0; i < t.length; i++) {
            double p = Math.clamp(e[i], clip, 1 - clip);
            s += mu1[i] - mu0[i] + (t[i] == 1 ? (y[i] - mu1[i]) / p : -(y[i] - mu0[i]) / (1 - p));
        }
        return s / t.length;
    }

    /**
     * Propensities from an L2-regularized logistic regression of {@code t} on {@code x},
     * fitted by Newton's method.
     *
     * @param x covariates {@code [n][d]}
     * @param t treatments
     * @param l2 the ridge on the weights (not the intercept)
     * @return {@code P(t = 1 | x)} per row
     */
    public static double[] propensity(double[][] x, int[] t, double l2) {
        int n = x.length;
        int d = x[0].length + 1;
        double[] w = new double[d];
        for (int it = 0; it < 25; it++) {
            double[] g = new double[d];
            double[][] h = new double[d][d];
            for (int i = 0; i < n; i++) {
                double[] r = row(x[i]);
                double p = sigmoid(dot(w, r));
                for (int a = 0; a < d; a++) {
                    g[a] += (p - t[i]) * r[a];
                    for (int b = 0; b < d; b++) {
                        h[a][b] += p * (1 - p) * r[a] * r[b];
                    }
                }
            }
            for (int a = 1; a < d; a++) {
                g[a] += l2 * w[a];
                h[a][a] += l2;
            }
            h[0][0] += 1e-9;
            double[] step = solve(h, g);
            double change = 0;
            for (int a = 0; a < d; a++) {
                w[a] -= step[a];
                change += Math.abs(step[a]);
            }
            if (change < 1e-9) {
                break;
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = sigmoid(dot(w, row(x[i])));
        }
        return out;
    }

    /**
     * Outcome predictions from one ridge regression per arm.
     *
     * @param x covariates
     * @param t treatments
     * @param y outcomes
     * @param l2 the ridge
     * @return {@code {mu0, mu1}}, each one value per row
     */
    public static double[][] outcomes(double[][] x, int[] t, double[] y, double l2) {
        double[][] out = new double[2][x.length];
        for (int arm = 0; arm < 2; arm++) {
            int d = x[0].length + 1;
            double[][] a = new double[d][d];
            double[] b = new double[d];
            for (int i = 0; i < x.length; i++) {
                if (t[i] != arm) {
                    continue;
                }
                double[] r = row(x[i]);
                for (int p = 0; p < d; p++) {
                    b[p] += r[p] * y[i];
                    for (int q = 0; q < d; q++) {
                        a[p][q] += r[p] * r[q];
                    }
                }
            }
            for (int p = 0; p < d; p++) {
                a[p][p] += p == 0 ? 1e-9 : l2;
            }
            double[] w = solve(a, b);
            for (int i = 0; i < x.length; i++) {
                out[arm][i] = dot(w, row(x[i]));
            }
        }
        return out;
    }

    private static double[] row(double[] x) {
        double[] r = new double[x.length + 1];
        r[0] = 1;
        System.arraycopy(x, 0, r, 1, x.length);
        return r;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    /** Solves {@code a x = b} with the core {@link Linalg#solve}, in float64. */
    static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[] flat = new double[n * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, flat, i * n, n);
        }
        Tensor ta = Tensor.of(Shape.of(n, n), flat, DType.FLOAT64, Device.cpu());
        Tensor tb = Tensor.of(Shape.of(n, 1), b, DType.FLOAT64, Device.cpu());
        return Linalg.solve(ta, tb).toDoubleArray();
    }
}
