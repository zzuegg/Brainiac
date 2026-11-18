package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

/**
 * A P-spline smoother (Eilers and Marx, Statistical Science 1996): a rich cubic B-spline basis on
 * equally spaced knots with a second-order difference penalty {@code lambda ||D2 c||^2} on
 * adjacent coefficients, so the fit is as smooth as the data allow whatever the number of knots.
 * Solved as one penalised (weighted) least-squares system; {@code lambda} is chosen by
 * generalised cross-validation {@code n RSS / (n - edf)^2} over a log grid when not given.
 * The smoother of {@link AdditiveModel}.
 */
public final class PenalizedSplines {

    private final double lo;
    private final double hi;
    private final int segments;
    private final double[] coef;
    private final double lambda;
    private final double edf;

    private PenalizedSplines(double lo, double hi, int segments, double[] coef, double lambda, double edf) {
        this.lo = lo;
        this.hi = hi;
        this.segments = segments;
        this.coef = coef;
        this.lambda = lambda;
        this.edf = edf;
    }

    /**
     * Fits a smoother.
     *
     * @param x the inputs
     * @param y the targets
     * @param weights row weights, or {@code null} for ones
     * @param segments B-spline segments (basis size {@code segments + 3})
     * @param lambda the penalty, or {@code NaN} to choose it by GCV
     * @return the smoother
     */
    public static PenalizedSplines fit(double[] x, double[] y, double[] weights, int segments, double lambda) {
        int n = x.length;
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        if (hi - lo < 1e-12) {
            hi = lo + 1;
        }
        int k = segments + 3;
        double[][] btb = new double[k][k];
        double[] bty = new double[k];
        double[] row = new double[4];
        for (int i = 0; i < n; i++) {
            double w = weights == null ? 1 : weights[i];
            int j = basis(x[i], lo, hi, segments, row);
            for (int a = 0; a < 4; a++) {
                bty[j + a] += w * row[a] * y[i];
                for (int b = 0; b < 4; b++) {
                    btb[j + a][j + b] += w * row[a] * row[b];
                }
            }
        }
        double[][] pen = new double[k][k];
        for (int r = 0; r + 2 < k; r++) {
            double[] d = {1, -2, 1};
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    pen[r + a][r + b] += d[a] * d[b];
                }
            }
        }
        double wsum = 0;
        for (int i = 0; i < n; i++) {
            wsum += weights == null ? 1 : weights[i];
        }
        if (!Double.isNaN(lambda)) {
            double[][] res = solve(btb, pen, bty, lambda);
            return new PenalizedSplines(lo, hi, segments, res[0], lambda, res[1][0]);
        }
        double bestG = Double.POSITIVE_INFINITY;
        PenalizedSplines best = null;
        for (double lam = 1e-4; lam <= 1e5; lam *= 3) {
            double[][] res = solve(btb, pen, bty, lam);
            PenalizedSplines s = new PenalizedSplines(lo, hi, segments, res[0], lam, res[1][0]);
            double rss = 0;
            for (int i = 0; i < n; i++) {
                double e = y[i] - s.predict(x[i]);
                rss += (weights == null ? 1 : weights[i]) * e * e;
            }
            double g = wsum * rss / Math.pow(Math.max(1e-9, wsum - s.edf), 2);
            if (g < bestG) {
                bestG = g;
                best = s;
            }
        }
        return best;
    }

    /** Returns {coefficients, {edf}}. */
    private static double[][] solve(double[][] btb, double[][] pen, double[] bty, double lambda) {
        int k = bty.length;
        double[][] a = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                a[i][j] = btb[i][j] + lambda * pen[i][j];
            }
            a[i][i] += 1e-9;
        }
        double[][] l = MlMatrices.cholesky(a);
        double[] c = MlMatrices.choleskySolve(l, bty);
        double[][] h = MlMatrices.choleskySolve(l, btb);
        double tr = 0;
        for (int i = 0; i < k; i++) {
            tr += h[i][i];
        }
        return new double[][] {c, {tr}};
    }

    /**
     * Evaluates the four non-zero uniform cubic B-splines at {@code v} (clamped to the range).
     *
     * @param v the input
     * @param lo the range start
     * @param hi the range end
     * @param segments segments
     * @param out receives the four values
     * @return the index of the first non-zero basis function
     */
    public static int basis(double v, double lo, double hi, int segments, double[] out) {
        double h = (hi - lo) / segments;
        double t = (Math.min(hi, Math.max(lo, v)) - lo) / h;
        int j = Math.min(segments - 1, (int) Math.floor(t));
        double u = t - j;
        double u2 = u * u;
        double u3 = u2 * u;
        out[0] = (1 - u) * (1 - u) * (1 - u) / 6;
        out[1] = (3 * u3 - 6 * u2 + 4) / 6;
        out[2] = (-3 * u3 + 3 * u2 + 3 * u + 1) / 6;
        out[3] = u3 / 6;
        return j;
    }

    /**
     * Evaluates the smooth.
     *
     * @param v the input
     * @return the fitted value
     */
    public double predict(double v) {
        double[] row = new double[4];
        int j = basis(v, lo, hi, segments, row);
        double s = 0;
        for (int a = 0; a < 4; a++) {
            s += row[a] * coef[j + a];
        }
        return s;
    }

    /**
     * The chosen (or given) penalty.
     *
     * @return lambda
     */
    public double lambda() {
        return lambda;
    }

    /**
     * The effective degrees of freedom {@code tr(H)}.
     *
     * @return edf
     */
    public double edf() {
        return edf;
    }
}
