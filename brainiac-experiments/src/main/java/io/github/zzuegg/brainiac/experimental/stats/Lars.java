package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;

import java.util.ArrayList;
import java.util.List;

/**
 * Least angle regression with the lasso modification (Efron, Hastie, Johnstone and Tibshirani,
 * Annals of Statistics 2004, <a href="https://arxiv.org/abs/math/0406456">arXiv:math/0406456</a>):
 * the exact, piecewise-linear lasso path. Starting from zero, the coefficients of the active set
 * move along the equiangular direction (every active feature keeps the same absolute
 * correlation with the residual) until an inactive feature catches up (it joins) or an active
 * coefficient crosses zero (it leaves, the lasso modification). Each knot costs one small linear
 * solve, so the whole path costs about as much as one least-squares fit when {@code p} is small.
 *
 * <p>Uses the same scaling as {@link PenalizedRegression}: standardised columns (variance 1 with
 * divisor {@code n}), a centred response, and the objective {@code (1/2n)||y - X b||^2 + lambda
 * ||b||_1}, so both solvers agree at every lambda.
 */
public final class Lars {

    /**
     * The path: knots and the coefficients there, on the original scale.
     *
     * @param lambdas the knots, decreasing
     * @param intercepts the intercept at each knot
     * @param betas the coefficients at each knot
     */
    public record LassoPath(double[] lambdas, double[] intercepts, double[][] betas) {

        /**
         * Interpolates the (piecewise-linear in lambda) path.
         *
         * @param lambda the penalty level
         * @return the fit at {@code lambda}
         */
        public PenalizedRegression.Fit at(double lambda) {
            int last = lambdas.length - 1;
            if (lambda >= lambdas[0]) {
                return new PenalizedRegression.Fit(lambda, intercepts[0], betas[0].clone());
            }
            if (lambda <= lambdas[last]) {
                return new PenalizedRegression.Fit(lambda, intercepts[last], betas[last].clone());
            }
            int k = 0;
            while (lambdas[k + 1] > lambda) {
                k++;
            }
            double t = (lambdas[k] - lambda) / (lambdas[k] - lambdas[k + 1]);
            double[] b = new double[betas[k].length];
            for (int j = 0; j < b.length; j++) {
                b[j] = betas[k][j] + t * (betas[k + 1][j] - betas[k][j]);
            }
            return new PenalizedRegression.Fit(lambda, intercepts[k] + t * (intercepts[k + 1] - intercepts[k]), b);
        }
    }

    private Lars() {
    }

    /**
     * Computes the lasso path down to {@code minLambda} (or until {@code min(n - 1, p)} features
     * are active).
     *
     * @param x the design {@code [n][p]}
     * @param y the response
     * @param minLambda where to stop
     * @return the path
     */
    public static LassoPath lassoPath(double[][] x, double[] y, double minLambda) {
        int n = y.length;
        int p = x[0].length;
        double[] mean = new double[p];
        double[] sd = new double[p];
        double[][] xs = new double[p][n];
        for (int j = 0; j < p; j++) {
            double m = 0;
            for (int i = 0; i < n; i++) {
                m += x[i][j];
            }
            m /= n;
            double v = 0;
            for (int i = 0; i < n; i++) {
                v += (x[i][j] - m) * (x[i][j] - m);
            }
            double s = Math.sqrt(v / n);
            if (s < 1e-12) {
                s = 1;
            }
            mean[j] = m;
            sd[j] = s;
            for (int i = 0; i < n; i++) {
                xs[j][i] = (x[i][j] - m) / s;
            }
        }
        double ym = 0;
        for (double v : y) {
            ym += v;
        }
        ym /= n;
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            r[i] = y[i] - ym;
        }
        double[] b = new double[p];
        double[] c = new double[p];
        for (int j = 0; j < p; j++) {
            c[j] = dot(xs[j], r) / n;
        }
        List<Integer> active = new ArrayList<>();
        int first = 0;
        for (int j = 1; j < p; j++) {
            if (Math.abs(c[j]) > Math.abs(c[first])) {
                first = j;
            }
        }
        double lam = Math.abs(c[first]);
        active.add(first);
        List<double[]> knots = new ArrayList<>();
        List<Double> lams = new ArrayList<>();
        lams.add(lam);
        knots.add(b.clone());
        int maxActive = Math.min(n - 1, p);
        int guard = 0;
        while (lam > minLambda && guard++ < 8 * p) {
            int k = active.size();
            double[][] gram = new double[k][k];
            double[] s = new double[k];
            for (int a = 0; a < k; a++) {
                int ja = active.get(a);
                s[a] = Math.signum(c[ja]);
                for (int d = 0; d < k; d++) {
                    gram[a][d] = dot(xs[ja], xs[active.get(d)]) / n;
                }
            }
            double[] dir = Recalibrators.solve(gram, s);
            // u = X_A dir: the change of the fitted values per unit step
            double[] u = new double[n];
            for (int a = 0; a < k; a++) {
                double[] col = xs[active.get(a)];
                for (int i = 0; i < n; i++) {
                    u[i] += dir[a] * col[i];
                }
            }
            double step = lam - Math.max(minLambda, 0);
            int join = -1;
            int drop = -1;
            boolean[] isActive = new boolean[p];
            for (int ja : active) {
                isActive[ja] = true;
            }
            if (k < maxActive) {
                for (int j = 0; j < p; j++) {
                    if (isActive[j]) {
                        continue;
                    }
                    double aj = dot(xs[j], u) / n;
                    double g1 = (lam - c[j]) / (1 - aj);
                    double g2 = (lam + c[j]) / (1 + aj);
                    for (double gm : new double[] {g1, g2}) {
                        if (gm > 1e-12 && gm < step) {
                            step = gm;
                            join = j;
                        }
                    }
                }
            }
            for (int a = 0; a < k; a++) {
                int ja = active.get(a);
                if (dir[a] != 0) {
                    double gm = -b[ja] / dir[a];
                    if (gm > 1e-12 && gm < step) {
                        step = gm;
                        drop = a;
                        join = -1;
                    }
                }
            }
            for (int a = 0; a < k; a++) {
                b[active.get(a)] += step * dir[a];
            }
            for (int i = 0; i < n; i++) {
                r[i] -= step * u[i];
            }
            lam -= step;
            for (int j = 0; j < p; j++) {
                c[j] = dot(xs[j], r) / n;
            }
            if (drop >= 0) {
                b[active.get(drop)] = 0;
                active.remove(drop);
            } else if (join >= 0) {
                active.add(join);
            }
            lams.add(lam);
            knots.add(b.clone());
            if (join < 0 && drop < 0) {
                break;
            }
        }
        int m = knots.size();
        double[] l = new double[m];
        double[] icpt = new double[m];
        double[][] betas = new double[m][p];
        for (int t = 0; t < m; t++) {
            l[t] = lams.get(t);
            double b0 = ym;
            for (int j = 0; j < p; j++) {
                betas[t][j] = knots.get(t)[j] / sd[j];
                b0 -= betas[t][j] * mean[j];
            }
            icpt[t] = b0;
        }
        return new LassoPath(l, icpt, betas);
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }
}
