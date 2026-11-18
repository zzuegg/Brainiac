package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;
import java.util.function.Function;

/**
 * Levenberg-Marquardt (Levenberg 1944; Marquardt 1963,
 * <a href="https://epubs.siam.org/doi/10.1137/0111030">doi:10.1137/0111030</a>), the damped
 * Gauss-Newton trainer for small least-squares networks: with the full Jacobian {@code J} of
 * the residuals {@code r}, solve {@code (J^T J + lambda diag(J^T J)) delta = -J^T r} exactly
 * by Cholesky, accept the step if it lowers {@code |r|^2} and divide {@code lambda} by 10,
 * otherwise multiply {@code lambda} by 10 and solve again. It converges quadratically near a
 * minimum; the cost grows with the cube of the parameter count, so it suits networks of up to
 * a few thousand weights.
 */
public final class LevenbergMarquardt {

    private double lambda;
    private final double minLambda;
    private final double maxLambda;

    /**
     * @param initialLambda the starting damping, for example 1e-2
     */
    public LevenbergMarquardt(double initialLambda) {
        this.lambda = initialLambda;
        this.minLambda = 1e-9;
        this.maxLambda = 1e9;
    }

    /** @return the current damping */
    public double damping() {
        return lambda;
    }

    /**
     * One Levenberg-Marquardt iteration.
     *
     * @param params the parameters
     * @param residuals computes the residuals (any shape) from the parameters
     * @return the mean squared residual after the iteration
     */
    public double step(List<Var> params, Function<List<Var>, Var> residuals) {
        Var out = residuals.apply(params);
        float[] rv = out.value().toArray();
        int m = rv.length;
        double sse0 = 0;
        for (float v : rv) {
            sse0 += (double) v * v;
        }
        double[][] j = CurvatureProducts.jacobian(out, params);
        int n = j[0].length;
        double[][] jtj = new double[n][n];
        double[] jtr = new double[n];
        for (int k = 0; k < m; k++) {
            double[] row = j[k];
            for (int a = 0; a < n; a++) {
                double ra = row[a];
                if (ra == 0) {
                    continue;
                }
                jtr[a] += ra * rv[k];
                for (int b = a; b < n; b++) {
                    jtj[a][b] += ra * row[b];
                }
            }
        }
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < a; b++) {
                jtj[a][b] = jtj[b][a];
            }
        }
        double[] x = CurvatureProducts.values(params);
        for (int attempt = 0; attempt < 12; attempt++) {
            double[][] sys = new double[n][];
            for (int a = 0; a < n; a++) {
                sys[a] = jtj[a].clone();
                sys[a][a] += lambda * Math.max(jtj[a][a], 1e-9) + 1e-12;
            }
            double[] neg = new double[n];
            for (int a = 0; a < n; a++) {
                neg[a] = -jtr[a];
            }
            double[] delta = CurvatureProducts.solveSpd(sys, neg);
            CurvatureProducts.assign(params, CurvatureProducts.axpy(x, 1, delta));
            float[] r1 = Var.noGrad(() -> residuals.apply(params)).value().toArray();
            double sse = 0;
            for (float v : r1) {
                sse += (double) v * v;
            }
            if (Double.isFinite(sse) && sse < sse0) {
                lambda = Math.max(minLambda, lambda / 10);
                return sse / m;
            }
            lambda = Math.min(maxLambda, lambda * 10);
        }
        CurvatureProducts.assign(params, x);
        return sse0 / m;
    }

    @Override
    public String toString() {
        return "LevenbergMarquardt(lambda=" + lambda + ")";
    }
}
