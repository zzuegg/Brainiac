package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.function.UnaryOperator;

/**
 * Anderson acceleration (Anderson 1965; Walker and Ni 2011,
 * <a href="https://epubs.siam.org/doi/10.1137/10078356X">doi:10.1137/10078356X</a>) for fixed
 * points {@code z = f(z)}: instead of iterating {@code z <- f(z)}, it mixes the last {@code m}
 * iterates with the weights {@code alpha} ({@code sum alpha = 1}) that minimize the norm of the
 * combined residual {@code sum alpha_i (f(z_i) - z_i)}, a multisecant quasi-Newton step:
 * {@code z <- beta sum alpha_i f(z_i) + (1 - beta) sum alpha_i z_i}. For a tensor of rank 2 or
 * more, each row (example) is solved as its own problem with its own weights, as deep
 * equilibrium models do. It is the forward and
 * backward solver of deep equilibrium models ({@code DeepEquilibriumLayer}).
 */
public final class AndersonSolver {

    /**
     * A solve's outcome.
     *
     * @param z the fixed point found
     * @param iterations evaluations of {@code f}
     * @param residual the final {@code |f(z) - z| / |f(z)|}
     */
    public record Result(Tensor z, int iterations, double residual) {
    }

    private final int memory;
    private final int maxIterations;
    private final double tolerance;
    private final double beta;

    /**
     * @param memory the iterates mixed, {@code m}, for example 5; 1 gives the plain (damped)
     *     fixed-point iteration
     * @param maxIterations the most evaluations of {@code f}
     * @param tolerance stops when the relative residual falls below it
     * @param beta the mixing parameter, 1 for undamped
     */
    public AndersonSolver(int memory, int maxIterations, double tolerance, double beta) {
        this.memory = memory;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.beta = beta;
    }

    /**
     * Finds a fixed point.
     *
     * @param f the map
     * @param z0 the start
     * @return the solution and its statistics
     */
    public Result solve(UnaryOperator<Tensor> f, Tensor z0) {
        int n = Math.toIntExact(z0.size());
        int rows = z0.rank() >= 2 ? z0.shape().dim(0) : 1;
        int d = n / rows;
        double[][] xs = new double[memory][];
        double[][] gs = new double[memory][];
        double[][] fs = new double[memory][];
        double[] x = toDouble(z0);
        double residual = Double.POSITIVE_INFINITY;
        int k = 0;
        while (k < maxIterations) {
            double[] fx = toDouble(f.apply(Tensor.of(z0.shape(), toFloat(x))));
            k++;
            double[] g = new double[n];
            double gn = 0;
            double fn = 0;
            for (int i = 0; i < n; i++) {
                g[i] = fx[i] - x[i];
                gn += g[i] * g[i];
                fn += fx[i] * fx[i];
            }
            residual = Math.sqrt(gn) / (Math.sqrt(fn) + 1e-12);
            if (residual < tolerance) {
                x = fx;
                break;
            }
            int slot = (k - 1) % memory;
            xs[slot] = x;
            gs[slot] = g;
            fs[slot] = fx;
            int used = Math.min(k, memory);
            double[] next = new double[n];
            for (int r = 0; r < rows; r++) {
                // each row (example) is its own fixed-point problem with its own mixing weights;
                // a row that has converged keeps its last image instead of mixing in older iterates
                double rg = 0;
                double rf = 0;
                for (int i = r * d; i < (r + 1) * d; i++) {
                    rg += g[i] * g[i];
                    rf += fx[i] * fx[i];
                }
                if (Math.sqrt(rg) <= 0.1 * tolerance * (Math.sqrt(rf) + 1e-12)) {
                    System.arraycopy(fx, r * d, next, r * d, d);
                    continue;
                }
                double[] alpha = weights(gs, used, r * d, d);
                for (int j = 0; j < used; j++) {
                    double a = alpha[j];
                    for (int i = r * d; i < (r + 1) * d; i++) {
                        next[i] += a * (beta * fs[j][i] + (1 - beta) * xs[j][i]);
                    }
                }
            }
            x = next;
        }
        return new Result(Tensor.of(z0.shape(), toFloat(x)), k, residual);
    }

    /** The mixing weights: {@code H^-1 1 / (1^T H^-1 1)} with {@code H = G G^T + reg}. */
    private static double[] weights(double[][] g, int used, int from, int length) {
        if (used == 1) {
            return new double[] {1};
        }
        double[][] h = new double[used][used];
        double trace = 0;
        for (int a = 0; a < used; a++) {
            for (int b = a; b < used; b++) {
                double s = 0;
                for (int i = from; i < from + length; i++) {
                    s += g[a][i] * g[b][i];
                }
                h[a][b] = s;
                h[b][a] = s;
            }
            trace += h[a][a];
        }
        for (int a = 0; a < used; a++) {
            h[a][a] += 1e-3 * trace / used + 1e-20;
        }
        double[] ones = new double[used];
        java.util.Arrays.fill(ones, 1);
        double[] y = MlMatrices.choleskySolve(MlMatrices.cholesky(h), ones);
        double sum = 0;
        for (double v : y) {
            sum += v;
        }
        for (int a = 0; a < used; a++) {
            y[a] /= sum;
        }
        return y;
    }

    private static double[] toDouble(Tensor t) {
        float[] f = t.toArray();
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    private static float[] toFloat(double[] d) {
        float[] f = new float[d.length];
        for (int i = 0; i < d.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }

    @Override
    public String toString() {
        return "AndersonSolver(m=" + memory + ")";
    }
}
