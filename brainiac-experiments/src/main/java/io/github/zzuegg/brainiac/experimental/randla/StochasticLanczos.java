package io.github.zzuegg.brainiac.experimental.randla;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;

/**
 * Stochastic Lanczos quadrature (Ubaru, Chen and Saad, SIMAX 2017; Golub and Meurant 2010) for
 * spectral sums {@code tr f(A) = sum_i f(lambda_i)} of a symmetric matrix seen only through
 * matrix-vector products.
 *
 * <p>For a Rademacher probe {@code v}, {@code v^T f(A) v} is a Riemann-Stieltjes integral over the
 * spectrum that {@code m} Lanczos steps turn into an {@code m}-point Gauss quadrature: with the
 * tridiagonal {@code T = Q diag(theta) Q^T}, {@code v^T f(A) v ~ |v|^2 sum_k Q_{0k}^2 f(theta_k)}.
 * Averaging a few probes (Hutchinson) gives {@code tr f(A)}: log-determinants of kernel matrices,
 * spectral densities (a Gaussian-smoothed sum of the nodes) and Hessian spectra of networks
 * through Hessian-vector products. Lanczos uses full re-orthogonalisation, which the small
 * {@code m} makes affordable and which keeps ghost eigenvalues away.
 */
public final class StochasticLanczos {

    /** A symmetric matrix as a product. */
    @FunctionalInterface
    public interface MatVec {

        /**
         * {@code A v}.
         *
         * @param v the vector
         * @return the product
         */
        double[] apply(double[] v);

        /**
         * A dense symmetric matrix as a product.
         *
         * @param a the matrix
         * @return the operator
         */
        static MatVec of(double[][] a) {
            return v -> {
                double[] y = new double[a.length];
                for (int i = 0; i < a.length; i++) {
                    double s = 0;
                    for (int j = 0; j < v.length; j++) {
                        s += a[i][j] * v[j];
                    }
                    y[i] = s;
                }
                return y;
            };
        }
    }

    /**
     * Gauss quadrature nodes and weights from one Lanczos run.
     *
     * @param nodes the Ritz values {@code theta_k}
     * @param weights {@code Q_{0k}^2}, summing to one
     */
    public record Quadrature(double[] nodes, double[] weights) {
    }

    private StochasticLanczos() {
    }

    /**
     * Runs {@code steps} Lanczos iterations from {@code start} and returns the Gauss quadrature
     * of the spectral measure of {@code start / |start|}.
     *
     * @param a the operator
     * @param start the starting vector
     * @param steps the Lanczos steps {@code m}
     * @return the quadrature
     */
    public static Quadrature lanczos(MatVec a, double[] start, int steps) {
        int n = start.length;
        int m = Math.min(steps, n);
        List<double[]> basis = new ArrayList<>();
        double[] alpha = new double[m];
        double[] beta = new double[m];
        double[] q = start.clone();
        double nq = SketchedLeastSquares.norm(q);
        for (int i = 0; i < n; i++) {
            q[i] /= nq;
        }
        int used = 0;
        for (int k = 0; k < m; k++) {
            basis.add(q);
            double[] w = a.apply(q);
            double al = 0;
            for (int i = 0; i < n; i++) {
                al += w[i] * q[i];
            }
            alpha[k] = al;
            used = k + 1;
            // full re-orthogonalisation (twice is enough)
            for (int pass = 0; pass < 2; pass++) {
                for (double[] b : basis) {
                    double d = 0;
                    for (int i = 0; i < n; i++) {
                        d += w[i] * b[i];
                    }
                    for (int i = 0; i < n; i++) {
                        w[i] -= d * b[i];
                    }
                }
            }
            double bt = SketchedLeastSquares.norm(w);
            if (k == m - 1 || bt < 1e-12) {
                break;
            }
            beta[k] = bt;
            double[] next = new double[n];
            for (int i = 0; i < n; i++) {
                next[i] = w[i] / bt;
            }
            q = next;
        }
        double[][] t = new double[used][used];
        for (int k = 0; k < used; k++) {
            t[k][k] = alpha[k];
            if (k + 1 < used) {
                t[k][k + 1] = beta[k];
                t[k + 1][k] = beta[k];
            }
        }
        MlMatrices.Eigen e = MlMatrices.eigh(t);
        double[] w = new double[used];
        for (int k = 0; k < used; k++) {
            w[k] = e.vectors()[0][k] * e.vectors()[0][k];
        }
        return new Quadrature(e.values(), w);
    }

    /**
     * Estimates {@code tr f(A)}.
     *
     * @param a the operator
     * @param n the dimension
     * @param f the spectral function, for example {@code Math::log}
     * @param probes Rademacher probes, for example 10
     * @param steps Lanczos steps per probe, for example 30
     * @param seed the seed
     * @return the estimate
     */
    public static double trace(MatVec a, int n, DoubleUnaryOperator f, int probes, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        for (int p = 0; p < probes; p++) {
            Quadrature qd = lanczos(a, rademacher(n, r), steps);
            double s = 0;
            for (int k = 0; k < qd.nodes().length; k++) {
                s += qd.weights()[k] * f.applyAsDouble(qd.nodes()[k]);
            }
            total += n * s / probes;   // |v|^2 = n for Rademacher probes
        }
        return total;
    }

    /**
     * The log-determinant of a symmetric positive-definite operator.
     *
     * @param a the operator
     * @param n the dimension
     * @param probes probes
     * @param steps Lanczos steps
     * @param seed the seed
     * @return the estimate of {@code log det A}
     */
    public static double logDet(MatVec a, int n, int probes, int steps, long seed) {
        return trace(a, n, x -> Math.log(Math.max(x, 1e-300)), probes, steps, seed);
    }

    /**
     * Hutchinson's trace estimator {@code mean v^T A v}: the baseline for {@code tr A}.
     *
     * @param a the operator
     * @param n the dimension
     * @param probes probes
     * @param seed the seed
     * @return the estimate
     */
    public static double hutchinson(MatVec a, int n, int probes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        for (int p = 0; p < probes; p++) {
            double[] v = rademacher(n, r);
            double[] av = a.apply(v);
            double s = 0;
            for (int i = 0; i < n; i++) {
                s += v[i] * av[i];
            }
            total += s / probes;
        }
        return total;
    }

    /**
     * The spectral density {@code (1/n) sum_i N(x; lambda_i, sigma^2)} on a grid.
     *
     * @param a the operator
     * @param n the dimension
     * @param grid the evaluation points
     * @param sigma the Gaussian smoothing width
     * @param probes probes
     * @param steps Lanczos steps
     * @param seed the seed
     * @return the density at each grid point
     */
    public static double[] spectralDensity(MatVec a, int n, double[] grid, double sigma, int probes, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] density = new double[grid.length];
        double norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
        for (int p = 0; p < probes; p++) {
            Quadrature qd = lanczos(a, rademacher(n, r), steps);
            for (int g = 0; g < grid.length; g++) {
                double s = 0;
                for (int k = 0; k < qd.nodes().length; k++) {
                    double z = (grid[g] - qd.nodes()[k]) / sigma;
                    s += qd.weights()[k] * norm * Math.exp(-0.5 * z * z);
                }
                density[g] += s / probes;
            }
        }
        return density;
    }

    /**
     * The Hessian of a scalar loss as a product, by double backpropagation:
     * {@code H v = d/dw (grad L . v)}.
     *
     * @param parameters the parameters (their current values are the expansion point)
     * @param loss builds the loss from the parameters
     * @return the Hessian-vector product over the flattened parameters
     */
    public static MatVec hessian(List<Var> parameters, Supplier<Var> loss) {
        return v -> {
            Var l = loss.get();
            List<Var> grads = Var.gradients(l, parameters, true);
            Var dot = null;
            int offset = 0;
            for (int p = 0; p < parameters.size(); p++) {
                Tensor value = parameters.get(p).value();
                int size = (int) value.shape().size();
                float[] slice = new float[size];
                for (int i = 0; i < size; i++) {
                    slice[i] = (float) v[offset + i];
                }
                offset += size;
                Var term = grads.get(p).mul(Tensor.of(value.shape(), slice)).sum();
                dot = dot == null ? term : dot.add(term);
            }
            List<Var> hv = Var.gradients(dot, parameters, false);
            double[] out = new double[offset];
            int o = 0;
            for (int p = 0; p < parameters.size(); p++) {
                Var g = hv.get(p);
                int size = (int) parameters.get(p).value().shape().size();
                if (g != null) {
                    double[] gv = g.value().toDoubleArray();
                    System.arraycopy(gv, 0, out, o, size);
                }
                o += size;
            }
            for (Var p : parameters) {
                p.zeroGrad();
            }
            return out;
        };
    }

    private static double[] rademacher(int n, SplittableRandom r) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.nextBoolean() ? 1 : -1;
        }
        return v;
    }
}
