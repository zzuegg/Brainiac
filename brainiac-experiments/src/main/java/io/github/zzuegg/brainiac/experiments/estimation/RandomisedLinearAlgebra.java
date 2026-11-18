package io.github.zzuegg.brainiac.experiments.estimation;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.randla.HouseholderQr;
import io.github.zzuegg.brainiac.experimental.randla.RandomizedSvd;
import io.github.zzuegg.brainiac.experimental.randla.SketchedLeastSquares;
import io.github.zzuegg.brainiac.experimental.randla.StochasticLanczos;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Randomised numerical linear algebra (12.5.14) against exact dense methods.
 * <ul>
 *   <li>{@link #lowRank}: randomised SVD with 0-2 power iterations on matrices with polynomial
 *       and exponential spectral decay, by its rank-{@code k} error relative to the optimum and its
 *       speed-up over the full {@link ThinSvd};</li>
 *   <li>{@link #leastSquares}: a tall problem with condition number {@code 10^8}, solved by
 *       sketch-and-precondition LSQR with each sketch, plain LSQR and the normal equations,
 *       scored by the relative error against Householder QR;</li>
 *   <li>{@link #logDeterminant}: stochastic Lanczos quadrature of a GP kernel matrix's
 *       log-determinant against Cholesky;</li>
 *   <li>{@link #hessianSpectrum}: the Hessian of a small trained digits MLP, accessed only through
 *       Hessian-vector products: Lanczos' top eigenvalue and SLQ's count of negative
 *       eigenvalues against the exact eigendecomposition, and Hutchinson's trace.</li>
 * </ul>
 */
public final class RandomisedLinearAlgebra {

    private RandomisedLinearAlgebra() {
    }

    /** A matrix {@code U diag(s) V^T} with random orthonormal factors. */
    static double[][] withSpectrum(int m, int n, double[] s, SplittableRandom r) {
        double[][] u = HouseholderQr.of(gaussian(m, n, r)).q();
        double[][] v = HouseholderQr.of(gaussian(n, n, r)).q();
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                double c = u[i][k] * s[k];
                if (c == 0) {
                    continue;
                }
                for (int j = 0; j < n; j++) {
                    a[i][j] += c * v[j][k];
                }
            }
        }
        return a;
    }

    private static double[][] gaussian(int m, int n, SplittableRandom r) {
        double[][] g = new double[m][n];
        for (double[] row : g) {
            for (int j = 0; j < n; j++) {
                row[j] = r.nextGaussian();
            }
        }
        return g;
    }

    /**
     * Randomised SVD against the exact decomposition.
     *
     * @param m rows, for example 800
     * @param n columns, for example 200
     * @param rank the target rank, for example 20
     * @param seed the seed
     * @return "decay / q=..." to {error / optimal error, speed-up over ThinSvd}; "decay / exact"
     *     holds the exact SVD's own error ratio and time (ms)
     */
    public static Map<String, double[]> lowRank(int m, int n, int rank, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        for (String decay : new String[] {"polynomial", "exponential"}) {
            double[] s = new double[n];
            for (int i = 0; i < n; i++) {
                s[i] = decay.equals("polynomial") ? 1.0 / (i + 1) : Math.exp(-i / 8.0);
            }
            double[][] a = withSpectrum(m, n, s, r);
            double optimal = 0;
            for (int i = rank; i < n; i++) {
                optimal += s[i] * s[i];
            }
            optimal = Math.sqrt(optimal);
            long t0 = System.nanoTime();
            ThinSvd exact = ThinSvd.of(a);
            double exactMs = (System.nanoTime() - t0) / 1e6;
            out.put(decay + " / exact", new double[] {RandomizedSvd.residual(a, exact, rank) / optimal, exactMs});
            for (int q = 0; q <= 2; q++) {
                long t1 = System.nanoTime();
                ThinSvd approx = RandomizedSvd.of(a, rank, 10, q, RandomizedSvd.TestMatrix.GAUSSIAN, seed + q);
                double ms = (System.nanoTime() - t1) / 1e6;
                out.put(decay + " / q=" + q, new double[] {RandomizedSvd.residual(a, approx, rank) / optimal,
                    exactMs / ms});
            }
        }
        return out;
    }

    /**
     * Tall ill-conditioned least squares.
     *
     * @param m rows, for example 6000
     * @param n columns, for example 50
     * @param kappa the condition number, for example 1e8
     * @param seed the seed
     * @return method to {relative solution error versus QR, LSQR iterations (0 for direct)}
     */
    public static Map<String, double[]> leastSquares(int m, int n, double kappa, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = Math.pow(kappa, -i / (double) (n - 1));
        }
        double[][] a = withSpectrum(m, n, s, r);
        double[] truth = new double[n];
        for (int j = 0; j < n; j++) {
            truth[j] = r.nextGaussian();
        }
        double[] b = SketchedLeastSquares.Operator.of(a).apply(truth);
        for (int i = 0; i < m; i++) {
            b[i] += 1e-6 * r.nextGaussian();
        }
        double[] ref = HouseholderQr.of(a).solve(b);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (SketchedLeastSquares.Sketch sk : SketchedLeastSquares.Sketch.values()) {
            SketchedLeastSquares.Result res = SketchedLeastSquares.solve(a, b, 4 * n, sk, 1e-14, 100, seed + 1);
            out.put("sketch+LSQR " + sk, new double[] {relError(res.x(), ref), res.iterations()});
        }
        SketchedLeastSquares.Result plain = SketchedLeastSquares.lsqr(SketchedLeastSquares.Operator.of(a), b, 1e-14,
                100);
        out.put("plain LSQR (100 it)", new double[] {relError(plain.x(), ref), plain.iterations()});
        out.put("normal equations", new double[] {relError(SketchedLeastSquares.normalEquations(a, b), ref), 0});
        return out;
    }

    private static double relError(double[] x, double[] ref) {
        double num = 0;
        double den = 0;
        for (int i = 0; i < x.length; i++) {
            num += (x[i] - ref[i]) * (x[i] - ref[i]);
            den += ref[i] * ref[i];
        }
        return Math.sqrt(num / den);
    }

    /**
     * GP kernel log-determinant by SLQ.
     *
     * @param n points, for example 300
     * @param probes probes, for example 10
     * @param steps Lanczos steps, for example 30
     * @param seed the seed
     * @return {exact log det, SLQ estimate, relative error}
     */
    public static double[] logDeterminant(int n, int probes, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][2];
        for (double[] p : x) {
            p[0] = r.nextDouble(-2, 2);
            p[1] = r.nextDouble(-2, 2);
        }
        double[][] k = KernelFunction.rbf(2).gram(x);
        for (int i = 0; i < n; i++) {
            k[i][i] += 0.05;
        }
        double exact = MlMatrices.logDet(MlMatrices.cholesky(k));
        double slq = StochasticLanczos.logDet(StochasticLanczos.MatVec.of(k), n, probes, steps, seed + 1);
        return new double[] {exact, slq, Math.abs(slq - exact) / Math.abs(exact)};
    }

    /**
     * The Hessian spectrum of a trained 64-6-10 digits MLP.
     *
     * @param seed the seed
     * @return "top eigenvalue" to {exact, Lanczos}, "trace" to {exact, Hutchinson},
     *     "negative eigenvalues" to {exact count, SLQ estimate}, "params" to {count}
     */
    public static Map<String, double[]> hessianSpectrum(long seed) {
        ToyTasks.Data data = ToyTasks.digits(10, 20, 8, 0.1, seed);
        SplittableRandom r = new SplittableRandom(seed);
        Var w1 = Var.param(Tensor.normal(Shape.of(64, 6), 0, 0.15f, r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(1, 6)));
        Var w2 = Var.param(Tensor.normal(Shape.of(6, 10), 0, 0.4f, r));
        Var b2 = Var.param(Tensor.zeros(Shape.of(1, 10)));
        List<Var> ps = List.of(w1, b1, w2, b2);
        Tensor xs = data.inputs();
        Tensor ids = data.ids();
        java.util.function.Supplier<Var> loss = () -> {
            Var h = Var.of(xs).matmul(w1).add(b1).tanh();
            return Functional.crossEntropy(h.matmul(w2).add(b2), ids);
        };
        Optimizer opt = Optimizer.adam(0.02f);
        for (int s = 0; s < 150; s++) {
            loss.get().backward();
            opt.step(ps);
        }
        for (Var p : ps) {
            p.zeroGrad();
        }
        StochasticLanczos.MatVec hv = StochasticLanczos.hessian(ps, loss);
        int n = 64 * 6 + 6 + 60 + 10;
        double[][] h = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] e = new double[n];
            e[i] = 1;
            h[i] = hv.apply(e);
        }
        for (int i = 0; i < n; i++) {   // symmetrise float round-off
            for (int j = i + 1; j < n; j++) {
                double m = 0.5 * (h[i][j] + h[j][i]);
                h[i][j] = m;
                h[j][i] = m;
            }
        }
        MlMatrices.Eigen eig = MlMatrices.eigh(h);
        double trace = 0;
        for (double v : eig.values()) {
            trace += v;
        }
        double[] start = new double[n];
        for (int i = 0; i < n; i++) {
            start[i] = r.nextGaussian();
        }
        StochasticLanczos.Quadrature q = StochasticLanczos.lanczos(hv, start, 30);
        double top = Double.NEGATIVE_INFINITY;
        for (double v : q.nodes()) {
            top = Math.max(top, v);
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("params", new double[] {n});
        out.put("top eigenvalue", new double[] {eig.values()[0], top});
        int negative = 0;
        for (double v : eig.values()) {
            negative += v < 0 ? 1 : 0;
        }
        out.put("trace", new double[] {trace, StochasticLanczos.hutchinson(hv, n, 10, seed + 3)});
        out.put("negative eigenvalues", new double[] {negative,
            StochasticLanczos.trace(hv, n, x -> x < 0 ? 1 : 0, 10, 30, seed + 4)});
        return out;
    }
}
