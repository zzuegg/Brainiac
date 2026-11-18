package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * Variance-reduced stochastic gradient methods for finite sums
 * {@code f(w) = (1/n) sum_i f_i(w)}.
 *
 * <p>Plain SGD's gradient noise does not vanish at the optimum, so it needs decreasing steps and
 * converges sublinearly. The variance-reduced estimators do vanish there, so a constant step
 * converges linearly on strongly convex problems:
 * <ul>
 *   <li>{@link Method#SVRG} (Johnson and Zhang, NeurIPS 2013): a full gradient {@code mu} at a
 *       snapshot {@code w~} every epoch; steps use {@code grad f_i(w) - grad f_i(w~) + mu};</li>
 *   <li>{@link Method#SAGA} (Defazio, Bach and Lacoste-Julien, NeurIPS 2014,
 *       <a href="https://arxiv.org/abs/1407.0202">arXiv:1407.0202</a>): a table of the last gradient
 *       of every example; steps use {@code grad f_i(w) - table_i + mean(table)};</li>
 *   <li>{@link Method#SARAH} (Nguyen et al., ICML 2017, <a href="https://arxiv.org/abs/1703.00102">
 *       arXiv:1703.00102</a>): the recursive estimate {@code v_t = grad f_i(w_t) - grad f_i(w_{t-1}) +
 *       v_{t-1}}, restarted from a full gradient every epoch.</li>
 * </ul>
 * {@link Method#SGD} (optionally with a {@code 1 / (1 + t / n)} decay) and {@link Method#ADAM} are
 * the baselines. Problems are {@link FiniteSum}s on flat vectors: least squares and logistic
 * regression, or the flattened parameters of any autograd model via {@link #ofModel}.
 */
public final class VarianceReduced {

    /** The solvers. */
    public enum Method {
        /** Stochastic gradient descent. */
        SGD,
        /** Stochastic variance-reduced gradient. */
        SVRG,
        /** SAGA. */
        SAGA,
        /** StochAstic Recursive grAdient algoritHm. */
        SARAH,
        /** Adam on single-example gradients. */
        ADAM
    }

    /** A finite sum of per-example losses. */
    public interface FiniteSum {

        /**
         * The number of examples {@code n}.
         *
         * @return {@code n}
         */
        int size();

        /**
         * The parameter dimension.
         *
         * @return the dimension
         */
        int dimension();

        /**
         * The mean loss.
         *
         * @param w the parameters
         * @return {@code f(w)}
         */
        double loss(double[] w);

        /**
         * Writes the gradient of example {@code i}'s loss.
         *
         * @param i the example
         * @param w the parameters
         * @param out receives {@code grad f_i(w)}
         */
        void gradient(int i, double[] w, double[] out);

        /**
         * The full gradient, by default the mean of the example gradients.
         *
         * @param w the parameters
         * @param out receives {@code grad f(w)}
         */
        default void fullGradient(double[] w, double[] out) {
            java.util.Arrays.fill(out, 0);
            double[] g = new double[out.length];
            int n = size();
            for (int i = 0; i < n; i++) {
                gradient(i, w, g);
                for (int j = 0; j < g.length; j++) {
                    out[j] += g[j] / n;
                }
            }
        }
    }

    /**
     * A run.
     *
     * @param w the final parameters
     * @param losses {@code f(w)} after every epoch (index 0 at the start)
     * @param gradientEvaluations single-example gradient evaluations used
     */
    public record Trace(double[] w, double[] losses, long gradientEvaluations) {
    }

    private VarianceReduced() {
    }

    /**
     * Ridge-regularised least squares {@code (1/n) sum 1/2 (a_i . w - b_i)^2 + l2/2 |w|^2}.
     *
     * @param a the design
     * @param b the targets
     * @param l2 the ridge
     * @return the finite sum
     */
    public static FiniteSum leastSquares(double[][] a, double[] b, double l2) {
        return new LinearSum(a, b, l2, false);
    }

    /**
     * {@code l2}-regularised logistic regression with 0/1 labels.
     *
     * @param a the design
     * @param b the labels
     * @param l2 the ridge
     * @return the finite sum
     */
    public static FiniteSum logistic(double[][] a, double[] b, double l2) {
        return new LinearSum(a, b, l2, true);
    }

    private record LinearSum(double[][] a, double[] b, double l2, boolean logistic) implements FiniteSum {
        @Override
        public int size() {
            return a.length;
        }

        @Override
        public int dimension() {
            return a[0].length;
        }

        @Override
        public double loss(double[] w) {
            double s = 0;
            for (int i = 0; i < a.length; i++) {
                double z = dot(a[i], w);
                if (logistic) {
                    // log(1 + e^z) - b z, stably
                    s += (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - b[i] * z;
                } else {
                    s += 0.5 * (z - b[i]) * (z - b[i]);
                }
            }
            return s / a.length + 0.5 * l2 * dot(w, w);
        }

        @Override
        public void gradient(int i, double[] w, double[] out) {
            double z = dot(a[i], w);
            double r = logistic ? 1 / (1 + Math.exp(-z)) - b[i] : z - b[i];
            for (int j = 0; j < out.length; j++) {
                out[j] = r * a[i][j] + l2 * w[j];
            }
        }
    }

    private static double dot(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s += x[i] * y[i];
        }
        return s;
    }

    /**
     * The flattened parameters of an autograd model as a finite sum.
     *
     * @param parameters the parameters, overwritten while evaluating
     * @param exampleLoss builds the loss of example {@code i} from the current parameter values
     * @param n the number of examples
     * @return the finite sum
     */
    public static FiniteSum ofModel(List<Var> parameters, IntFunction<Var> exampleLoss, int n) {
        Objects.requireNonNull(exampleLoss, "exampleLoss");
        List<Var> ps = new ArrayList<>(parameters);
        int dim = HamiltonianMonteCarlo.flatten(ps).length;
        return new FiniteSum() {
            @Override
            public int size() {
                return n;
            }

            @Override
            public int dimension() {
                return dim;
            }

            @Override
            public double loss(double[] w) {
                HamiltonianMonteCarlo.assign(ps, w);
                double s = 0;
                for (int i = 0; i < n; i++) {
                    int idx = i;
                    s += Var.noGrad(() -> exampleLoss.apply(idx)).value().toDoubleArray()[0] / n;
                }
                return s;
            }

            @Override
            public void gradient(int i, double[] w, double[] out) {
                HamiltonianMonteCarlo.assign(ps, w);
                for (Var p : ps) {
                    p.zeroGrad();
                }
                exampleLoss.apply(i).backward();
                int o = 0;
                for (Var p : ps) {
                    int size = (int) p.value().shape().size();
                    if (p.grad() == null) {
                        java.util.Arrays.fill(out, o, o + size, 0);
                    } else {
                        double[] g = p.grad().toDoubleArray();
                        System.arraycopy(g, 0, out, o, size);
                    }
                    o += size;
                }
            }
        };
    }

    /**
     * Runs a solver.
     *
     * @param method the method
     * @param f the problem
     * @param start the initial parameters (not modified)
     * @param step the step size (for ADAM its learning rate)
     * @param epochs passes; an epoch is {@code n} stochastic steps (SVRG and SARAH add one full
     *     gradient per epoch)
     * @param decay for SGD, use {@code step / (1 + t / n)}
     * @param seed the seed
     * @return the trace
     */
    public static Trace run(Method method, FiniteSum f, double[] start, double step, int epochs, boolean decay,
            long seed) {
        Objects.requireNonNull(method, "method");
        int n = f.size();
        int d = f.dimension();
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = start.clone();
        double[] losses = new double[epochs + 1];
        losses[0] = f.loss(w);
        long evals = 0;
        double[] g = new double[d];
        double[] g2 = new double[d];
        double[] mu = new double[d];
        double[] snap = new double[d];
        double[][] table = method == Method.SAGA ? new double[n][d] : null;
        double[] tableMean = new double[d];
        double[] m = new double[d];
        double[] v = new double[d];
        double[] prev = new double[d];
        double[] est = new double[d];
        long t = 0;
        if (method == Method.SAGA) {
            for (int i = 0; i < n; i++) {
                f.gradient(i, w, table[i]);
                for (int j = 0; j < d; j++) {
                    tableMean[j] += table[i][j] / n;
                }
            }
            evals += n;
        }
        for (int e = 0; e < epochs; e++) {
            if (method == Method.SVRG || method == Method.SARAH) {
                System.arraycopy(w, 0, snap, 0, d);
                f.fullGradient(w, mu);
                evals += n;
                if (method == Method.SARAH) {
                    System.arraycopy(mu, 0, est, 0, d);
                    System.arraycopy(w, 0, prev, 0, d);
                    for (int j = 0; j < d; j++) {
                        w[j] -= step * est[j];
                    }
                }
            }
            for (int s = 0; s < n; s++) {
                int i = r.nextInt(n);
                t++;
                switch (method) {
                    case SGD -> {
                        f.gradient(i, w, g);
                        evals++;
                        double lr = decay ? step / (1 + (double) t / n) : step;
                        for (int j = 0; j < d; j++) {
                            w[j] -= lr * g[j];
                        }
                    }
                    case SVRG -> {
                        f.gradient(i, w, g);
                        f.gradient(i, snap, g2);
                        evals += 2;
                        for (int j = 0; j < d; j++) {
                            w[j] -= step * (g[j] - g2[j] + mu[j]);
                        }
                    }
                    case SAGA -> {
                        f.gradient(i, w, g);
                        evals++;
                        for (int j = 0; j < d; j++) {
                            double old = table[i][j];
                            w[j] -= step * (g[j] - old + tableMean[j]);
                            tableMean[j] += (g[j] - old) / n;
                            table[i][j] = g[j];
                        }
                    }
                    case SARAH -> {
                        f.gradient(i, w, g);
                        f.gradient(i, prev, g2);
                        evals += 2;
                        System.arraycopy(w, 0, prev, 0, d);
                        for (int j = 0; j < d; j++) {
                            est[j] = g[j] - g2[j] + est[j];
                            w[j] -= step * est[j];
                        }
                    }
                    case ADAM -> {
                        f.gradient(i, w, g);
                        evals++;
                        for (int j = 0; j < d; j++) {
                            m[j] = 0.9 * m[j] + 0.1 * g[j];
                            v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
                            double mh = m[j] / (1 - Math.pow(0.9, t));
                            double vh = v[j] / (1 - Math.pow(0.999, t));
                            w[j] -= step * mh / (Math.sqrt(vh) + 1e-8);
                        }
                    }
                    default -> throw new IllegalStateException();
                }
            }
            losses[e + 1] = f.loss(w);
        }
        return new Trace(w, losses, evals);
    }
}
