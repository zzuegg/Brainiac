package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Entropic optimal transport by Sinkhorn iterations (Cuturi 2013,
 * <a href="https://arxiv.org/abs/1306.0895">arXiv:1306.0895</a>) and the Sinkhorn divergence
 * (Feydy et al. 2019, <a href="https://arxiv.org/abs/1810.08278">arXiv:1810.08278</a>).
 *
 * <p>{@code OT_eps(a, b) = min <P, C> + eps KL(P | a b^T)} over couplings {@code P} with
 * marginals {@code a} and {@code b}. Its dual potentials {@code f, g} are found by alternating
 * soft-min updates {@code f_i = -eps log sum_j b_j exp((g_j - C_ij) / eps)} (and symmetrically
 * for {@code g}): {@link #sinkhorn} runs them in the log domain, stable for any {@code eps};
 * {@link #sinkhornScaling} runs the classic row and column scalings of the Gibbs kernel
 * {@code exp(-C / eps)}, many times faster per iteration when {@code eps} is not tiny. Either
 * way an iteration is {@code O(n m)}, against {@code O(n^3)} for the exact Hungarian solver.
 *
 * <p>The debiased Sinkhorn divergence
 * {@code S_eps = OT_eps(a, b) - 1/2 OT_eps(a, a) - 1/2 OT_eps(b, b)} is non-negative, zero only
 * for equal measures, and interpolates between MMD ({@code eps -> infinity}) and OT
 * ({@code eps -> 0}); {@link #divergenceLoss} makes it a differentiable loss on generated points
 * (its gradient by the envelope theorem, {@code sum_j P_ij 2 (x_i - y_j)}). {@link #pairing}
 * samples a one-to-one-ish minibatch pairing from a coupling, the large-coupling OT-CFM recipe of
 * Klein et al. 2025 (<a href="https://arxiv.org/abs/2506.05526">arXiv:2506.05526</a>).
 */
public final class EntropicTransport {

    /**
     * A solved entropic transport problem.
     *
     * @param f the source potential
     * @param g the target potential
     * @param epsilon the regularisation
     * @param cost the cost matrix {@code [n][m]}
     * @param a the source weights
     * @param b the target weights
     * @param iterations the iterations run
     */
    public record Plan(double[] f, double[] g, double epsilon, double[][] cost, double[] a, double[] b,
            int iterations) {

        /**
         * The coupling {@code P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps)}.
         *
         * @return {@code [n][m]}
         */
        public double[][] coupling() {
            double[][] p = new double[f.length][g.length];
            for (int i = 0; i < f.length; i++) {
                for (int j = 0; j < g.length; j++) {
                    p[i][j] = a[i] * b[j] * Math.exp((f[i] + g[j] - cost[i][j]) / epsilon);
                }
            }
            return p;
        }

        /**
         * The dual value {@code <a, f> + <b, g>}, which equals {@code OT_eps} at convergence.
         *
         * @return the entropic transport cost
         */
        public double value() {
            return MlMatrices.dot(a, f) + MlMatrices.dot(b, g);
        }

        /**
         * The transport cost {@code <P, C>} of the coupling.
         *
         * @return the primal cost without the entropy term
         */
        public double transportCost() {
            double[][] p = coupling();
            double s = 0;
            for (int i = 0; i < f.length; i++) {
                for (int j = 0; j < g.length; j++) {
                    s += p[i][j] * cost[i][j];
                }
            }
            return s;
        }
    }

    private EntropicTransport() {
    }

    /**
     * The squared Euclidean cost matrix.
     *
     * @param x source points {@code [n][d]}
     * @param y target points {@code [m][d]}
     * @return {@code |x_i - y_j|^2}
     */
    public static double[][] squaredEuclidean(double[][] x, double[][] y) {
        double[][] c = new double[x.length][y.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < y.length; j++) {
                c[i][j] = MlMatrices.squaredDistance(x[i], y[j]);
            }
        }
        return c;
    }

    /**
     * Uniform weights.
     *
     * @param n the count
     * @return {@code 1/n} each
     */
    public static double[] uniform(int n) {
        double[] w = new double[n];
        java.util.Arrays.fill(w, 1.0 / n);
        return w;
    }

    /**
     * Log-domain Sinkhorn.
     *
     * @param a source weights (sum 1)
     * @param b target weights (sum 1)
     * @param cost the cost matrix
     * @param epsilon the regularisation
     * @param iterations the maximum iterations
     * @param tolerance stop when the source marginal error falls below this
     * @return the plan
     */
    public static Plan sinkhorn(double[] a, double[] b, double[][] cost, double epsilon, int iterations,
            double tolerance) {
        int n = a.length;
        int m = b.length;
        double[] f = new double[n];
        double[] g = new double[m];
        double[] logA = new double[n];
        double[] logB = new double[m];
        for (int i = 0; i < n; i++) {
            logA[i] = Math.log(a[i]);
        }
        for (int j = 0; j < m; j++) {
            logB[j] = Math.log(b[j]);
        }
        int it = 0;
        double[] tmp = new double[Math.max(n, m)];
        for (; it < iterations; it++) {
            for (int i = 0; i < n; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < m; j++) {
                    tmp[j] = logB[j] + (g[j] - cost[i][j]) / epsilon;
                    max = Math.max(max, tmp[j]);
                }
                double s = 0;
                for (int j = 0; j < m; j++) {
                    s += Math.exp(tmp[j] - max);
                }
                f[i] = -epsilon * (max + Math.log(s));
            }
            double err = 0;
            for (int j = 0; j < m; j++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < n; i++) {
                    tmp[i] = logA[i] + (f[i] - cost[i][j]) / epsilon;
                    max = Math.max(max, tmp[i]);
                }
                double s = 0;
                for (int i = 0; i < n; i++) {
                    s += Math.exp(tmp[i] - max);
                }
                double gNew = -epsilon * (max + Math.log(s));
                err += Math.abs(gNew - g[j]) * b[j];
                g[j] = gNew;
            }
            if (err < tolerance * epsilon) {
                it++;
                break;
            }
        }
        return new Plan(f, g, epsilon, cost, a, b, it);
    }

    /**
     * Sinkhorn by matrix scaling of the Gibbs kernel: fast, but needs {@code C / eps} below
     * about 700 to avoid underflow.
     *
     * @param a source weights
     * @param b target weights
     * @param cost the cost matrix
     * @param epsilon the regularisation
     * @param iterations the iterations
     * @return the plan (potentials {@code eps log u}, {@code eps log v})
     */
    public static Plan sinkhornScaling(double[] a, double[] b, double[][] cost, double epsilon, int iterations) {
        int n = a.length;
        int m = b.length;
        float[][] k = new float[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                k[i][j] = (float) Math.exp(-cost[i][j] / epsilon);
            }
        }
        double[] u = new double[n];
        double[] v = new double[m];
        java.util.Arrays.fill(u, 1);
        java.util.Arrays.fill(v, 1);
        double[] kv = new double[n];
        double[] ktu = new double[m];
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < n; i++) {
                double s = 0;
                float[] row = k[i];
                for (int j = 0; j < m; j++) {
                    s += row[j] * v[j] * b[j];
                }
                kv[i] = s;
                u[i] = 1 / Math.max(s, 1e-300);
            }
            java.util.Arrays.fill(ktu, 0);
            for (int i = 0; i < n; i++) {
                float[] row = k[i];
                double ua = u[i] * a[i];
                for (int j = 0; j < m; j++) {
                    ktu[j] += row[j] * ua;
                }
            }
            for (int j = 0; j < m; j++) {
                v[j] = 1 / Math.max(ktu[j], 1e-300);
            }
        }
        double[] f = new double[n];
        double[] g = new double[m];
        for (int i = 0; i < n; i++) {
            f[i] = epsilon * Math.log(u[i]);
        }
        for (int j = 0; j < m; j++) {
            g[j] = epsilon * Math.log(v[j]);
        }
        return new Plan(f, g, epsilon, cost, a, b, iterations);
    }

    /**
     * {@code OT_eps} between two uniform point clouds with the squared Euclidean cost: the fast
     * scaling iterations when {@code max C / eps} is small enough, the log domain otherwise.
     *
     * @param x one cloud
     * @param y another
     * @param epsilon the regularisation
     * @param iterations the iterations
     * @return the plan
     */
    public static Plan transport(double[][] x, double[][] y, double epsilon, int iterations) {
        double[][] c = squaredEuclidean(x, y);
        double max = 0;
        for (double[] row : c) {
            for (double v : row) {
                max = Math.max(max, v);
            }
        }
        if (max / epsilon < 80) {
            return sinkhornScaling(uniform(x.length), uniform(y.length), c, epsilon, iterations);
        }
        return sinkhorn(uniform(x.length), uniform(y.length), c, epsilon, iterations, 1e-6);
    }

    /**
     * The debiased Sinkhorn divergence between two uniform point clouds.
     *
     * @param x one cloud
     * @param y another
     * @param epsilon the regularisation
     * @param iterations the iterations
     * @return {@code S_eps(x, y)}
     */
    public static double sinkhornDivergence(double[][] x, double[][] y, double epsilon, int iterations) {
        return transport(x, y, epsilon, iterations).value() - 0.5 * transport(x, x, epsilon, iterations).value()
                - 0.5 * transport(y, y, epsilon, iterations).value();
    }

    /**
     * The gradient of {@code S_eps} with respect to the source points.
     *
     * @param x the source cloud {@code [n][d]}
     * @param y the target cloud
     * @param epsilon the regularisation
     * @param iterations the iterations
     * @return {@code dS/dx}, {@code [n][d]}
     */
    public static double[][] divergenceGradient(double[][] x, double[][] y, double epsilon, int iterations) {
        int n = x.length;
        int d = x[0].length;
        double[][] pxy = transport(x, y, epsilon, iterations).coupling();
        double[][] pxx = transport(x, x, epsilon, iterations).coupling();
        double[][] grad = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < y.length; j++) {
                for (int k = 0; k < d; k++) {
                    grad[i][k] += 2 * pxy[i][j] * (x[i][k] - y[j][k]);
                }
            }
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < d; k++) {
                    grad[i][k] -= 2 * pxx[i][j] * (x[i][k] - x[j][k]);
                }
            }
        }
        return grad;
    }

    /**
     * The Sinkhorn divergence to a fixed target cloud as an autograd loss on generated points.
     *
     * @param x generated points {@code [n, d]}
     * @param target the target cloud
     * @param epsilon the regularisation
     * @param iterations the iterations
     * @return the scalar loss
     */
    public static Var divergenceLoss(Var x, double[][] target, double epsilon, int iterations) {
        int n = x.shape().dim(0);
        int d = x.shape().dim(1);
        double[][][] grad = new double[1][][];
        return Var.custom(in -> {
            double[] v = in[0].toDoubleArray();
            double[][] pts = new double[n][d];
            for (int i = 0; i < n; i++) {
                System.arraycopy(v, i * d, pts[i], 0, d);
            }
            grad[0] = divergenceGradient(pts, target, epsilon, iterations);
            return Tensor.scalar((float) sinkhornDivergence(pts, target, epsilon, iterations));
        }, (in, out, g) -> {
            float s = g.item();
            float[] r = new float[n * d];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < d; k++) {
                    r[i * d + k] = (float) (s * grad[0][i][k]);
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(n, d), r)};
        }, x);
    }

    /**
     * Samples a pairing from a coupling: each source row picks a target column with probability
     * proportional to its row of {@code P}.
     *
     * @param plan the plan
     * @param random the random source
     * @return the chosen target index per source point
     */
    public static int[] pairing(Plan plan, SplittableRandom random) {
        double[][] p = plan.coupling();
        int[] out = new int[p.length];
        for (int i = 0; i < p.length; i++) {
            double s = 0;
            for (double v : p[i]) {
                s += v;
            }
            double u = random.nextDouble() * s;
            int j = 0;
            while (j < p[i].length - 1 && u > p[i][j]) {
                u -= p[i][j];
                j++;
            }
            out[i] = j;
        }
        return out;
    }

    /**
     * The barycentric projection of a coupling: each source point's {@code P}-weighted mean of
     * the targets, an estimate of the Monge map at the source points.
     *
     * @param plan the plan
     * @param y the target points
     * @return the mapped source points
     */
    public static double[][] barycentricProjection(Plan plan, double[][] y) {
        double[][] p = plan.coupling();
        int d = y[0].length;
        double[][] out = new double[p.length][d];
        for (int i = 0; i < p.length; i++) {
            double s = 0;
            for (int j = 0; j < y.length; j++) {
                s += p[i][j];
                for (int k = 0; k < d; k++) {
                    out[i][k] += p[i][j] * y[j][k];
                }
            }
            for (int k = 0; k < d; k++) {
                out[i][k] /= s;
            }
        }
        return out;
    }
}
