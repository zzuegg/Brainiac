package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.Function;

/**
 * Exact curvature matrices and natural-gradient steps for tiny classifiers (up to a few
 * hundred parameters), the diagnostic of Kunstner, Balles and Hennig (2019,
 * <a href="https://arxiv.org/abs/1905.12558">arXiv:1905.12558</a>). For a softmax
 * cross-entropy over {@code n} examples it forms, from the per-example output Jacobians
 * {@code J_i}:
 *
 * <ul>
 *   <li>the true Fisher {@code F = 1/n sum J_i^T (diag p_i - p_i p_i^T) J_i}, labels sampled
 *       from the model in expectation (equal to the Gauss-Newton matrix for this loss);</li>
 *   <li>the empirical Fisher {@code EF = 1/n sum g_i g_i^T} from the per-example gradients on
 *       the <em>true</em> labels, which rescales by squared gradients like Adam and can distort
 *       the step badly;</li>
 *   <li>the Hessian, column by column from exact Hessian-vector products.</li>
 * </ul>
 *
 * <p>Steps use an eigendecomposition, {@code d = -sum v (v.g) / (|lambda| + damping)}, so the
 * indefinite Hessian gives a saddle-free Newton step. The exact directions calibrate the
 * approximations ({@link Kfac}, {@link AdaFisher}, Shampoo, Adam) through
 * {@link CurvatureProducts#cosine}.
 */
public final class ExactNaturalGradient {

    /** Which curvature matrix preconditions the step. */
    public enum Curvature {
        /** The true Fisher. */
        FISHER,
        /** The empirical Fisher. */
        EMPIRICAL_FISHER,
        /** The Hessian (absolute eigenvalues). */
        HESSIAN,
        /** None: plain gradient descent, for reference. */
        IDENTITY
    }

    /**
     * The exact matrices at one point.
     *
     * @param fisher the true Fisher
     * @param empiricalFisher the empirical Fisher
     * @param hessian the Hessian, or {@code null} when not requested
     * @param gradient the mean gradient
     * @param loss the mean cross-entropy
     */
    public record Matrices(double[][] fisher, double[][] empiricalFisher, double[][] hessian, double[] gradient,
            double loss) {

        /**
         * @param kind the curvature
         * @return that matrix ({@code null} for {@link Curvature#IDENTITY})
         */
        public double[][] get(Curvature kind) {
            return switch (kind) {
                case FISHER -> fisher;
                case EMPIRICAL_FISHER -> empiricalFisher;
                case HESSIAN -> hessian;
                case IDENTITY -> null;
            };
        }
    }

    private final Curvature kind;
    private float learningRate;
    private final double damping;

    /**
     * @param kind the curvature that preconditions the steps
     * @param learningRate the step size
     * @param damping added to the eigenvalues' magnitudes
     */
    public ExactNaturalGradient(Curvature kind, float learningRate, double damping) {
        this.kind = kind;
        this.learningRate = learningRate;
        this.damping = damping;
    }

    /**
     * Computes the exact matrices.
     *
     * @param params the parameters
     * @param logits computes {@code [n, classes]} logits from the parameters
     * @param labels {@code [n]} true class ids
     * @param withHessian whether to also form the Hessian (one HVP per parameter)
     * @return the matrices
     */
    public static Matrices compute(List<Var> params, Function<List<Var>, Var> logits, Tensor labels,
            boolean withHessian) {
        Var out = logits.apply(params);
        int n = out.shape().dim(0);
        int c = out.shape().dim(1);
        Tensor p = LearningMath.softmax(out.value());
        float[] pv = p.toArray();
        float[] y = labels.toArray();
        double[][] j = CurvatureProducts.jacobian(out, params);
        int dim = j[0].length;
        double[][] f = new double[dim][dim];
        double[][] ef = new double[dim][dim];
        double[] grad = new double[dim];
        double loss = 0;
        for (int i = 0; i < n; i++) {
            double[] gi = new double[dim];
            loss -= Math.log(Math.max(1e-30, pv[i * c + (int) y[i]])) / n;
            for (int a = 0; a < c; a++) {
                double r = pv[i * c + a] - (a == (int) y[i] ? 1 : 0);
                double[] row = j[i * c + a];
                for (int k = 0; k < dim; k++) {
                    gi[k] += r * row[k];
                }
            }
            // F_i = J^T diag(p) J - (J^T p)(J^T p)^T
            double[] jp = new double[dim];
            for (int a = 0; a < c; a++) {
                double pa = pv[i * c + a];
                double[] row = j[i * c + a];
                for (int k = 0; k < dim; k++) {
                    jp[k] += pa * row[k];
                }
                for (int r = 0; r < dim; r++) {
                    double v = pa * row[r] / n;
                    if (v == 0) {
                        continue;
                    }
                    for (int s = r; s < dim; s++) {
                        f[r][s] += v * row[s];
                    }
                }
            }
            for (int r = 0; r < dim; r++) {
                double jr = jp[r] / n;
                double er = gi[r] / n;
                grad[r] += gi[r] / n;
                for (int s = r; s < dim; s++) {
                    f[r][s] -= jr * jp[s];
                    ef[r][s] += er * gi[s];
                }
            }
        }
        symmetrize(f);
        symmetrize(ef);
        double[][] h = null;
        if (withHessian) {
            h = new double[dim][];
            double[] e = new double[dim];
            for (int k = 0; k < dim; k++) {
                e[k] = 1;
                h[k] = CurvatureProducts.hessianVector(() -> Functional.crossEntropy(logits.apply(params), labels),
                        params, e);
                e[k] = 0;
            }
            for (int r = 0; r < dim; r++) {
                for (int s = 0; s < r; s++) {
                    double m = 0.5 * (h[r][s] + h[s][r]);
                    h[r][s] = m;
                    h[s][r] = m;
                }
            }
        }
        return new Matrices(f, ef, h, grad, loss);
    }

    private static void symmetrize(double[][] m) {
        for (int r = 0; r < m.length; r++) {
            for (int s = 0; s < r; s++) {
                m[r][s] = m[s][r];
            }
        }
    }

    /**
     * The preconditioned descent direction {@code -sum v (v.g) / (|lambda| + damping)}.
     *
     * @param matrix the curvature, or {@code null} for {@code -g}
     * @param gradient the gradient
     * @param damping the damping
     * @return the direction
     */
    public static double[] direction(double[][] matrix, double[] gradient, double damping) {
        if (matrix == null) {
            double[] d = gradient.clone();
            for (int i = 0; i < d.length; i++) {
                d[i] = -d[i];
            }
            return d;
        }
        MlMatrices.Eigen eig = MlMatrices.eigh(matrix);
        int n = gradient.length;
        double[] d = new double[n];
        for (int k = 0; k < n; k++) {
            double[] v = eig.vector(k);
            double coef = CurvatureProducts.dot(v, gradient) / (Math.abs(eig.values()[k]) + damping);
            for (int i = 0; i < n; i++) {
                d[i] -= coef * v[i];
            }
        }
        return d;
    }

    /**
     * Takes one exact step.
     *
     * @param params the parameters
     * @param logits computes the logits from the parameters
     * @param labels the true class ids
     * @return the loss before the step
     */
    public double step(List<Var> params, Function<List<Var>, Var> logits, Tensor labels) {
        Matrices m = compute(params, logits, labels, kind == Curvature.HESSIAN);
        double[] d = direction(m.get(kind), m.gradient(), damping);
        CurvatureProducts.assign(params, CurvatureProducts.axpy(CurvatureProducts.values(params), learningRate, d));
        return m.loss();
    }

    /** @return the step size */
    public float learningRate() {
        return learningRate;
    }

    /** @param learningRate the new step size */
    public void learningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return "ExactNaturalGradient(" + kind + ")";
    }
}
