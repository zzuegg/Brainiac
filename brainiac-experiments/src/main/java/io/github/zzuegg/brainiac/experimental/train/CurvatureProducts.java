package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * The curvature toolkit shared by the second-order optimizers ({@link HessianFree},
 * {@link ExactNaturalGradient}, {@link LevenbergMarquardt}, {@link AdaHessian}) and the
 * implicit layers: parameters flattened to {@code double[]} vectors, exact Hessian-vector
 * products by double backpropagation (Pearlmutter 1994), Jacobian-vector products by the
 * double-vector-Jacobian trick, Gauss-Newton-vector products {@code J^T H_out J v} (Schraudolph
 * 2002) that never form the Gauss-Newton matrix, full Jacobians for small networks, and a
 * preconditioned, warm-started conjugate-gradient solver.
 *
 * <p>Everything runs on core autograd ({@link Var#gradients(Var, Var, List, boolean)}); the
 * vectors are host-side doubles because the solvers are sequential and the networks small.
 */
public final class CurvatureProducts {

    private CurvatureProducts() {
    }

    /**
     * The curvature of a loss with respect to the network's outputs, as a product with a
     * tangent of the outputs' shape: {@code H_out t}. Positive semi-definite for the losses
     * here, so Gauss-Newton products built on it are too.
     */
    @FunctionalInterface
    public interface OutputCurvature {

        /**
         * @param outputs the outputs' values
         * @param tangent a tangent of the same shape
         * @return {@code H_out tangent}, same shape
         */
        Tensor apply(Tensor outputs, Tensor tangent);

        /**
         * Softmax cross-entropy averaged over the rows of {@code [batch, classes]} logits:
         * {@code H = (diag(p) - p p^T) / batch} per row.
         *
         * @return the curvature
         */
        static OutputCurvature softmaxCrossEntropy() {
            return (logits, t) -> {
                Tensor p = LearningMath.softmax(logits);
                Tensor pt = p.mul(t);
                Tensor out = pt.sub(p.mul(pt.sum(1, true)));
                return out.div(logits.shape().dim(0));
            };
        }

        /**
         * A squared error {@code scale * sum (out - y)^2}, whose curvature is {@code 2 scale I};
         * for a mean over {@code n} values pass {@code 1/n}.
         *
         * @param scale the loss' factor
         * @return the curvature
         */
        static OutputCurvature squaredError(float scale) {
            return (outputs, t) -> t.mul(2 * scale);
        }
    }

    /**
     * @param params the parameters
     * @return their total number of values
     */
    public static int size(List<Var> params) {
        long n = 0;
        for (Var p : params) {
            n += p.shape().size();
        }
        return Math.toIntExact(n);
    }

    /**
     * Concatenates tensors into one vector; {@code null} entries count as zeros of the
     * parameter's shape.
     *
     * @param tensors the tensors
     * @param like the parameters giving the shapes
     * @return the flat vector
     */
    public static double[] flatten(List<Tensor> tensors, List<Var> like) {
        double[] out = new double[size(like)];
        int o = 0;
        for (int i = 0; i < like.size(); i++) {
            int n = Math.toIntExact(like.get(i).shape().size());
            Tensor t = tensors.get(i);
            if (t != null) {
                float[] v = t.toArray();
                for (int j = 0; j < n; j++) {
                    out[o + j] = v[j];
                }
            }
            o += n;
        }
        return out;
    }

    /**
     * Splits a flat vector into tensors shaped like the parameters.
     *
     * @param flat the vector
     * @param like the parameters
     * @return one tensor per parameter
     */
    public static List<Tensor> unflatten(double[] flat, List<Var> like) {
        List<Tensor> out = new ArrayList<>(like.size());
        int o = 0;
        for (Var p : like) {
            int n = Math.toIntExact(p.shape().size());
            float[] v = new float[n];
            for (int j = 0; j < n; j++) {
                v[j] = (float) flat[o + j];
            }
            out.add(Tensor.of(p.shape(), v, p.value().device()));
            o += n;
        }
        return out;
    }

    /**
     * @param params the parameters
     * @return their current values as one vector
     */
    public static double[] values(List<Var> params) {
        List<Tensor> v = new ArrayList<>(params.size());
        for (Var p : params) {
            v.add(p.value());
        }
        return flatten(v, params);
    }

    /**
     * @param params the parameters
     * @return their gradients as one vector, zeros where a parameter has none
     */
    public static double[] grads(List<Var> params) {
        List<Tensor> g = new ArrayList<>(params.size());
        for (Var p : params) {
            g.add(p.grad());
        }
        return flatten(g, params);
    }

    /**
     * Sets the parameters' values from a vector.
     *
     * @param params the parameters
     * @param flat the new values
     */
    public static void assign(List<Var> params, double[] flat) {
        List<Tensor> t = unflatten(flat, params);
        for (int i = 0; i < params.size(); i++) {
            params.get(i).assign(t.get(i));
        }
    }

    /**
     * Stores a vector as the parameters' gradients, for any optimizer to step on.
     *
     * @param params the parameters
     * @param flat the gradients
     */
    public static void setGrads(List<Var> params, double[] flat) {
        List<Tensor> t = unflatten(flat, params);
        for (int i = 0; i < params.size(); i++) {
            params.get(i).setGrad(t.get(i));
        }
    }

    /**
     * The gradient of a scalar without touching the parameters' {@code grad}.
     *
     * @param loss the scalar
     * @param params the parameters
     * @return the flat gradient
     */
    public static double[] gradient(Var loss, List<Var> params) {
        return flatten(tensors(Var.gradients(loss, params, false)), params);
    }

    /**
     * The exact Hessian-vector product {@code H v} of a scalar loss: the gradient of
     * {@code grad(loss) . v}, by double backpropagation.
     *
     * @param loss computes the loss; called once
     * @param params the parameters
     * @param v the vector
     * @return {@code H v}
     */
    public static double[] hessianVector(Supplier<Var> loss, List<Var> params, double[] v) {
        List<Var> g = Var.gradients(loss.get(), params, true);
        List<Tensor> vs = unflatten(v, params);
        Var dot = null;
        for (int i = 0; i < params.size(); i++) {
            Var term = g.get(i).mul(Var.of(vs.get(i))).sum();
            dot = dot == null ? term : dot.add(term);
        }
        return gradient(dot, params);
    }

    /**
     * The Jacobian-vector product {@code J v} of a function of the parameters, of any output
     * shape, with the double-vector-Jacobian trick: {@code u -> J^T u} is linear in a
     * symbolic seed {@code u}, so the gradient of {@code (J^T u) . v} with respect to
     * {@code u} is {@code J v}.
     *
     * @param output the function's output, computed from the parameters
     * @param params the parameters
     * @param v the direction
     * @return {@code J v}, shaped like the output
     */
    public static Tensor jacobianVector(Var output, List<Var> params, double[] v) {
        Var seed = Var.param(Tensor.zeros(output.shape(), output.dtype(), output.value().device()));
        List<Var> vjp = Var.gradients(output, seed, params, true);
        List<Tensor> vs = unflatten(v, params);
        Var dot = null;
        for (int i = 0; i < params.size(); i++) {
            Var term = vjp.get(i).mul(Var.of(vs.get(i))).sum();
            dot = dot == null ? term : dot.add(term);
        }
        if (dot == null) {
            return Tensor.zeros(output.shape());
        }
        return Var.gradients(dot, List.of(seed), false).get(0).value();
    }

    /**
     * The vector-Jacobian product {@code J^T u}.
     *
     * @param output the function's output
     * @param seed {@code u}, shaped like the output
     * @param params the parameters
     * @return the flat {@code J^T u}
     */
    public static double[] vectorJacobian(Var output, Tensor seed, List<Var> params) {
        return flatten(tensors(Var.gradients(output, Var.of(seed), params, false)), params);
    }

    /**
     * The Gauss-Newton-vector product {@code G v = J^T H_out J v}: exact, never formed, and
     * positive semi-definite even where the Hessian is not.
     *
     * @param outputs computes the network's outputs from the parameters; called once
     * @param curvature the loss' curvature in output space
     * @param params the parameters
     * @param v the vector
     * @return {@code G v}
     */
    public static double[] gaussNewtonVector(Function<List<Var>, Var> outputs, OutputCurvature curvature,
            List<Var> params, double[] v) {
        return gaussNewtonVector(outputs.apply(params), curvature, params, v);
    }

    /**
     * {@link #gaussNewtonVector(Function, OutputCurvature, List, double[])} on outputs already
     * computed, so that many products (a CG solve) share one forward graph.
     *
     * @param out the network's outputs, computed from the parameters
     * @param curvature the loss' curvature in output space
     * @param params the parameters
     * @param v the vector
     * @return {@code G v}
     */
    public static double[] gaussNewtonVector(Var out, OutputCurvature curvature, List<Var> params, double[] v) {
        Tensor jv = jacobianVector(out, params, v);
        return vectorJacobian(out, curvature.apply(out.value(), jv), params);
    }

    /**
     * The Jacobian of a function of the parameters, one row per output value, from one
     * vector-Jacobian product per output value: for networks of a few thousand parameters and
     * a few hundred outputs.
     *
     * @param output the function's output
     * @param params the parameters
     * @return {@code [outputs][parameters]}
     */
    public static double[][] jacobian(Var output, List<Var> params) {
        int m = Math.toIntExact(output.shape().size());
        double[][] j = new double[m][];
        float[] seed = new float[m];
        for (int r = 0; r < m; r++) {
            seed[r] = 1;
            j[r] = vectorJacobian(output, Tensor.of(output.shape(), seed.clone()), params);
            seed[r] = 0;
        }
        return j;
    }

    /**
     * The outcome of a conjugate-gradient solve.
     *
     * @param x the solution
     * @param iterations the iterations run
     * @param residual the final residual norm relative to {@code |b|}
     */
    public record Solution(double[] x, int iterations, double residual) {
    }

    /**
     * Solves {@code A x = b} for a symmetric positive (semi-)definite {@code A} given only as
     * a product, with conjugate gradients warm-started from {@code x0} and an optional
     * diagonal preconditioner.
     *
     * @param a the product {@code v -> A v}
     * @param b the right-hand side
     * @param x0 the start, {@code null} for zeros
     * @param preconditioner the diagonal of {@code M^-1}, {@code null} for none
     * @param maxIterations the largest number of iterations
     * @param tolerance stops once {@code |r| <= tolerance |b|}
     * @return the solution
     */
    public static Solution conjugateGradient(UnaryOperator<double[]> a, double[] b, double[] x0,
            double[] preconditioner, int maxIterations, double tolerance) {
        int n = b.length;
        double[] x = x0 == null ? new double[n] : x0.clone();
        double[] r = b.clone();
        if (x0 != null) {
            double[] ax = a.apply(x);
            for (int i = 0; i < n; i++) {
                r[i] -= ax[i];
            }
        }
        double bn = Math.sqrt(dot(b, b));
        if (bn == 0) {
            return new Solution(new double[n], 0, 0);
        }
        double[] z = precondition(preconditioner, r);
        double[] p = z.clone();
        double rz = dot(r, z);
        int it = 0;
        while (it < maxIterations && Math.sqrt(dot(r, r)) > tolerance * bn) {
            double[] ap = a.apply(p);
            double pap = dot(p, ap);
            if (pap <= 0) {
                break;  // not positive definite along p: stop with the current iterate
            }
            double alpha = rz / pap;
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            z = precondition(preconditioner, r);
            double rzNew = dot(r, z);
            double beta = rzNew / rz;
            rz = rzNew;
            for (int i = 0; i < n; i++) {
                p[i] = z[i] + beta * p[i];
            }
            it++;
        }
        return new Solution(x, it, Math.sqrt(dot(r, r)) / bn);
    }

    private static double[] precondition(double[] m, double[] r) {
        if (m == null) {
            return r.clone();
        }
        double[] z = new double[r.length];
        for (int i = 0; i < r.length; i++) {
            z[i] = m[i] * r[i];
        }
        return z;
    }

    /**
     * @param a a vector
     * @param b a vector of the same length
     * @return their dot product
     */
    public static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * @param a a vector
     * @param b a vector of the same length
     * @return their cosine similarity, 0 if either is zero
     */
    public static double cosine(double[] a, double[] b) {
        double n = Math.sqrt(dot(a, a) * dot(b, b));
        return n == 0 ? 0 : dot(a, b) / n;
    }

    /**
     * @param x a vector
     * @param scale the factor of {@code y}
     * @param y a vector of the same length
     * @return {@code x + scale y}, new
     */
    public static double[] axpy(double[] x, double scale, double[] y) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] + scale * y[i];
        }
        return out;
    }

    /**
     * Solves a dense symmetric positive-definite system by Cholesky, with jitter added to the
     * diagonal until it factors.
     *
     * @param a the matrix {@code [n][n]}, unchanged
     * @param b the right-hand side
     * @return {@code A^-1 b}
     */
    public static double[] solveSpd(double[][] a, double[] b) {
        return io.github.zzuegg.brainiac.experimental.ml.MlMatrices.choleskySolve(
                io.github.zzuegg.brainiac.experimental.ml.MlMatrices.cholesky(a), b);
    }

    private static List<Tensor> tensors(List<Var> vars) {
        List<Tensor> out = new ArrayList<>(vars.size());
        for (Var v : vars) {
            out.add(v.value());
        }
        return out;
    }

    /**
     * A tensor of independent Rademacher signs.
     *
     * @param shape the shape
     * @param random the random source
     * @return {@code +-1} values
     */
    public static Tensor rademacher(Shape shape, java.util.random.RandomGenerator random) {
        float[] s = new float[Math.toIntExact(shape.size())];
        for (int i = 0; i < s.length; i++) {
            s[i] = random.nextBoolean() ? 1 : -1;
        }
        return Tensor.of(shape, s);
    }
}
