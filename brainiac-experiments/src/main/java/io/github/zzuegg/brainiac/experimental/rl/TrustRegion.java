package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * Natural-gradient machinery shared by trust-region policy methods ({@link Trpo},
 * {@link ConstrainedPolicyOptimization}): flattening parameters and gradients into one
 * vector, Fisher-vector products as Hessian-vector products of the KL divergence (by
 * differentiating its gradient again with {@link Var#gradients(Var, List, boolean)}), and
 * conjugate gradient, which solves {@code F x = g} using only such products.
 *
 * <p>Schulman et al., <a href="https://arxiv.org/abs/1502.05477">arXiv:1502.05477</a>, Appendix C.
 */
public final class TrustRegion {

    private TrustRegion() {
    }

    /**
     * @param params the parameters
     * @return the total number of values
     */
    public static int size(List<Var> params) {
        int n = 0;
        for (Var p : params) {
            n += (int) p.value().size();
        }
        return n;
    }

    /**
     * @param values parameters or gradients
     * @return their values, concatenated
     */
    public static float[] flatten(List<Var> values) {
        float[] out = new float[size(values)];
        int at = 0;
        for (Var p : values) {
            float[] v = p.value().toArray();
            System.arraycopy(v, 0, out, at, v.length);
            at += v.length;
        }
        return out;
    }

    /**
     * Writes a flat vector back into parameters.
     *
     * @param params the parameters
     * @param flat the values, {@link #size} long
     */
    public static void assign(List<Var> params, float[] flat) {
        int at = 0;
        for (Var p : params) {
            Tensor t = p.value();
            int n = (int) t.size();
            float[] v = new float[n];
            System.arraycopy(flat, at, v, 0, n);
            p.assign(Tensor.of(t.shape(), v, t.device()));
            at += n;
        }
    }

    /**
     * Gradient of a scalar, flattened.
     *
     * @param loss the scalar
     * @param params the parameters
     * @return the gradient
     */
    public static float[] gradient(Var loss, List<Var> params) {
        return flatten(Var.gradients(loss, params, false));
    }

    /**
     * Fisher-vector product {@code (F + damping I) v}, with the Fisher matrix taken as the
     * Hessian of the mean KL divergence from the current (frozen) policy, evaluated there.
     *
     * @param kl computes the mean KL from the frozen policy at the current parameters
     * @param params the parameters
     * @param v the vector
     * @param damping added multiple of {@code v}, for stability
     * @return the product
     */
    public static float[] fisherVectorProduct(Supplier<Var> kl, List<Var> params, float[] v, float damping) {
        List<Var> g = Var.gradients(kl.get(), params, true);
        Var gv = null;
        int at = 0;
        for (int i = 0; i < params.size(); i++) {
            Tensor t = params.get(i).value();
            int n = (int) t.size();
            float[] slice = new float[n];
            System.arraycopy(v, at, slice, 0, n);
            at += n;
            Var term = g.get(i).mul(Tensor.of(t.shape(), slice, t.device())).sum();
            gv = gv == null ? term : gv.add(term);
        }
        float[] hv = flatten(Var.gradients(gv, params, false));
        for (int i = 0; i < hv.length; i++) {
            hv[i] += damping * v[i];
        }
        return hv;
    }

    /**
     * Conjugate gradient: approximately solves {@code A x = b} for a symmetric positive
     * definite {@code A} given only products with it.
     *
     * @param product computes {@code A v}
     * @param b the right-hand side
     * @param iterations the most iterations, 10 in TRPO
     * @return {@code x}
     */
    public static float[] conjugateGradient(UnaryOperator<float[]> product, float[] b, int iterations) {
        float[] x = new float[b.length];
        float[] r = b.clone();
        float[] p = b.clone();
        double rr = dot(r, r);
        for (int k = 0; k < iterations && rr > 1e-10; k++) {
            float[] ap = product.apply(p);
            double alpha = rr / Math.max(1e-12, dot(p, ap));
            for (int i = 0; i < x.length; i++) {
                x[i] += (float) (alpha * p[i]);
                r[i] -= (float) (alpha * ap[i]);
            }
            double next = dot(r, r);
            double beta = next / rr;
            for (int i = 0; i < p.length; i++) {
                p[i] = (float) (r[i] + beta * p[i]);
            }
            rr = next;
        }
        return x;
    }

    /**
     * @param a a vector
     * @param b another of the same length
     * @return their dot product
     */
    public static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (double) a[i] * b[i];
        }
        return s;
    }

    /**
     * @param x a vector
     * @param scale the factor of {@code y}
     * @param y another vector of the same length
     * @return {@code x + scale * y}
     */
    public static float[] axpy(float[] x, double scale, float[] y) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) (x[i] + scale * y[i]);
        }
        return out;
    }
}
