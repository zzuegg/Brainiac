package io.github.zzuegg.brainiac.experimental.multitask;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * The per-task gradient plumbing that gradient-manipulation methods share: one backward pass
 * per task on the shared parameters ({@link #of}, through {@link Var#gradients}), each
 * gradient flattened to one vector, and the combined direction written back into the shared
 * parameters' gradients ({@link #assign}) after a normal backward pass has filled the
 * task-specific heads. Also the small vector helpers the methods use.
 */
public final class TaskGradients {

    private TaskGradients() {
    }

    /**
     * Per-task gradients of the shared parameters.
     *
     * @param losses one scalar loss per task
     * @param shared the shared parameters
     * @return {@code [tasks][total size of shared]}
     */
    public static double[][] of(List<Var> losses, List<Var> shared) {
        double[][] g = new double[losses.size()][];
        for (int t = 0; t < losses.size(); t++) {
            g[t] = flatten(Var.gradients(losses.get(t), shared, false));
        }
        return g;
    }

    /**
     * Concatenates gradient values into one vector.
     *
     * @param grads the gradients
     * @return the flat vector
     */
    public static double[] flatten(List<Var> grads) {
        long size = 0;
        for (Var g : grads) {
            size += g.value().size();
        }
        double[] out = new double[Math.toIntExact(size)];
        int off = 0;
        for (Var g : grads) {
            float[] v = g.value().toArray();
            for (float x : v) {
                out[off++] = x;
            }
        }
        return out;
    }

    /**
     * Replaces the gradients of the shared parameters by a flat direction.
     *
     * @param direction the update direction (to be descended), as {@link #flatten} lays it out
     * @param shared the shared parameters
     */
    public static void assign(double[] direction, List<Var> shared) {
        int off = 0;
        for (Var p : shared) {
            int n = Math.toIntExact(p.value().size());
            float[] v = new float[n];
            for (int i = 0; i < n; i++) {
                v[i] = (float) direction[off + i];
            }
            off += n;
            p.setGrad(Tensor.of(p.shape(), v, p.value().device()));
        }
    }

    /**
     * @param a a vector
     * @param b a vector
     * @return the dot product
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
     * @return its Euclidean norm
     */
    public static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    /**
     * @param a a vector
     * @param b a vector
     * @return their cosine (0 if either is zero)
     */
    public static double cosine(double[] a, double[] b) {
        double d = norm(a) * norm(b);
        return d == 0 ? 0 : dot(a, b) / d;
    }

    /**
     * The weighted sum {@code sum_t w_t g_t}.
     *
     * @param g the vectors
     * @param w the weights
     * @return the combination
     */
    public static double[] combine(double[][] g, double[] w) {
        double[] out = new double[g[0].length];
        for (int t = 0; t < g.length; t++) {
            for (int i = 0; i < out.length; i++) {
                out[i] += w[t] * g[t][i];
            }
        }
        return out;
    }

    /**
     * The Gram matrix {@code G G^T}.
     *
     * @param g the vectors
     * @return {@code [tasks][tasks]}
     */
    public static double[][] gram(double[][] g) {
        int t = g.length;
        double[][] k = new double[t][t];
        for (int i = 0; i < t; i++) {
            for (int j = i; j < t; j++) {
                k[i][j] = dot(g[i], g[j]);
                k[j][i] = k[i][j];
            }
        }
        return k;
    }

    /**
     * The share of task pairs whose gradients conflict (negative cosine).
     *
     * @param g the per-task gradients
     * @return in {@code [0, 1]}
     */
    public static double conflictShare(double[][] g) {
        int pairs = 0;
        int conflicts = 0;
        for (int i = 0; i < g.length; i++) {
            for (int j = i + 1; j < g.length; j++) {
                pairs++;
                conflicts += dot(g[i], g[j]) < 0 ? 1 : 0;
            }
        }
        return pairs == 0 ? 0 : (double) conflicts / pairs;
    }
}
