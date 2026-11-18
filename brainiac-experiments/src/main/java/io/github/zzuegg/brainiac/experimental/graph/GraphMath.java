package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/** Shared dense-graph helpers for the graph blocks (package-private). */
final class GraphMath {

    private GraphMath() {
    }

    /** A Glorot-uniform weight {@code [in, out]}. */
    static Var weight(int in, int out, RandomGenerator random) {
        float bound = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -bound, bound, random));
    }

    /** A zero bias {@code [1, out]}. */
    static Var bias(int out) {
        return Var.param(Tensor.zeros(Shape.of(1, out)));
    }

    /** {@code x W + b}. */
    static Var linear(Var x, Var w, Var b) {
        return x.matmul(w).add(b);
    }

    /** The adjacency values, row-major. */
    static float[] values(Tensor adjacency) {
        return adjacency.toArray();
    }

    /**
     * The per-feature maximum over each node's neighbours ({@code adjacency[i][j] != 0}),
     * 0 for a node without neighbours.
     */
    static Var neighbourMax(Var x, Tensor adjacency) {
        int n = x.shape().dim(0);
        int f = x.shape().dim(1);
        float[] a = adjacency.toArray();
        float[] pen = new float[n * n];
        float[] has = new float[n];
        for (int i = 0; i < n * n; i++) {
            if (a[i] != 0) {
                has[i / n] = 1;
            } else {
                pen[i] = -1e9f;
            }
        }
        Var all = x.reshape(1, n, f).add(Var.of(Tensor.of(Shape.of(n, n, 1), pen)));
        return all.max(1, false).mul(Tensor.of(Shape.of(n, 1), has));
    }

    /** The normalised adjacency {@code D^-1/2 A D^-1/2} without self loops (isolated rows are 0). */
    static Tensor symmetricNormalize(Tensor adjacency) {
        int n = adjacency.shape().dim(0);
        float[] a = adjacency.toArray();
        float[] out = new float[n * n];
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d[i] += a[i * n + j] != 0 && i != j ? 1 : 0;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (a[i * n + j] != 0 && i != j) {
                    out[i * n + j] = (float) (1 / Math.sqrt(d[i] * d[j]));
                }
            }
        }
        return Tensor.of(Shape.of(n, n), out);
    }

    /** Per-node layer normalisation without learned affine parameters. */
    static Var layerNorm(Var x) {
        Var c = x.sub(x.mean(1, true));
        return c.div(c.square().mean(1, true).add(1e-5f).sqrt());
    }

    /** The element-wise non-zero pattern of {@code adjacency}, off the diagonal, as 0/1 floats. */
    static float[] binary(Tensor adjacency) {
        int n = adjacency.shape().dim(0);
        float[] a = adjacency.toArray();
        float[] b = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            b[i] = a[i] != 0 && i / n != i % n ? 1 : 0;
        }
        return b;
    }
}
