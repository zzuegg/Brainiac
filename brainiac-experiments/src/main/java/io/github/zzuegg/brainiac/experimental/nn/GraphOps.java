package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Adjacency-matrix helpers for the graph layers ({@link GcnLayer}, {@link GatLayer},
 * {@link SageLayer}, {@link GinLayer}). A graph of {@code n} nodes is a dense
 * {@code [n, n]} adjacency (1 for an edge, 0 otherwise, no self-loops needed) next to
 * {@code [n, features]} node features; a batch of graphs is one block-diagonal graph
 * (see {@code GraphBatch} in {@code experimental.data}). Dense adjacency keeps
 * everything plain matrix products, which is the fastest choice for the small graphs
 * these layers are meant for.
 */
public final class GraphOps {

    private GraphOps() {
    }

    /**
     * @param adjacency {@code [n, n]}
     * @return {@code A + I}
     */
    public static Tensor withSelfLoops(Tensor adjacency) {
        int n = adjacency.shape().dim(0);
        float[] eye = new float[n * n];
        for (int i = 0; i < n; i++) {
            eye[i * n + i] = 1;
        }
        return adjacency.maximum(Tensor.of(Shape.of(n, n), eye).to(adjacency.device()));
    }

    /**
     * The symmetric normalization of Kipf and Welling: {@code D^-1/2 (A + I) D^-1/2}.
     *
     * @param adjacency {@code [n, n]}
     * @return the normalized propagation matrix
     */
    public static Tensor gcnNormalize(Tensor adjacency) {
        Tensor a = withSelfLoops(adjacency);
        Tensor inv = a.sum(1, true).sqrt().pow(-1f);
        return a.mul(inv).mul(inv.reshape(1, inv.shape().dim(0)));
    }

    /**
     * Row normalization {@code D^-1 A}: multiplying features by it averages each node's
     * neighbours (isolated nodes get zeros).
     *
     * @param adjacency {@code [n, n]}
     * @return the mean-aggregation matrix
     */
    public static Tensor meanNormalize(Tensor adjacency) {
        Tensor deg = adjacency.sum(1, true).maximum(Tensor.full(Shape.of(1, 1), 1f).to(adjacency.device()));
        return adjacency.div(deg);
    }

    /**
     * An additive attention mask: 0 on edges and the diagonal, a large negative number
     * elsewhere.
     *
     * @param adjacency {@code [n, n]}
     * @return the mask
     */
    public static Tensor attentionBias(Tensor adjacency) {
        return withSelfLoops(adjacency).sub(1f).mul(1e9f);
    }

    /**
     * Leaky ReLU with a fixed negative slope.
     *
     * @param x the input
     * @param slope the negative slope
     * @return {@code x > 0 ? x : slope * x}
     */
    public static Var leakyRelu(Var x, float slope) {
        return Var.where(x.value().greater(Tensor.full(Shape.of(1), 0f).to(x.value().device())), x, x.mul(slope));
    }

    /**
     * Softmax along the last axis.
     *
     * @param logits the logits
     * @return the probabilities
     */
    public static Var softmax(Var logits) {
        int axis = logits.shape().rank() - 1;
        Var e = logits.sub(logits.value().max(axis, true)).exp();
        return e.div(e.sum(axis, true));
    }
}
