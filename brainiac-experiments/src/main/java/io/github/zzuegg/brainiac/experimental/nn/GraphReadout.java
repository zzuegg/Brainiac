package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;

/**
 * Pools node embeddings into one vector per graph, for graph-level prediction over a
 * batch of graphs merged into one block-diagonal graph. Inputs: node embeddings
 * {@code [n, f]} and a membership matrix {@code [graphs, n]} (1 where a node belongs to a
 * graph). {@link Mode#SUM} keeps graph size and counts (the injective choice of GIN),
 * {@link Mode#MEAN} is size-invariant, {@link Mode#MAX} picks the strongest node per
 * feature.
 */
public final class GraphReadout implements MultiInputLayer {

    /** How node embeddings are combined. */
    public enum Mode {
        /** sum over the graph's nodes */
        SUM,
        /** mean over the graph's nodes */
        MEAN,
        /** maximum over the graph's nodes, per feature */
        MAX
    }

    private final Mode mode;

    /**
     * Creates the readout.
     *
     * @param mode the pooling
     */
    public GraphReadout(Mode mode) {
        this.mode = Objects.requireNonNull(mode);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var h = inputs.get(0);
        Tensor m = inputs.get(1).value();
        return switch (mode) {
            case SUM -> Var.of(m).matmul(h);
            case MEAN -> Var.of(m.div(m.sum(1, true).maximum(Tensor.full(Shape.of(1, 1), 1f)))).matmul(h);
            case MAX -> {
                int g = m.shape().dim(0);
                int n = m.shape().dim(1);
                // push non-members far down, then take the max over nodes
                Tensor penalty = m.sub(1f).mul(1e9f).reshape(g, n, 1);
                Var all = h.reshape(1, n, h.shape().dim(1)).add(Var.of(penalty));
                yield all.max(1, false);
            }
        };
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(1).dim(0), inputs.get(0).dim(1));
    }

    @Override
    public String toString() {
        return "GraphReadout(" + mode + ")";
    }
}
