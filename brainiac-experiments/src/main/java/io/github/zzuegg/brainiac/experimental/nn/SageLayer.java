package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Graphs;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * GraphSAGE with the mean aggregator (Hamilton et al., 2017,
 * <a href="https://arxiv.org/abs/1706.02216">arXiv:1706.02216</a>):
 * {@code h_v' = act(W_self h_v + W_neigh mean_{u in N(v)} h_u + b)}, optionally
 * L2-normalized. Keeping the node's own features apart from its neighbourhood's (rather
 * than averaging them in, as {@link GcnLayer} does) helps on heterophilous graphs, and the
 * layer is inductive: it applies to nodes and graphs never seen in training.
 *
 * <p>Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]} (read as 0/1:
 * non-zero entries are edges). A {@link MultiInputLayer} around the core
 * {@link Graphs#sage}, so it drops into {@code Node.merge} graphs.
 */
public final class SageLayer implements MultiInputLayer {

    private final int out;
    private final Var self;
    private final Var neighbour;
    private final Var bias;
    private final Activation activation;
    private final boolean normalize;

    /**
     * Creates the layer.
     *
     * @param in input features per node
     * @param out output features per node
     * @param activation the activation
     * @param normalize whether to L2-normalize every output row
     * @param random the initialization source
     */
    public SageLayer(int in, int out, Activation activation, boolean normalize, SplittableRandom random) {
        this.out = out;
        float bound = (float) Math.sqrt(6.0 / (in + out));
        self = Var.param(Tensor.uniform(Shape.of(in, out), -bound, bound, random));
        neighbour = Var.param(Tensor.uniform(Shape.of(in, out), -bound, bound, random));
        bias = Var.param(Tensor.zeros(Shape.of(1, out)));
        this.activation = Objects.requireNonNull(activation).bind(out);
        this.normalize = normalize;
    }

    /**
     * @param out output features per node
     * @param activation the activation
     * @return a spec for {@code Node.merge(spec, adjacency)}
     */
    public static MultiInputLayer.Spec spec(int out, Activation activation) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return Shape.of(inputs.get(0).dim(0), out);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new SageLayer(inputs.get(0).dim(-1), out, activation, false, random);
            }
        };
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        Graphs.Graph g = GraphOps.toGraph(x.value(), inputs.get(1).value(), true);
        // without edges, every neighbourhood mean is zero
        Var h = g != null ? Graphs.sage(g, x, self, neighbour, bias)
                : x.matmul(self).add(Var.of(Tensor.zeros(x.shape()).to(x.value().device())).matmul(neighbour)).add(bias);
        h = activation.forward(h, training);
        if (normalize) {
            h = h.div(h.square().sum(1, true).add(1e-8f).sqrt());
        }
        return h;
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(self, neighbour, bias));
        p.addAll(activation.parameters());
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "SageLayer(-> " + out + ")";
    }
}
