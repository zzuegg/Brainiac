package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The graph isomorphism network layer (Xu et al., 2018, <a href="https://arxiv.org/abs/1810.00826">arXiv:1810.00826</a>):
 * {@code h_v' = MLP((1 + eps) h_v + sum_{u in N(v)} h_u)} with a learnable {@code eps}
 * and a two-layer ReLU MLP. Sum aggregation (unlike mean or max) is injective on
 * multisets, which makes GIN as discriminative as the Weisfeiler-Lehman test: the
 * strongest message-passing layer for telling graphs apart, hence the usual choice for
 * graph classification (with a sum readout, see {@link GraphReadout}).
 *
 * <p>Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]}.
 */
public final class GinLayer implements MultiInputLayer {

    private final int out;
    private final Var eps;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    /**
     * Creates the layer.
     *
     * @param in input features per node
     * @param hidden the MLP's hidden width
     * @param out output features per node
     * @param random the initialization source
     */
    public GinLayer(int in, int hidden, int out, SplittableRandom random) {
        this.out = out;
        eps = Var.param(Tensor.zeros(Shape.of(1, 1)));
        w1 = Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(2.0 / in), random));
        b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        w2 = Var.param(Tensor.normal(Shape.of(hidden, out), 0, (float) Math.sqrt(2.0 / hidden), random));
        b2 = Var.param(Tensor.zeros(Shape.of(1, out)));
    }

    /**
     * @param hidden the MLP's hidden width
     * @param out output features per node
     * @return a spec for {@code Node.merge(spec, adjacency)}
     */
    public static MultiInputLayer.Spec spec(int hidden, int out) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return Shape.of(inputs.get(0).dim(0), out);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new GinLayer(inputs.get(0).dim(-1), hidden, out, random);
            }
        };
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        Var agg = Var.of(inputs.get(1).value()).matmul(x).add(x.mul(eps.add(1f)));
        return agg.matmul(w1).add(b1).relu().matmul(w2).add(b2);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        return List.of(eps, w1, b1, w2, b2);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "GinLayer(-> " + out + ")";
    }
}
