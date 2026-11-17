package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A graph attention layer (Velickovic et al., 2017, <a href="https://arxiv.org/abs/1710.10903">arXiv:1710.10903</a>):
 * every node attends over its neighbours (and itself) with learned weights
 * {@code alpha_ij = softmax_j(LeakyReLU(a_l . W h_i + a_r . W h_j))} and takes the
 * weighted sum of their transformed features. Several heads run in parallel and are
 * concatenated (hidden layers) or averaged (the output layer). Unlike {@link GcnLayer}'s
 * fixed degree weights, the weights depend on the features, so a node can ignore
 * unhelpful neighbours.
 *
 * <p>Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]}.
 */
public final class GatLayer implements MultiInputLayer {

    private final int heads;
    private final int size;
    private final boolean concat;
    private final Var weight;
    private final Var attnLeft;
    private final Var attnRight;
    private final Var bias;
    private final Activation activation;

    /**
     * Creates the layer.
     *
     * @param in input features per node
     * @param heads attention heads
     * @param size output features per head
     * @param concat whether to concatenate heads ({@code heads * size} outputs) or average them ({@code size})
     * @param activation the activation
     * @param random the initialization source
     */
    public GatLayer(int in, int heads, int size, boolean concat, Activation activation, SplittableRandom random) {
        this.heads = heads;
        this.size = size;
        this.concat = concat;
        float bound = (float) Math.sqrt(6.0 / (in + heads * size));
        weight = Var.param(Tensor.uniform(Shape.of(in, heads * size), -bound, bound, random));
        float ab = (float) Math.sqrt(6.0 / (size + 1));
        attnLeft = Var.param(Tensor.uniform(Shape.of(heads, 1, size), -ab, ab, random));
        attnRight = Var.param(Tensor.uniform(Shape.of(heads, 1, size), -ab, ab, random));
        bias = Var.param(Tensor.zeros(Shape.of(1, concat ? heads * size : size)));
        this.activation = Objects.requireNonNull(activation).bind(concat ? heads * size : size);
    }

    /**
     * @param heads attention heads
     * @param size features per head
     * @param concat whether heads are concatenated (else averaged)
     * @param activation the activation
     * @return a spec for {@code Node.merge(spec, adjacency)}
     */
    public static MultiInputLayer.Spec spec(int heads, int size, boolean concat, Activation activation) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return Shape.of(inputs.get(0).dim(0), concat ? heads * size : size);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new GatLayer(inputs.get(0).dim(-1), heads, size, concat, activation, random);
            }
        };
    }

    /**
     * The attention weights of every head.
     *
     * @param x node features {@code [n, in]}
     * @param adjacency {@code [n, n]}
     * @return {@code [heads, n, n]}, each row summing to 1 over the node's neighbourhood
     */
    public Tensor attention(Tensor x, Tensor adjacency) {
        return weights(Var.of(x).matmul(weight), adjacency).value();
    }

    private Var weights(Var wh, Tensor adjacency) {
        int n = wh.shape().dim(0);
        Var perHead = wh.reshape(n, heads, size).permute(1, 0, 2); // [heads, n, size]
        Var left = perHead.mul(attnLeft).sum(2, true); // [heads, n, 1]
        Var right = perHead.mul(attnRight).sum(2, true).reshape(heads, 1, n); // [heads, 1, n]
        Var logits = GraphOps.leakyRelu(left.add(right), 0.2f).add(GraphOps.attentionBias(adjacency));
        return GraphOps.softmax(logits);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        int n = x.shape().dim(0);
        Var wh = x.matmul(weight);
        Var alpha = weights(wh, inputs.get(1).value());
        Var perHead = wh.reshape(n, heads, size).permute(1, 0, 2);
        Var mixed = alpha.matmul(perHead); // [heads, n, size]
        Var h = concat ? mixed.permute(1, 0, 2).reshape(n, heads * size) : mixed.mean(0, false);
        return activation.forward(h.add(bias), training);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), concat ? heads * size : size);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(weight, attnLeft, attnRight, bias));
        p.addAll(activation.parameters());
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return concat ? heads * size : size;
    }

    @Override
    public String toString() {
        return "GatLayer(" + heads + " heads x " + size + (concat ? ", concat)" : ", mean)");
    }
}
