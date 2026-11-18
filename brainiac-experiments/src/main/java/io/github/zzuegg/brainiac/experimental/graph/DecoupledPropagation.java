package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Graph models that separate feature transformation from propagation.
 * <ul>
 *   <li>{@link #sgc} (Wu et al., 2019, <a href="https://arxiv.org/abs/1902.07153">arXiv:1902.07153</a>):
 *       Simplifying Graph Convolution drops every non-linearity, so {@code K} GCN layers
 *       collapse to {@code softmax(S^K X W)} with {@code S} the GCN-normalised adjacency.
 *       {@code S^K X} has no parameters and is computed once ({@link #sgcFeatures}; the
 *       layer caches it), and training is plain logistic regression;</li>
 *   <li>{@link #appnp} (Klicpera et al., 2019, <a href="https://arxiv.org/abs/1810.05997">arXiv:1810.05997</a>):
 *       predict {@code H = MLP(X)} per node, then run {@code K} steps of personalised
 *       PageRank {@code Z <- (1 - alpha) S Z + alpha H}. The teleport term anchors each
 *       node to its own prediction, so {@code K = 10..20} hops do not over-smooth, and the
 *       propagation adds no parameters.</li>
 * </ul>
 * Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]}; the output is
 * {@code [n, out]} logits.
 */
public final class DecoupledPropagation implements MultiInputLayer {

    private final boolean personalised;
    private final int hops;
    private final float alpha;
    private final int out;
    private final float dropout;
    private final SplittableRandom masks;
    private final List<Var> weights = new ArrayList<>();
    private Tensor cachedX;
    private Tensor cachedAdjacency;
    private Tensor cachedFeatures;

    private DecoupledPropagation(boolean personalised, int in, int hidden, int out, int hops, float alpha,
            float dropout, SplittableRandom random) {
        this.personalised = personalised;
        this.dropout = dropout;
        this.hops = hops;
        this.alpha = alpha;
        this.out = out;
        if (personalised) {
            weights.add(GraphMath.weight(in, hidden, random));
            weights.add(GraphMath.bias(hidden));
            weights.add(GraphMath.weight(hidden, out, random));
            weights.add(GraphMath.bias(out));
        } else {
            weights.add(GraphMath.weight(in, out, random));
            weights.add(GraphMath.bias(out));
        }
        this.masks = random.split();
    }

    /**
     * Simplifying Graph Convolution.
     *
     * @param in input features
     * @param out classes
     * @param hops the propagation steps {@code K}
     * @param random the initialization source
     * @return the model
     */
    public static DecoupledPropagation sgc(int in, int out, int hops, SplittableRandom random) {
        return new DecoupledPropagation(false, in, 0, out, hops, 0, 0, random);
    }

    /**
     * APPNP: an MLP followed by personalised-PageRank propagation.
     *
     * @param in input features
     * @param hidden the MLP's hidden width
     * @param out classes
     * @param hops the propagation steps {@code K}
     * @param alpha the teleport probability
     * @param dropout the dropout rate on the MLP's input and hidden layer while training
     *     (0.5 in the paper)
     * @param random the initialization source
     * @return the model
     */
    public static DecoupledPropagation appnp(int in, int hidden, int out, int hops, float alpha, float dropout,
            SplittableRandom random) {
        return new DecoupledPropagation(true, in, hidden, out, hops, alpha, dropout, random);
    }

    private Var drop(Var x, boolean training) {
        if (!training || dropout <= 0) {
            return x;
        }
        int size = Math.toIntExact(x.value().size());
        float[] keep = new float[size];
        for (int i = 0; i < size; i++) {
            keep[i] = masks.nextDouble() < dropout ? 0 : 1 / (1 - dropout);
        }
        return x.mul(Tensor.of(x.shape(), keep));
    }

    /**
     * {@code S^K X} with {@code S = D^-1/2 (A + I) D^-1/2}.
     *
     * @param x node features {@code [n, f]}
     * @param adjacency {@code [n, n]}
     * @param hops {@code K}
     * @return {@code [n, f]}
     */
    public static Tensor sgcFeatures(Tensor x, Tensor adjacency, int hops) {
        Tensor s = GraphOps.gcnNormalize(adjacency);
        Tensor h = x;
        for (int k = 0; k < hops; k++) {
            h = s.matmul(h);
        }
        return h;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Tensor x = inputs.get(0).value();
        Tensor adjacency = inputs.get(1).value();
        if (!personalised) {
            if (x != cachedX || adjacency != cachedAdjacency) {
                cachedX = x;
                cachedAdjacency = adjacency;
                cachedFeatures = sgcFeatures(x, adjacency, hops);
            }
            return GraphMath.linear(Var.of(cachedFeatures), weights.get(0), weights.get(1));
        }
        Var hidden = GraphMath.linear(drop(inputs.get(0), training), weights.get(0), weights.get(1)).relu();
        Var h = GraphMath.linear(drop(hidden, training), weights.get(2), weights.get(3));
        Tensor s = GraphOps.gcnNormalize(adjacency);
        Var z = h;
        for (int k = 0; k < hops; k++) {
            z = Var.of(s).matmul(z).mul(1 - alpha).add(h.mul(alpha));
        }
        return z;
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        return List.copyOf(weights);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return personalised ? "APPNP(K=" + hops + ", alpha=" + alpha + ")" : "SGC(K=" + hops + ")";
    }
}
