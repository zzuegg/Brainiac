package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * GPR-GNN (Chien et al., 2021, <a href="https://arxiv.org/abs/2006.07988">arXiv:2006.07988</a>):
 * an MLP predicts {@code H = MLP(X)} per node, then {@code Z = sum_{k=0..K} gamma_k S^k H}
 * with <em>learned</em> generalised-PageRank weights {@code gamma_k}. Initialised as
 * personalised PageRank ({@code gamma_k = alpha (1 - alpha)^k}, a low-pass filter); since
 * the weights may turn negative, training can also find a high-pass filter
 * ({@code gamma_1 < 0}: "be unlike your neighbours"), which is what heterophilous graphs
 * need.
 *
 * <p>Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]}; output
 * {@code [n, out]} logits.
 */
public final class GeneralizedPageRank implements MultiInputLayer {

    private final int out;
    private final int hops;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var gamma;

    /**
     * Creates the model.
     *
     * @param in input features
     * @param hidden the MLP's hidden width
     * @param out classes
     * @param hops the propagation steps {@code K}
     * @param alpha the PPR teleport used to initialise {@code gamma}
     * @param random the initialization source
     */
    public GeneralizedPageRank(int in, int hidden, int out, int hops, float alpha, SplittableRandom random) {
        this.out = out;
        this.hops = hops;
        this.w1 = GraphMath.weight(in, hidden, random);
        this.b1 = GraphMath.bias(hidden);
        this.w2 = GraphMath.weight(hidden, out, random);
        this.b2 = GraphMath.bias(out);
        float[] g = new float[hops + 1];
        for (int k = 0; k <= hops; k++) {
            g[k] = (float) (alpha * Math.pow(1 - alpha, k));
        }
        g[hops] = (float) Math.pow(1 - alpha, hops);
        this.gamma = Var.param(Tensor.of(Shape.of(hops + 1), g));
    }

    /** @return the current filter weights {@code gamma_0..gamma_K} */
    public float[] gamma() {
        return gamma.value().toArray();
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var h = GraphMath.linear(GraphMath.linear(inputs.get(0), w1, b1).relu(), w2, b2);
        Var s = Var.of(GraphOps.gcnNormalize(inputs.get(1).value()));
        Var z = h.mul(gamma.slice(0, 0, 1));
        Var p = h;
        for (int k = 1; k <= hops; k++) {
            p = s.matmul(p);
            z = z.add(p.mul(gamma.slice(0, k, k + 1)));
        }
        return z;
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1, b1, w2, b2, gamma);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "GeneralizedPageRank(K=" + hops + " -> " + out + ")";
    }
}
