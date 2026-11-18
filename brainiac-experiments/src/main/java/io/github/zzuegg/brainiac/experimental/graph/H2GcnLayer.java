package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * H2GCN (Zhu et al., 2020, <a href="https://arxiv.org/abs/2006.11468">arXiv:2006.11468</a>),
 * a node classifier built for heterophily, where neighbours tend to carry other labels and
 * averaging them with the node itself destroys its own signal. Three designs:
 * <ol>
 *   <li>ego and neighbour embeddings are kept apart: the node's own embedding is never
 *       mixed into the aggregation (no self loops), and parts are concatenated, not summed;</li>
 *   <li>1-hop and strict 2-hop neighbourhoods are aggregated separately
 *       ({@code r_k = [S1 r_{k-1} || S2 r_{k-1}]}, each {@code S} symmetric-normalised), since
 *       under heterophily the 2-hop neighbours are often of the node's own class;</li>
 *   <li>every round's representation is concatenated before the classifier
 *       ({@code [r_0 || r_1 || ... || r_K] W}), so the ego embedding {@code r_0} survives.</li>
 * </ol>
 * The only learned parts are the input embedding {@code r_0 = relu(X W_e)} and the
 * classifier.
 *
 * <p>Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]}; output
 * {@code [n, out]} logits.
 */
public final class H2GcnLayer implements MultiInputLayer {

    private final int out;
    private final int rounds;
    private final Var embed;
    private final Var embedBias;
    private final Var classifier;
    private final Var classifierBias;
    private Tensor cachedAdjacency;
    private Tensor[] cachedHops;

    /**
     * Creates the model.
     *
     * @param in input features
     * @param hidden the ego embedding width
     * @param out classes
     * @param rounds the aggregation rounds {@code K}
     * @param random the initialization source
     */
    public H2GcnLayer(int in, int hidden, int out, int rounds, SplittableRandom random) {
        this.out = out;
        this.rounds = rounds;
        this.embed = GraphMath.weight(in, hidden, random);
        this.embedBias = GraphMath.bias(hidden);
        int total = hidden * ((1 << (rounds + 1)) - 1);
        this.classifier = GraphMath.weight(total, out, random);
        this.classifierBias = GraphMath.bias(out);
    }

    /**
     * The normalised 1-hop and strict 2-hop adjacencies (2-hop: reachable in two steps, not
     * in one, not the node itself).
     *
     * @param adjacency {@code [n, n]}
     * @return {S1, S2}
     */
    public static Tensor[] hops(Tensor adjacency) {
        int n = adjacency.shape().dim(0);
        float[] a = GraphMath.binary(adjacency);
        float[] two = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                if (a[i * n + k] != 0) {
                    for (int j = 0; j < n; j++) {
                        if (a[k * n + j] != 0 && j != i && a[i * n + j] == 0) {
                            two[i * n + j] = 1;
                        }
                    }
                }
            }
        }
        return new Tensor[] {GraphMath.symmetricNormalize(Tensor.of(Shape.of(n, n), a)),
            GraphMath.symmetricNormalize(Tensor.of(Shape.of(n, n), two))};
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Tensor adjacency = inputs.get(1).value();
        if (adjacency != cachedAdjacency) {
            cachedAdjacency = adjacency;
            cachedHops = hops(adjacency);
        }
        Var s1 = Var.of(cachedHops[0]);
        Var s2 = Var.of(cachedHops[1]);
        Var r = GraphMath.linear(inputs.get(0), embed, embedBias).relu();
        List<Var> parts = new ArrayList<>(List.of(r));
        for (int k = 0; k < rounds; k++) {
            r = Var.concat(1, s1.matmul(r), s2.matmul(r));
            parts.add(r);
        }
        return GraphMath.linear(Var.concat(1, parts.toArray(Var[]::new)), classifier, classifierBias);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        return List.of(embed, embedBias, classifier, classifierBias);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "H2GcnLayer(K=" + rounds + " -> " + out + ")";
    }
}
