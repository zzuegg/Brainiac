package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * DiffPool, differentiable hierarchical pooling (Ying et al., 2018, <a href="https://arxiv.org/abs/1806.08804">arXiv:1806.08804</a>).
 * A graph convolution predicts a soft assignment of every node to {@code c} clusters,
 * {@code S = softmax(GCN(Z, A))}, and the graph is coarsened to
 * <pre>
 * X' = S^T Z   [c, f],     A' = S^T A S   [c, c]
 * </pre>
 * so the next layers run on a learned summary graph instead of reading all nodes out at
 * once. Two auxiliary losses keep the assignment meaningful: link prediction
 * {@code |A - S S^T|_F} (nearby nodes share clusters) and the assignment entropy (each
 * node commits to one cluster); {@link #auxiliaryLoss} returns their sum after a forward.
 *
 * <p>Works on a block-diagonal batch: the assignment of graph {@code g} only covers its
 * own nodes, giving {@code [graphs, c, f]} features and {@code [graphs, c, c]} adjacencies.
 * Inputs to {@link #forward}: node embeddings {@code [n, f]}, the adjacency {@code [n, n]}
 * and the membership {@code [graphs, n]}; the output is the pooled features.
 */
public final class DiffPool implements MultiInputLayer {

    private final int features;
    private final int clusters;
    private final Var weight;
    private final Var bias;
    private Var auxiliary;

    /**
     * The coarsened batch.
     *
     * @param features {@code [graphs, clusters, f]}
     * @param adjacency {@code [graphs, clusters, clusters]}
     * @param assignment {@code [n, clusters]}
     * @param auxiliaryLoss the link-prediction plus entropy loss
     */
    public record Pooled(Var features, Var adjacency, Var assignment, Var auxiliaryLoss) {
    }

    /**
     * Creates the pooling.
     *
     * @param features node embedding width
     * @param clusters clusters per graph {@code c}
     * @param random the initialization source
     */
    public DiffPool(int features, int clusters, SplittableRandom random) {
        this.features = features;
        this.clusters = clusters;
        this.weight = GraphMath.weight(features, clusters, random);
        this.bias = GraphMath.bias(clusters);
    }

    /**
     * Pools a batch.
     *
     * @param z node embeddings {@code [n, f]}
     * @param adjacency {@code [n, n]}
     * @param membership {@code [graphs, n]}
     * @return the coarsened graphs and the auxiliary loss
     */
    public Pooled pool(Var z, Tensor adjacency, Tensor membership) {
        int n = z.shape().dim(0);
        int g = membership.shape().dim(0);
        Var s = GraphOps.softmax(Var.of(GraphOps.gcnNormalize(adjacency)).matmul(z).matmul(weight).add(bias));
        Var sg = s.reshape(1, n, clusters).mul(membership.reshape(g, n, 1));
        Var sgT = sg.permute(0, 2, 1);
        Var x = sgT.matmul(z);
        Var a = sgT.matmul(Var.of(adjacency)).matmul(sg);
        Tensor same = membership.transpose().matmul(membership);
        Var link = Var.of(adjacency).sub(s.matmul(s.transpose())).mul(same).square().sum().div((float) n);
        Var entropy = s.mul(s.add(1e-8f).log()).sum().neg().div((float) n);
        auxiliary = link.add(entropy);
        return new Pooled(x, a, s, auxiliary);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        return pool(inputs.get(0), inputs.get(1).value(), inputs.get(2).value()).features();
    }

    @Override
    public Var auxiliaryLoss() {
        return auxiliary;
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(2).dim(0), clusters, features);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "DiffPool(" + features + " -> " + clusters + " clusters)";
    }
}
