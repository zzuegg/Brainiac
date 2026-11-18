package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Top-k node pooling: keep the {@code ceil(ratio * n_g)} highest-scoring nodes of every
 * graph and the edges among them, and gate the kept features by {@code tanh(score)} so
 * the score receives a gradient.
 * <ul>
 *   <li>{@link Score#PROJECTION}: gPool (Gao and Ji, 2019, <a href="https://arxiv.org/abs/1905.05178">arXiv:1905.05178</a>),
 *       score {@code y = X p / |p|}, from the node features only;</li>
 *   <li>{@link Score#GNN}: SAGPool (Lee et al., 2019, <a href="https://arxiv.org/abs/1904.08082">arXiv:1904.08082</a>),
 *       score {@code y = S X theta} from a one-channel graph convolution, so the choice also
 *       sees the structure around a node.</li>
 * </ul>
 * Works per graph on a block-diagonal batch; the kept nodes keep their graph membership.
 */
public final class TopKPooling {

    /** How nodes are scored. */
    public enum Score {
        /** a learned projection of the features (gPool) */
        PROJECTION,
        /** a one-channel graph convolution (SAGPool) */
        GNN
    }

    private final double ratio;
    private final Score score;
    private final Var projection;

    /**
     * A pooled batch.
     *
     * @param features {@code [k, f]}, gated by {@code tanh(score)}
     * @param adjacency {@code [k, k]}
     * @param membership {@code [graphs, k]}
     * @param kept the original index of every kept node
     */
    public record Pooled(Var features, Tensor adjacency, Tensor membership, int[] kept) {
    }

    /**
     * Creates the pooling.
     *
     * @param features node feature width
     * @param ratio the fraction of nodes kept per graph
     * @param score the scoring rule
     * @param random the initialization source
     */
    public TopKPooling(int features, double ratio, Score score, SplittableRandom random) {
        this.ratio = ratio;
        this.score = Objects.requireNonNull(score);
        this.projection = GraphMath.weight(features, 1, random);
    }

    /**
     * Pools a batch.
     *
     * @param x node features {@code [n, f]}
     * @param adjacency {@code [n, n]}
     * @param membership {@code [graphs, n]}
     * @return the kept nodes
     */
    public Pooled pool(Var x, Tensor adjacency, Tensor membership) {
        int n = x.shape().dim(0);
        Var y = score == Score.GNN ? Var.of(GraphOps.gcnNormalize(adjacency)).matmul(x).matmul(projection)
                : x.matmul(projection).div(projection.square().sum().sqrt());
        float[] ys = y.value().toArray();
        float[] mem = membership.toArray();
        int g = membership.shape().dim(0);
        List<Integer> kept = new ArrayList<>();
        List<Integer> owners = new ArrayList<>();
        for (int c = 0; c < g; c++) {
            List<Integer> nodes = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (mem[c * n + i] > 0) {
                    nodes.add(i);
                }
            }
            int k = Math.max(1, (int) Math.ceil(ratio * nodes.size()));
            nodes.sort(Comparator.comparingDouble(i -> -ys[i]));
            List<Integer> top = new ArrayList<>(nodes.subList(0, Math.min(k, nodes.size())));
            top.sort(Integer::compare);
            for (int i : top) {
                kept.add(i);
                owners.add(c);
            }
        }
        int k = kept.size();
        float[] idx = new float[k];
        float[] a = adjacency.toArray();
        float[] a2 = new float[k * k];
        float[] mem2 = new float[g * k];
        for (int i = 0; i < k; i++) {
            idx[i] = kept.get(i);
            mem2[owners.get(i) * k + i] = 1;
            for (int j = 0; j < k; j++) {
                a2[i * k + j] = a[kept.get(i) * n + kept.get(j)];
            }
        }
        Tensor index = Tensor.of(Shape.of(k), idx);
        Var gated = x.gather(0, index).mul(y.gather(0, index).tanh());
        return new Pooled(gated, Tensor.of(Shape.of(k, k), a2), Tensor.of(Shape.of(g, k), mem2),
                kept.stream().mapToInt(Integer::intValue).toArray());
    }

    /** @return the score parameters */
    public List<Var> parameters() {
        return List.of(projection);
    }

    @Override
    public String toString() {
        return "TopKPooling(" + score + ", ratio=" + ratio + ")";
    }
}
