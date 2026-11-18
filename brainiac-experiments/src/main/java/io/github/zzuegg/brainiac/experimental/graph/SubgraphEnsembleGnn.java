package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SetEncoder;
import io.github.zzuegg.brainiac.nn.Graphs;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * ESAN, equivariant subgraph aggregation networks, in the DS-GNN form (Bevilacqua et al.,
 * 2022, <a href="https://arxiv.org/abs/2110.02910">arXiv:2110.02910</a>). A graph is
 * represented as a bag of its subgraphs (each with one node or one edge deleted); a
 * shared GIN embeds every subgraph, a sum readout (batch-normalised over all
 * subgraphs of the batch) turns each into a vector, and a
 * DeepSets network ({@link SetEncoder}) pools the bag. Two graphs that 1-WL cannot tell
 * apart (for example circular skip-link graphs, which are all 4-regular) usually have
 * distinguishable <em>bags</em>: deleting a node breaks the regularity, and WL refinement
 * of the damaged graph sees the distances from the hole. The bag is sampled (ESAN's
 * stochastic variant) with {@code bag} subgraphs per graph.
 *
 * <p>The GIN runs on edge lists with the core {@link Graphs#propagate}, so a batch of bags
 * stays sparse. Inputs: node features {@code [n, in]}, the block-diagonal adjacency
 * {@code [n, n]} and the membership {@code [graphs, n]}; output {@code [graphs, out]}.
 */
public final class SubgraphEnsembleGnn implements MultiInputLayer {

    /** How subgraphs are made. */
    public enum Policy {
        /** isolate one node (remove all its edges) */
        NODE_DELETED,
        /** remove one edge */
        EDGE_DELETED
    }

    private final int width;
    private final int out;
    private final Policy policy;
    private final int bag;
    private final RandomGenerator sampler;
    private final List<Var[]> layers = new ArrayList<>();
    private final SetEncoder pool;

    /**
     * Creates the network.
     *
     * @param in input features per node
     * @param width the GIN width
     * @param out output features per graph
     * @param depth GIN layers
     * @param policy the subgraph policy
     * @param bag subgraphs sampled per graph
     * @param random the initialization source (and the subgraph sampler's seed)
     */
    public SubgraphEnsembleGnn(int in, int width, int out, int depth, Policy policy, int bag,
            SplittableRandom random) {
        this.width = width;
        this.out = out;
        this.policy = Objects.requireNonNull(policy);
        this.bag = bag;
        for (int l = 0; l < depth; l++) {
            int i = l == 0 ? in : width;
            layers.add(new Var[] {Var.param(Tensor.zeros(Shape.of(1, 1))), GraphMath.weight(i, width, random),
                GraphMath.bias(width), GraphMath.weight(width, width, random), GraphMath.bias(width)});
        }
        this.pool = new SetEncoder(width, width, out, SetEncoder.Kind.DEEP_SETS, 0, 1, random);
        this.sampler = random.split();
    }

    private Var gin(Var h, Tensor sources, Tensor targets) {
        for (Var[] p : layers) {
            Var self = h.mul(p[0].add(1f));
            Var agg = sources == null ? self : Graphs.propagate(h, sources, targets).add(self);
            h = GraphMath.linear(GraphMath.linear(agg, p[1], p[2]).relu(), p[3], p[4]).relu();
        }
        return h;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        float[] a = inputs.get(1).value().toArray();
        Tensor membership = inputs.get(2).value();
        int n = x.shape().dim(0);
        int graphs = membership.shape().dim(0);
        float[] mem = membership.toArray();
        List<Float> index = new ArrayList<>();
        List<Float> owner = new ArrayList<>();
        List<Float> src = new ArrayList<>();
        List<Float> dst = new ArrayList<>();
        int base = 0;
        for (int c = 0; c < graphs; c++) {
            List<Integer> nodes = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (mem[c * n + i] > 0) {
                    nodes.add(i);
                }
            }
            int m = nodes.size();
            List<int[]> edges = new ArrayList<>();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    if (i != j && a[nodes.get(i) * n + nodes.get(j)] != 0) {
                        edges.add(new int[] {i, j});
                    }
                }
            }
            for (int b = 0; b < bag; b++) {
                int deletedNode = policy == Policy.NODE_DELETED ? sampler.nextInt(m) : -1;
                int[] deletedEdge = policy == Policy.EDGE_DELETED && !edges.isEmpty()
                        ? edges.get(sampler.nextInt(edges.size())) : null;
                for (int i = 0; i < m; i++) {
                    index.add((float) nodes.get(i));
                    owner.add((float) (c * bag + b));
                }
                for (int[] e : edges) {
                    boolean gone = e[0] == deletedNode || e[1] == deletedNode
                            || deletedEdge != null && (e[0] == deletedEdge[0] && e[1] == deletedEdge[1]
                            || e[0] == deletedEdge[1] && e[1] == deletedEdge[0]);
                    if (!gone) {
                        src.add((float) (base + e[1]));
                        dst.add((float) (base + e[0]));
                    }
                }
                base += m;
            }
        }
        Var h = gin(x.gather(0, vector(index)), src.isEmpty() ? null : vector(src), vector(dst));
        // sum readout (injective, as in GIN), then batch-normalised across all subgraphs so the
        // component every 4-regular subgraph shares is removed and the small structural
        // differences carry the signal
        Var sum = h.scatterAdd(0, vector(owner), graphs * bag);
        Var centred = sum.sub(sum.mean(0, true));
        Var readout = centred.div(centred.square().mean(0, true).add(1e-5f).sqrt()).reshape(graphs, bag, width);
        return pool.forward(readout, training);
    }

    private static Tensor vector(List<Float> values) {
        float[] v = new float[values.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = values.get(i);
        }
        return Tensor.of(Shape.of(v.length), v);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(2).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var[] l : layers) {
            p.addAll(List.of(l));
        }
        p.addAll(pool.parameters());
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "SubgraphEnsembleGnn(" + policy + ", bag=" + bag + ", -> " + out + ")";
    }
}
