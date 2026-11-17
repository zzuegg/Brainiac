package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Graph neural network building blocks on {@code gather} and {@code scatterAdd}: message
 * passing, GCN, GraphSAGE and GAT layers as functions of node features and an edge list, and
 * batching of several graphs into one disjoint graph.
 *
 * <p>Edges are directed, from {@code sources[e]} to {@code targets[e]}; add both directions
 * for an undirected graph.
 */
public final class Graphs {

    private Graphs() {
    }

    /**
     * A graph: its node features and its edges.
     *
     * @param nodes node features {@code [n, features]}
     * @param sources per edge the node it leaves, as whole numbers {@code [edges]}
     * @param targets per edge the node it reaches {@code [edges]}
     */
    public record Graph(Tensor nodes, Tensor sources, Tensor targets) {

        /** @return the number of nodes */
        public int size() {
            return nodes.shape().dim(0);
        }

        /** @return the same graph with an edge from every node to itself added */
        public Graph withSelfLoops() {
            Tensor self = Tensor.arange(size()).to(sources.device()).to(sources.dtype());
            return new Graph(nodes, Tensor.concat(0, sources, self), Tensor.concat(0, targets, self));
        }
    }

    /**
     * Several graphs as one: nodes stacked, edge indices shifted, and per node the graph it
     * came from, for pooling one vector per graph with {@link #pool}.
     *
     * @param graph the joined graph
     * @param graphOf per node its graph's index {@code [n]}
     * @param count the number of graphs
     */
    public record Batch(Graph graph, Tensor graphOf, int count) {
    }

    /**
     * Joins graphs into one disjoint graph.
     *
     * @param graphs the graphs, with the same feature width
     * @return the batch
     */
    public static Batch batch(List<Graph> graphs) {
        Tensor[] nodes = new Tensor[graphs.size()];
        Tensor[] sources = new Tensor[graphs.size()];
        Tensor[] targets = new Tensor[graphs.size()];
        Tensor[] owner = new Tensor[graphs.size()];
        int offset = 0;
        for (int i = 0; i < graphs.size(); i++) {
            Graph g = graphs.get(i);
            nodes[i] = g.nodes();
            sources[i] = g.sources().add(offset);
            targets[i] = g.targets().add(offset);
            owner[i] = Tensor.full(Shape.of(g.size()), i, g.nodes().device());
            offset += g.size();
        }
        return new Batch(new Graph(Tensor.concat(0, nodes), Tensor.concat(0, sources), Tensor.concat(0, targets)),
                Tensor.concat(0, owner), graphs.size());
    }

    /**
     * Sums each node's incoming messages: {@code out[v] = sum over edges u -> v of x[u]}.
     *
     * @param x node features {@code [n, features]}
     * @param sources edge sources
     * @param targets edge targets
     * @return {@code [n, features]}
     */
    public static Var propagate(Var x, Tensor sources, Tensor targets) {
        return x.gather(0, sources).scatterAdd(0, targets, x.shape().dim(0));
    }

    /**
     * Each node's in-degree.
     *
     * @param targets edge targets
     * @param nodes the number of nodes
     * @return {@code [n]}
     */
    public static Tensor inDegree(Tensor targets, int nodes) {
        return Tensor.ones(Shape.of(targets.shape().dim(0))).to(targets.device()).scatterAdd(0, targets, nodes);
    }

    /**
     * A graph convolution (Kipf and Welling, 2017):
     * {@code D^-1/2 (A + I) D^-1/2 x W + b}, self loops included.
     *
     * @param g the graph (self loops are added here)
     * @param x node features {@code [n, in]}
     * @param weight {@code [in, out]}
     * @param bias {@code [out]}, or null
     * @return {@code [n, out]}
     */
    public static Var gcn(Graph g, Var x, Var weight, Var bias) {
        Graph s = g.withSelfLoops();
        int n = g.size();
        Tensor norm = inDegree(s.targets(), n).pow(-0.5f).to(x.dtype());
        Tensor edgeScale = norm.gather(0, s.sources()).mul(norm.gather(0, s.targets())).reshape(s.sources().shape().dim(0), 1);
        Var h = x.matmul(weight);
        Var out = h.gather(0, s.sources()).mul(Var.of(edgeScale)).scatterAdd(0, s.targets(), n);
        return bias != null ? out.add(bias) : out;
    }

    /**
     * A GraphSAGE layer with mean aggregation (Hamilton et al., 2017):
     * {@code x W_self + mean of neighbours x W_neighbours + b}.
     *
     * @param g the graph
     * @param x node features {@code [n, in]}
     * @param self {@code [in, out]}
     * @param neighbours {@code [in, out]}
     * @param bias {@code [out]}, or null
     * @return {@code [n, out]}
     */
    public static Var sage(Graph g, Var x, Var self, Var neighbours, Var bias) {
        int n = g.size();
        Tensor deg = inDegree(g.targets(), n).maximum(Tensor.full(Shape.of(), 1f, g.targets().device()))
                .to(x.dtype()).reshape(n, 1);
        Var mean = propagate(x, g.sources(), g.targets()).div(Var.of(deg));
        Var out = x.matmul(self).add(mean.matmul(neighbours));
        return bias != null ? out.add(bias) : out;
    }

    /**
     * A single-head graph attention layer (Velickovic et al., 2018): messages from each
     * node's neighbours (and itself) weighted by a softmax over its incoming edges of
     * {@code leakyRelu(a_src . h_u + a_dst . h_v)}, with {@code h = x W}.
     *
     * @param g the graph (self loops are added here)
     * @param x node features {@code [n, in]}
     * @param weight {@code [in, out]}
     * @param attSource {@code [out, 1]}
     * @param attTarget {@code [out, 1]}
     * @return {@code [n, out]}
     */
    public static Var gat(Graph g, Var x, Var weight, Var attSource, Var attTarget) {
        Graph s = g.withSelfLoops();
        int n = g.size();
        Var h = x.matmul(weight);
        int edges = s.sources().shape().dim(0);
        Var e = h.matmul(attSource).gather(0, s.sources()).add(h.matmul(attTarget).gather(0, s.targets()))
                .reshape(edges).activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.LEAKY_RELU, 0.2f);
        // a softmax per target node; shifting by the overall maximum keeps exp finite
        Var shifted = e.sub(Var.of(e.value().max()));
        Var w = shifted.exp();
        Var sums = w.scatterAdd(0, s.targets(), n).gather(0, s.targets());
        Var alpha = w.div(sums).reshape(edges, 1);
        return h.gather(0, s.sources()).mul(alpha).scatterAdd(0, s.targets(), n);
    }

    /**
     * One vector per graph of a batch: the mean (or sum) of its nodes.
     *
     * @param x node features {@code [n, features]}
     * @param batch the batch the nodes come from
     * @param mean whether to average rather than sum
     * @return {@code [graphs, features]}
     */
    public static Var pool(Var x, Batch batch, boolean mean) {
        Var sum = x.scatterAdd(0, batch.graphOf(), batch.count());
        if (!mean) {
            return sum;
        }
        Tensor counts = Tensor.ones(Shape.of(batch.graphOf().shape().dim(0))).to(batch.graphOf().device())
                .scatterAdd(0, batch.graphOf(), batch.count()).maximum(Tensor.full(Shape.of(), 1f,
                        batch.graphOf().device())).to(x.dtype()).reshape(batch.count(), 1);
        return sum.div(Var.of(counts));
    }
}
