package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Small synthetic graphs for testing graph neural networks, and batching of several
 * graphs into one block-diagonal graph.
 *
 * <ul>
 *   <li>{@link #communities}: a stochastic block model, dense inside communities and
 *       sparse across, with noisy node features; node labels are the communities
 *       (node classification, where neighbours carry the signal the features lack);</li>
 *   <li>{@link #treesAndCycles}: random trees versus cycles of the same sizes with
 *       constant node features (graph classification that only structure can solve, and
 *       that mean aggregation cannot, since every node looks alike);</li>
 *   <li>{@link #homophilous}: a block model with a set edge homophily, down to
 *       heterophilous graphs where neighbours tend to differ;</li>
 *   <li>{@link #circularSkipLinks}: 4-regular graphs that the 1-WL test cannot tell
 *       apart (expressivity);</li>
 *   <li>{@link #neighboursMatch}: the tree task of over-squashing;</li>
 *   <li>{@link #triangleCounting}, {@link #communityCount} and {@link #grids}: graph
 *       regression, community-count classification and planar-ish graphs for
 *       generation.</li>
 * </ul>
 */
public final class SyntheticGraphs {

    private SyntheticGraphs() {
    }

    /**
     * One graph.
     *
     * @param nodes the node count
     * @param adjacency {@code nodes x nodes}, row-major, symmetric, 0/1
     * @param features {@code nodes x featureSize}, row-major
     * @param featureSize features per node
     * @param nodeLabels one label per node
     * @param label the graph's label
     */
    public record Graph(int nodes, float[] adjacency, float[] features, int featureSize, int[] nodeLabels,
            int label) {

        /** @return the adjacency as a {@code [n, n]} tensor */
        public Tensor adjacencyTensor() {
            return Tensor.of(Shape.of(nodes, nodes), adjacency);
        }

        /** @return the features as a {@code [n, featureSize]} tensor */
        public Tensor featureTensor() {
            return Tensor.of(Shape.of(nodes, featureSize), features);
        }

        /** @return the number of undirected edges */
        public int edges() {
            int e = 0;
            for (float a : adjacency) {
                e += a > 0 ? 1 : 0;
            }
            return e / 2;
        }
    }

    /**
     * Several graphs as one disconnected graph.
     *
     * @param adjacency the block-diagonal adjacency {@code [n, n]}
     * @param features stacked features {@code [n, f]}
     * @param membership {@code [graphs, n]}, 1 where a node belongs to a graph
     * @param nodeLabels the node labels, stacked
     * @param labels the graph labels
     */
    public record Batch(Tensor adjacency, Tensor features, Tensor membership, int[] nodeLabels, int[] labels) {
    }

    /**
     * A stochastic block model.
     *
     * @param blocks the number of communities
     * @param size nodes per community
     * @param pIn the edge probability inside a community
     * @param pOut the edge probability across communities
     * @param featureNoise the standard deviation of Gaussian noise added to the one-hot
     *     community features (large values hide the community in the features)
     * @param seed the seed
     * @return the graph; features have {@code blocks} columns
     */
    public static Graph communities(int blocks, int size, double pIn, double pOut, double featureNoise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = blocks * size;
        float[] adj = new float[n * n];
        int[] label = new int[n];
        for (int i = 0; i < n; i++) {
            label[i] = i / size;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (r.nextDouble() < (label[i] == label[j] ? pIn : pOut)) {
                    adj[i * n + j] = 1;
                    adj[j * n + i] = 1;
                }
            }
        }
        float[] feat = new float[n * blocks];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < blocks; c++) {
                feat[i * blocks + c] = (float) ((c == label[i] ? 1 : 0) + featureNoise * r.nextGaussian());
            }
        }
        return new Graph(n, adj, feat, blocks, label, 0);
    }

    /**
     * Random trees (label 0) and cycles (label 1) with 5 to 10 nodes and the constant
     * feature 1 on every node.
     *
     * @param count how many graphs, alternating labels
     * @param seed the seed
     * @return the graphs
     */
    public static List<Graph> treesAndCycles(int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Graph> out = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            int n = 5 + r.nextInt(6);
            float[] adj = new float[n * n];
            int label = g % 2;
            for (int i = 1; i < n; i++) {
                int j = label == 0 ? r.nextInt(i) : i - 1;
                adj[i * n + j] = 1;
                adj[j * n + i] = 1;
            }
            if (label == 1) {
                adj[(n - 1) * n] = 1;
                adj[n - 1] = 1;
            }
            float[] feat = new float[n];
            java.util.Arrays.fill(feat, 1);
            out.add(new Graph(n, adj, feat, 1, new int[n], label));
        }
        return out;
    }

    /**
     * Merges graphs into one block-diagonal graph.
     *
     * @param graphs the graphs, all with the same feature size
     * @return the batch
     */
    public static Batch batch(List<Graph> graphs) {
        int total = graphs.stream().mapToInt(Graph::nodes).sum();
        int f = graphs.getFirst().featureSize();
        float[] adj = new float[total * total];
        float[] feat = new float[total * f];
        float[] member = new float[graphs.size() * total];
        int[] nodeLabels = new int[total];
        int[] labels = new int[graphs.size()];
        int offset = 0;
        for (int g = 0; g < graphs.size(); g++) {
            Graph gr = graphs.get(g);
            int n = gr.nodes();
            for (int i = 0; i < n; i++) {
                System.arraycopy(gr.adjacency(), i * n, adj, (offset + i) * total + offset, n);
                member[g * total + offset + i] = 1;
                nodeLabels[offset + i] = gr.nodeLabels()[i];
            }
            System.arraycopy(gr.features(), 0, feat, offset * f, n * f);
            labels[g] = gr.label();
            offset += n;
        }
        return new Batch(Tensor.of(Shape.of(total, total), adj), Tensor.of(Shape.of(total, f), feat),
                Tensor.of(Shape.of(graphs.size(), total), member), nodeLabels, labels);
    }

    /**
     * A stochastic block model with a set edge homophily: the expected fraction of a node's
     * edges that stay inside its class is {@code homophily}, at a fixed expected degree. Low
     * homophily (below {@code 1 / blocks}) makes the graph heterophilous: neighbours tend to
     * carry other labels, so averaging them mixes classes.
     *
     * @param blocks the number of classes
     * @param size nodes per class
     * @param degree the expected degree
     * @param homophily the expected fraction of same-class edges, in {@code [0, 1]}
     * @param featureNoise the noise on the one-hot class features
     * @param seed the seed
     * @return the graph
     */
    public static Graph homophilous(int blocks, int size, double degree, double homophily, double featureNoise,
            long seed) {
        double pIn = Math.min(1, homophily * degree / (size - 1));
        double pOut = Math.min(1, (1 - homophily) * degree / (size * (blocks - 1.0)));
        return communities(blocks, size, pIn, pOut, featureNoise, seed);
    }

    /**
     * The edge homophily of a node-labelled graph: the fraction of edges whose ends share a
     * label.
     *
     * @param g the graph
     * @return the ratio
     */
    public static double edgeHomophily(Graph g) {
        int n = g.nodes();
        int same = 0;
        int all = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (g.adjacency()[i * n + j] > 0) {
                    all++;
                    same += g.nodeLabels()[i] == g.nodeLabels()[j] ? 1 : 0;
                }
            }
        }
        return same / (double) Math.max(1, all);
    }

    /**
     * Circular skip-link graphs (Murphy et al., 2019, <a href="https://arxiv.org/abs/1903.02541">arXiv:1903.02541</a>):
     * a cycle of {@code nodes} nodes plus chords {@code i -- i + skip}. All are 4-regular,
     * so the 1-WL test (and every message-passing GNN with identical node features) gives
     * every graph the same colouring; with {@code nodes = 41} the standard skips
     * {@code {2, 3, 4, 5, 6, 9, 11, 12, 13, 16}} give 10 non-isomorphic classes. Node ids are
     * randomly permuted; features are the constant 1.
     *
     * @param nodes the cycle length
     * @param skips one skip length per class (label = index)
     * @param perClass graphs per class
     * @param seed the seed
     * @return the graphs, classes interleaved
     */
    public static List<Graph> circularSkipLinks(int nodes, int[] skips, int perClass, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Graph> out = new ArrayList<>();
        for (int k = 0; k < perClass; k++) {
            for (int c = 0; c < skips.length; c++) {
                int[] perm = permutation(nodes, r);
                float[] adj = new float[nodes * nodes];
                for (int i = 0; i < nodes; i++) {
                    for (int step : new int[] {1, skips[c]}) {
                        int a = perm[i];
                        int b = perm[(i + step) % nodes];
                        adj[a * nodes + b] = 1;
                        adj[b * nodes + a] = 1;
                    }
                }
                float[] feat = new float[nodes];
                java.util.Arrays.fill(feat, 1);
                out.add(new Graph(nodes, adj, feat, 1, new int[nodes], c));
            }
        }
        return out;
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /**
     * The tree neighbours-match task of Alon and Yahav (2021, <a href="https://arxiv.org/abs/2006.05205">arXiv:2006.05205</a>)
     * for over-squashing: a complete binary tree of the given depth whose
     * {@code L = 2^depth} leaves each carry a distinct key and a random value (one-hot,
     * {@code L} each), and whose root (node 0) carries a query key. The graph label is the
     * value of the leaf whose key matches the query, so the root must receive the
     * information of every leaf through {@code depth} levels of ever fewer edges. Features:
     * {@code [key one-hot (L) || value one-hot (L)]}; inner nodes are zero.
     *
     * @param depth the tree depth
     * @param count how many trees
     * @param seed the seed
     * @return the trees (the root is node 0 of each)
     */
    public static List<Graph> neighboursMatch(int depth, int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int leaves = 1 << depth;
        int n = 2 * leaves - 1;
        int f = 2 * leaves;
        List<Graph> out = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            float[] adj = new float[n * n];
            for (int i = 1; i < n; i++) {
                int parent = (i - 1) / 2;
                adj[i * n + parent] = 1;
                adj[parent * n + i] = 1;
            }
            float[] feat = new float[n * f];
            int[] keys = permutation(leaves, r);
            int query = r.nextInt(leaves);
            int answer = 0;
            for (int l = 0; l < leaves; l++) {
                int node = leaves - 1 + l;
                int value = r.nextInt(leaves);
                feat[node * f + keys[l]] = 1;
                feat[node * f + leaves + value] = 1;
                if (keys[l] == query) {
                    answer = value;
                }
            }
            feat[query] = 1;
            int[] nodeLabels = new int[n];
            nodeLabels[0] = answer;
            out.add(new Graph(n, adj, feat, f, nodeLabels, answer));
        }
        return out;
    }

    /**
     * Erdős–Rényi graphs labelled by their triangle count (a graph regression target that
     * 1-WL message passing cannot compute exactly). Features are the constant 1.
     *
     * @param count how many graphs
     * @param nodes nodes per graph
     * @param p the edge probability
     * @param seed the seed
     * @return the graphs; {@link Graph#label()} is the number of triangles
     */
    public static List<Graph> triangleCounting(int count, int nodes, double p, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Graph> out = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            float[] adj = new float[nodes * nodes];
            for (int i = 0; i < nodes; i++) {
                for (int j = i + 1; j < nodes; j++) {
                    if (r.nextDouble() < p) {
                        adj[i * nodes + j] = 1;
                        adj[j * nodes + i] = 1;
                    }
                }
            }
            int triangles = 0;
            for (int i = 0; i < nodes; i++) {
                for (int j = i + 1; j < nodes; j++) {
                    for (int k = j + 1; k < nodes; k++) {
                        if (adj[i * nodes + j] > 0 && adj[j * nodes + k] > 0 && adj[i * nodes + k] > 0) {
                            triangles++;
                        }
                    }
                }
            }
            float[] feat = new float[nodes];
            java.util.Arrays.fill(feat, 1);
            out.add(new Graph(nodes, adj, feat, 1, new int[nodes], triangles));
        }
        return out;
    }

    /**
     * Graph classification by community count: stochastic block models with two (label
     * 0) or three (label 1) communities of random sizes, about the same node count and
     * density, constant node features. Telling them apart needs the coarse structure, the
     * use case of hierarchical pooling.
     *
     * @param count how many graphs, alternating labels
     * @param seed the seed
     * @return the graphs
     */
    public static List<Graph> communityCount(int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Graph> out = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            int label = g % 2;
            int blocks = 2 + label;
            int n = 12 + r.nextInt(4);
            int[] block = new int[n];
            for (int i = 0; i < n; i++) {
                block[i] = i * blocks / n;
            }
            float[] adj = new float[n * n];
            double pIn = blocks == 2 ? 0.5 : 0.75;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (r.nextDouble() < (block[i] == block[j] ? pIn : 0.06)) {
                        adj[i * n + j] = 1;
                        adj[j * n + i] = 1;
                    }
                }
            }
            float[] feat = new float[n];
            java.util.Arrays.fill(feat, 1);
            out.add(new Graph(n, adj, feat, 1, block, label));
        }
        return out;
    }

    /**
     * Grid graphs {@code rows x cols} with each edge dropped with a small probability, a
     * planar-ish family for graph generation.
     *
     * @param count how many graphs
     * @param rows grid rows
     * @param cols grid columns
     * @param drop the edge-drop probability
     * @param seed the seed
     * @return the graphs (node ids randomly permuted)
     */
    public static List<Graph> grids(int count, int rows, int cols, double drop, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = rows * cols;
        List<Graph> out = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            int[] perm = permutation(n, r);
            float[] adj = new float[n * n];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int v = perm[i * cols + j];
                    if (j + 1 < cols && r.nextDouble() >= drop) {
                        int u = perm[i * cols + j + 1];
                        adj[v * n + u] = 1;
                        adj[u * n + v] = 1;
                    }
                    if (i + 1 < rows && r.nextDouble() >= drop) {
                        int u = perm[(i + 1) * cols + j];
                        adj[v * n + u] = 1;
                        adj[u * n + v] = 1;
                    }
                }
            }
            float[] feat = new float[n];
            java.util.Arrays.fill(feat, 1);
            out.add(new Graph(n, adj, feat, 1, new int[n], 0));
        }
        return out;
    }
}
