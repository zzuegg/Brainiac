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
 *       that mean aggregation cannot, since every node looks alike).</li>
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
}
