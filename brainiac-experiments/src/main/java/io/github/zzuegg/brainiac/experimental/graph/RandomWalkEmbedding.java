package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.experimental.llm.StaticWordEmbeddings;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Shallow node embeddings from random walks: DeepWalk (Perozzi et al., 2014,
 * <a href="https://arxiv.org/abs/1403.6652">arXiv:1403.6652</a>) and node2vec (Grover and
 * Leskovec, 2016, <a href="https://arxiv.org/abs/1607.00653">arXiv:1607.00653</a>). Walks
 * are treated as sentences and nodes as words, and skip-gram with negative sampling (the
 * trainer of {@link StaticWordEmbeddings}) learns a vector per node such that nodes that
 * co-occur on walks get similar vectors.
 *
 * <p>DeepWalk walks uniformly. node2vec biases each step from {@code v} (having arrived
 * from {@code t}) to a neighbour {@code x} by {@code 1/p} if {@code x = t} (return),
 * {@code 1} if {@code x} is also a neighbour of {@code t} (stay local, BFS-like) and
 * {@code 1/q} otherwise (move outward, DFS-like): small {@code q} explores communities,
 * large {@code q} structural roles.
 */
public final class RandomWalkEmbedding {

    private final float[][] vectors;

    private RandomWalkEmbedding(float[][] vectors) {
        this.vectors = vectors;
    }

    /**
     * Samples biased second-order random walks.
     *
     * @param adjacency {@code [n, n]}
     * @param walksPerNode walks started at every node
     * @param length nodes per walk
     * @param p the return parameter (1 for DeepWalk)
     * @param q the in-out parameter (1 for DeepWalk)
     * @param seed the seed
     * @return the walks, as node ids
     */
    public static int[][] walks(Tensor adjacency, int walksPerNode, int length, double p, double q, long seed) {
        int n = adjacency.shape().dim(0);
        float[] a = GraphMath.binary(adjacency);
        int[][] nbr = new int[n][];
        for (int i = 0; i < n; i++) {
            List<Integer> l = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (a[i * n + j] != 0) {
                    l.add(j);
                }
            }
            nbr[i] = l.stream().mapToInt(Integer::intValue).toArray();
        }
        SplittableRandom r = new SplittableRandom(seed);
        int[][] out = new int[walksPerNode * n][];
        int w = 0;
        for (int rep = 0; rep < walksPerNode; rep++) {
            for (int start = 0; start < n; start++) {
                int[] walk = new int[length];
                walk[0] = start;
                int len = 1;
                while (len < length && nbr[walk[len - 1]].length > 0) {
                    int v = walk[len - 1];
                    int[] cand = nbr[v];
                    if (len == 1 || p == 1 && q == 1) {
                        walk[len++] = cand[r.nextInt(cand.length)];
                        continue;
                    }
                    int t = walk[len - 2];
                    double[] weight = new double[cand.length];
                    double total = 0;
                    for (int k = 0; k < cand.length; k++) {
                        int x = cand[k];
                        weight[k] = x == t ? 1 / p : a[t * n + x] != 0 ? 1 : 1 / q;
                        total += weight[k];
                    }
                    double u = r.nextDouble() * total;
                    int k = 0;
                    while (k < cand.length - 1 && (u -= weight[k]) > 0) {
                        k++;
                    }
                    walk[len++] = cand[k];
                }
                out[w++] = java.util.Arrays.copyOf(walk, len);
            }
        }
        return out;
    }

    /**
     * Trains node2vec embeddings (DeepWalk with {@code p = q = 1}).
     *
     * @param adjacency {@code [n, n]}
     * @param dim the embedding size
     * @param walksPerNode walks per node
     * @param length walk length
     * @param p the return parameter
     * @param q the in-out parameter
     * @param window the skip-gram window
     * @param epochs passes over the walks
     * @param seed the seed
     * @return the embedding
     */
    public static RandomWalkEmbedding train(Tensor adjacency, int dim, int walksPerNode, int length, double p,
            double q, int window, int epochs, long seed) {
        int n = adjacency.shape().dim(0);
        List<String> vocabulary = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vocabulary.add("n" + i);
        }
        int[][] walks = walks(adjacency, walksPerNode, length, p, q, seed);
        StaticWordEmbeddings sgns = StaticWordEmbeddings.train(StaticWordEmbeddings.Kind.SGNS, vocabulary, walks,
                dim, window, epochs, seed + 1);
        float[][] v = new float[n][];
        for (int i = 0; i < n; i++) {
            v[i] = sgns.vector("n" + i).clone();
        }
        return new RandomWalkEmbedding(v);
    }

    /** @return the number of nodes */
    public int nodes() {
        return vectors.length;
    }

    /**
     * @param node a node
     * @return its vector (a copy)
     */
    public float[] vector(int node) {
        return vectors[node].clone();
    }

    /**
     * The link score of two nodes, the dot product of their vectors.
     *
     * @param i a node
     * @param j another node
     * @return the score
     */
    public double score(int i, int j) {
        double s = 0;
        for (int k = 0; k < vectors[i].length; k++) {
            s += vectors[i][k] * vectors[j][k];
        }
        return s;
    }
}
