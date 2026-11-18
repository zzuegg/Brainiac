package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * DropEdge (Rong et al., 2020, <a href="https://arxiv.org/abs/1907.10903">arXiv:1907.10903</a>):
 * remove a random fraction of the (undirected) edges at every training step. Fewer edges
 * slow down over-smoothing in deep stacks and act as a data augmentation, like dropout on
 * the graph; at evaluation the full graph is used.
 */
public final class DropEdge {

    private final double rate;
    private final RandomGenerator random;

    /**
     * Creates the augmentation.
     *
     * @param rate the fraction of edges removed per step
     * @param random the coin
     */
    public DropEdge(double rate, RandomGenerator random) {
        this.rate = rate;
        this.random = random;
    }

    /**
     * @param adjacency {@code [n, n]}, symmetric
     * @param training whether to drop (evaluation returns the input)
     * @return the adjacency with dropped edges while training
     */
    public Tensor apply(Tensor adjacency, boolean training) {
        return training && rate > 0 ? drop(adjacency, rate, random) : adjacency;
    }

    /**
     * Removes each undirected edge with probability {@code rate}, keeping symmetry.
     *
     * @param adjacency {@code [n, n]}, symmetric
     * @param rate the drop probability
     * @param random the coin
     * @return a new adjacency
     */
    public static Tensor drop(Tensor adjacency, double rate, RandomGenerator random) {
        int n = adjacency.shape().dim(0);
        float[] a = adjacency.toArray();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (a[i * n + j] != 0 && random.nextDouble() < rate) {
                    a[i * n + j] = 0;
                    a[j * n + i] = 0;
                }
            }
        }
        return Tensor.of(Shape.of(n, n), a);
    }
}
