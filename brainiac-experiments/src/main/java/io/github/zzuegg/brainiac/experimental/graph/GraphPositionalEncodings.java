package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Structural positional encodings for graph transformers and message-passing networks.
 * Attention over all nodes is permutation-invariant, so a node needs to be told where it
 * sits in the graph:
 * <ul>
 *   <li>{@link #laplacian} (LapPE, Dwivedi et al., <a href="https://arxiv.org/abs/2003.00982">arXiv:2003.00982</a>):
 *       the eigenvectors of the {@code k} smallest non-trivial eigenvalues of the
 *       normalised Laplacian {@code I - D^-1/2 A D^-1/2}, the graph analogue of sinusoidal
 *       positions. Eigenvectors are defined up to sign, so training flips their signs at
 *       random ({@link #randomSignFlip});</li>
 *   <li>{@link #randomWalk} (RWSE, Dwivedi et al., <a href="https://arxiv.org/abs/2110.07875">arXiv:2110.07875</a>):
 *       the return probabilities {@code diag(P^t)} of a random walk {@code P = D^-1 A} for
 *       {@code t = 1..K}. They have no sign ambiguity and count short cycles: the
 *       3-step return probability is non-zero exactly on nodes in a triangle, which no
 *       1-WL message-passing network can see.</li>
 * </ul>
 * Both take a block-diagonal batch with its {@code [graphs, n]} membership, and work per
 * graph, so the eigenvectors of different graphs do not mix.
 */
public final class GraphPositionalEncodings {

    private GraphPositionalEncodings() {
    }

    private static List<int[]> components(Tensor membership, int n) {
        List<int[]> out = new ArrayList<>();
        if (membership == null) {
            int[] all = new int[n];
            for (int i = 0; i < n; i++) {
                all[i] = i;
            }
            out.add(all);
            return out;
        }
        int g = membership.shape().dim(0);
        float[] m = membership.toArray();
        for (int c = 0; c < g; c++) {
            int count = 0;
            for (int i = 0; i < n; i++) {
                count += m[c * n + i] > 0 ? 1 : 0;
            }
            int[] nodes = new int[count];
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (m[c * n + i] > 0) {
                    nodes[k++] = i;
                }
            }
            out.add(nodes);
        }
        return out;
    }

    /**
     * Laplacian eigenvector encodings.
     *
     * @param adjacency {@code [n, n]}, possibly block-diagonal
     * @param membership {@code [graphs, n]}, or {@code null} for a single graph
     * @param k the number of eigenvectors (graphs with fewer nodes are zero-padded)
     * @return {@code [n, k]}
     */
    public static Tensor laplacian(Tensor adjacency, Tensor membership, int k) {
        int n = adjacency.shape().dim(0);
        float[] a = adjacency.toArray();
        float[] out = new float[n * k];
        for (int[] nodes : components(membership, n)) {
            int m = nodes.length;
            double[] deg = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    deg[i] += a[nodes[i] * n + nodes[j]] != 0 && i != j ? 1 : 0;
                }
            }
            float[] lap = new float[m * m];
            for (int i = 0; i < m; i++) {
                lap[i * m + i] = deg[i] > 0 ? 1 : 0;
                for (int j = 0; j < m; j++) {
                    if (i != j && a[nodes[i] * n + nodes[j]] != 0) {
                        lap[i * m + j] = (float) (-1 / Math.sqrt(deg[i] * deg[j]));
                    }
                }
            }
            Tensor[] e = Linalg.eigh(Tensor.of(Shape.of(m, m), lap));
            float[] vec = e[1].toArray();
            for (int c = 0; c < k && c + 1 < m; c++) {
                for (int i = 0; i < m; i++) {
                    out[nodes[i] * k + c] = vec[i * m + c + 1];
                }
            }
        }
        return Tensor.of(Shape.of(n, k), out);
    }

    /**
     * Random-walk structural encodings {@code diag(P^t)}, {@code t = 1..steps}.
     *
     * @param adjacency {@code [n, n]}, possibly block-diagonal (walks never leave a graph)
     * @param steps the walk lengths
     * @return {@code [n, steps]}
     */
    public static Tensor randomWalk(Tensor adjacency, int steps) {
        int n = adjacency.shape().dim(0);
        float[] a = adjacency.toArray();
        double[][] p = new double[n][n];
        for (int i = 0; i < n; i++) {
            double d = 0;
            for (int j = 0; j < n; j++) {
                d += a[i * n + j] != 0 && i != j ? 1 : 0;
            }
            for (int j = 0; j < n; j++) {
                p[i][j] = d > 0 && a[i * n + j] != 0 && i != j ? 1 / d : 0;
            }
        }
        float[] out = new float[n * steps];
        double[][] power = p;
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < n; i++) {
                out[i * steps + t] = (float) power[i][i];
            }
            if (t + 1 < steps) {
                double[][] next = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int k = 0; k < n; k++) {
                        double v = power[i][k];
                        if (v != 0) {
                            for (int j = 0; j < n; j++) {
                                next[i][j] += v * p[k][j];
                            }
                        }
                    }
                }
                power = next;
            }
        }
        return Tensor.of(Shape.of(n, steps), out);
    }

    /**
     * Flips the sign of every column with probability 1/2, the LapPE training augmentation.
     *
     * @param encoding {@code [n, k]}
     * @param random the coin
     * @return the encoding with random column signs
     */
    public static Tensor randomSignFlip(Tensor encoding, RandomGenerator random) {
        int k = encoding.shape().dim(1);
        float[] s = new float[k];
        for (int c = 0; c < k; c++) {
            s[c] = random.nextBoolean() ? 1 : -1;
        }
        return encoding.mul(Tensor.of(Shape.of(1, k), s));
    }
}
