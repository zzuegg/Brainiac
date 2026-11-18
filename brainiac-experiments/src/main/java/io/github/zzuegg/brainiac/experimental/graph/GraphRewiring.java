package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Graph rewiring against over-squashing (Alon and Yahav, 2021,
 * <a href="https://arxiv.org/abs/2006.05205">arXiv:2006.05205</a>): when the receptive field
 * of a node grows exponentially with depth but must flow through a few edges, the
 * information is squashed into fixed-size vectors. Two fixes that change only the graph:
 * <ul>
 *   <li>{@link #virtualNode} (Gilmer et al., 2017, <a href="https://arxiv.org/abs/1704.01212">arXiv:1704.01212</a>):
 *       one extra node per graph connected to every node, a global relay that puts any
 *       two nodes two hops apart. It is marked by an extra feature column;</li>
 *   <li>{@link #fosr} (Karhadkar et al., 2023, <a href="https://arxiv.org/abs/2210.11790">arXiv:2210.11790</a>):
 *       First-order Spectral Rewiring adds edges one at a time, each time the non-edge
 *       {@code (i, j)} with the most negative {@code x_i x_j / sqrt((1 + d_i)(1 + d_j))},
 *       where {@code x} is the second eigenvector of the normalised adjacency. That is the
 *       edge with the largest first-order increase of the spectral gap (the graph's
 *       bottleneck measure), and the original edges are all kept.</li>
 * </ul>
 */
public final class GraphRewiring {

    private GraphRewiring() {
    }

    /**
     * A rewired batch.
     *
     * @param adjacency {@code [n', n']}
     * @param features {@code [n', f']}
     * @param membership {@code [graphs, n']}
     */
    public record Rewired(Tensor adjacency, Tensor features, Tensor membership) {
    }

    /**
     * Appends one virtual node per graph, connected to all its nodes. Original nodes keep
     * their indices; virtual node {@code g} is node {@code n + g}. Features gain one
     * column, 1 on virtual nodes.
     *
     * @param adjacency {@code [n, n]}
     * @param features {@code [n, f]}
     * @param membership {@code [graphs, n]}
     * @return the rewired batch
     */
    public static Rewired virtualNode(Tensor adjacency, Tensor features, Tensor membership) {
        int n = adjacency.shape().dim(0);
        int f = features.shape().dim(1);
        int g = membership.shape().dim(0);
        int m = n + g;
        float[] a = adjacency.toArray();
        float[] x = features.toArray();
        float[] mem = membership.toArray();
        float[] a2 = new float[m * m];
        float[] x2 = new float[m * (f + 1)];
        float[] mem2 = new float[g * m];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a, i * n, a2, i * m, n);
            System.arraycopy(x, i * f, x2, i * (f + 1), f);
        }
        for (int c = 0; c < g; c++) {
            int v = n + c;
            x2[v * (f + 1) + f] = 1;
            mem2[c * m + v] = 1;
            for (int i = 0; i < n; i++) {
                if (mem[c * n + i] > 0) {
                    a2[v * m + i] = 1;
                    a2[i * m + v] = 1;
                    mem2[c * m + i] = 1;
                }
            }
        }
        return new Rewired(Tensor.of(Shape.of(m, m), a2), Tensor.of(Shape.of(m, f + 1), x2),
                Tensor.of(Shape.of(g, m), mem2));
    }

    /**
     * First-order spectral rewiring of every graph in a batch.
     *
     * @param adjacency {@code [n, n]}, symmetric
     * @param membership {@code [graphs, n]}, or {@code null} for one graph
     * @param edges edges to add per graph
     * @return the adjacency with the new edges
     */
    public static Tensor fosr(Tensor adjacency, Tensor membership, int edges) {
        int n = adjacency.shape().dim(0);
        float[] a = adjacency.toArray();
        List<int[]> groups = new ArrayList<>();
        if (membership == null) {
            int[] all = new int[n];
            for (int i = 0; i < n; i++) {
                all[i] = i;
            }
            groups.add(all);
        } else {
            float[] mem = membership.toArray();
            for (int c = 0; c < membership.shape().dim(0); c++) {
                List<Integer> nodes = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if (mem[c * n + i] > 0) {
                        nodes.add(i);
                    }
                }
                groups.add(nodes.stream().mapToInt(Integer::intValue).toArray());
            }
        }
        for (int[] nodes : groups) {
            for (int e = 0; e < edges; e++) {
                if (!addBestEdge(a, n, nodes)) {
                    break;
                }
            }
        }
        return Tensor.of(Shape.of(n, n), a);
    }

    private static boolean addBestEdge(float[] a, int n, int[] nodes) {
        int m = nodes.length;
        double[] deg = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                deg[i] += i != j && a[nodes[i] * n + nodes[j]] != 0 ? 1 : 0;
            }
        }
        float[] s = new float[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (i != j && a[nodes[i] * n + nodes[j]] != 0) {
                    s[i * m + j] = (float) (1 / Math.sqrt((deg[i] + 1) * (deg[j] + 1)));
                }
            }
            s[i * m + i] = (float) (1 / (deg[i] + 1));
        }
        Tensor[] eig = Linalg.eigh(Tensor.of(Shape.of(m, m), s));
        float[] vec = eig[1].toArray();
        int second = m - 2;
        double best = Double.POSITIVE_INFINITY;
        int bi = -1;
        int bj = -1;
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                if (a[nodes[i] * n + nodes[j]] == 0) {
                    double score = vec[i * m + second] * vec[j * m + second] / Math.sqrt((1 + deg[i]) * (1 + deg[j]));
                    if (score < best) {
                        best = score;
                        bi = i;
                        bj = j;
                    }
                }
            }
        }
        if (bi < 0) {
            return false;
        }
        a[nodes[bi] * n + nodes[bj]] = 1;
        a[nodes[bj] * n + nodes[bi]] = 1;
        return true;
    }
}
