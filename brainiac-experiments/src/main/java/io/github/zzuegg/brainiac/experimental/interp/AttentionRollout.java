package io.github.zzuegg.brainiac.experimental.interp;

import java.util.ArrayDeque;

/**
 * Attention rollout and attention flow (Abnar and Zuidema, ACL 2020): token-to-input
 * attributions from a transformer's attention maps. Raw attention in one layer is a poor
 * explanation because information also mixes through the residual stream across layers.
 *
 * <ul>
 *   <li>{@link #rollout}: average the heads of each layer, add the identity for the residual
 *       ({@code 0.5 A + 0.5 I}), renormalise the rows and multiply the layers from the input
 *       upwards; the target position's row is its attribution to the inputs.</li>
 *   <li>{@link #flow}: treat the layers as a flow network whose edges carry the
 *       residual-augmented attention weights as capacities, and take the maximum flow from
 *       each input token to the target position at the top (Edmonds-Karp).</li>
 *   <li>{@link #gradientRollout}: rollout of {@code relu(grad * attention)} averaged over
 *       heads (Chefer et al., 2021), so the maps depend on the class explained.</li>
 *   <li>{@link #rawLast}: the head-averaged last-layer attention, the naive baseline.</li>
 * </ul>
 *
 * <p>Attention maps are given as {@code [layers][heads][queries][keys]} with rows summing
 * to one, as an attention layer with a causal or full mask produces them.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2005.00928">arXiv:2005.00928</a>.
 */
public final class AttentionRollout {

    private AttentionRollout() {
    }

    private static double[][] headMean(double[][][] heads) {
        int t = heads[0].length;
        double[][] m = new double[t][t];
        for (double[][] h : heads) {
            for (int i = 0; i < t; i++) {
                for (int j = 0; j < t; j++) {
                    m[i][j] += h[i][j] / heads.length;
                }
            }
        }
        return m;
    }

    private static double[][] withResidual(double[][] a) {
        int t = a.length;
        double[][] out = new double[t][t];
        for (int i = 0; i < t; i++) {
            double s = 0;
            for (int j = 0; j < t; j++) {
                out[i][j] = 0.5 * a[i][j] + (i == j ? 0.5 : 0);
                s += out[i][j];
            }
            for (int j = 0; j < t; j++) {
                out[i][j] = s > 0 ? out[i][j] / s : 0;
            }
        }
        return out;
    }

    /**
     * The head-averaged last-layer attention of a position.
     *
     * @param attention {@code [layers][heads][t][t]}
     * @param target the position explained
     * @return its attribution to each position
     */
    public static double[] rawLast(double[][][][] attention, int target) {
        return headMean(attention[attention.length - 1])[target].clone();
    }

    /**
     * Attention rollout.
     *
     * @param attention {@code [layers][heads][t][t]}
     * @param target the position explained
     * @return its attribution to each input position
     */
    public static double[] rollout(double[][][][] attention, int target) {
        double[][] r = null;
        for (double[][][] layer : attention) {
            double[][] a = withResidual(headMean(layer));
            r = r == null ? a : InterpMath.matmul(a, r);
        }
        return r[target].clone();
    }

    /**
     * Gradient-weighted rollout.
     *
     * @param attention {@code [layers][heads][t][t]}
     * @param gradients the gradient of the explained output with respect to each attention
     *     weight, same shape
     * @param target the position explained
     * @return its attribution to each input position
     */
    public static double[] gradientRollout(double[][][][] attention, double[][][][] gradients, int target) {
        double[][] r = null;
        for (int l = 0; l < attention.length; l++) {
            int heads = attention[l].length;
            int t = attention[l][0].length;
            double[][] m = new double[t][t];
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < t; i++) {
                    for (int j = 0; j < t; j++) {
                        m[i][j] += Math.max(0, attention[l][h][i][j] * gradients[l][h][i][j]) / heads;
                    }
                }
            }
            double[][] a = new double[t][t];
            for (int i = 0; i < t; i++) {
                double s = 0;
                for (int j = 0; j < t; j++) {
                    a[i][j] = m[i][j] + (i == j ? 1 : 0);
                    s += a[i][j];
                }
                for (int j = 0; j < t; j++) {
                    a[i][j] /= s;
                }
            }
            r = r == null ? a : InterpMath.matmul(a, r);
        }
        return r[target].clone();
    }

    /**
     * Attention flow: the maximum flow from each input position to the target position at
     * the top, through the residual-augmented attention graph.
     *
     * @param attention {@code [layers][heads][t][t]}
     * @param target the position explained
     * @return per input position its maximum flow
     */
    public static double[] flow(double[][][][] attention, int target) {
        int layers = attention.length;
        int t = attention[0][0].length;
        int nodes = (layers + 1) * t;
        double[][] cap = new double[nodes][nodes];
        for (int l = 0; l < layers; l++) {
            double[][] a = withResidual(headMean(attention[l]));
            for (int i = 0; i < t; i++) {
                for (int j = 0; j < t; j++) {
                    cap[l * t + j][(l + 1) * t + i] = a[i][j];
                }
            }
        }
        double[] out = new double[t];
        for (int j = 0; j < t; j++) {
            out[j] = maxFlow(cap, j, layers * t + target);
        }
        return out;
    }

    /**
     * Edmonds-Karp maximum flow on a dense capacity matrix (the matrix is not changed).
     *
     * @param capacity {@code [n][n]} edge capacities
     * @param source the source node
     * @param sink the sink node
     * @return the maximum flow value
     */
    public static double maxFlow(double[][] capacity, int source, int sink) {
        int n = capacity.length;
        double[][] residual = new double[n][];
        for (int i = 0; i < n; i++) {
            residual[i] = capacity[i].clone();
        }
        double total = 0;
        int[] parent = new int[n];
        while (true) {
            java.util.Arrays.fill(parent, -1);
            parent[source] = source;
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            queue.add(source);
            while (!queue.isEmpty() && parent[sink] < 0) {
                int u = queue.poll();
                for (int v = 0; v < n; v++) {
                    if (parent[v] < 0 && residual[u][v] > 1e-12) {
                        parent[v] = u;
                        queue.add(v);
                    }
                }
            }
            if (parent[sink] < 0) {
                return total;
            }
            double push = Double.MAX_VALUE;
            for (int v = sink; v != source; v = parent[v]) {
                push = Math.min(push, residual[parent[v]][v]);
            }
            for (int v = sink; v != source; v = parent[v]) {
                residual[parent[v]][v] -= push;
                residual[v][parent[v]] += push;
            }
            total += push;
        }
    }
}
