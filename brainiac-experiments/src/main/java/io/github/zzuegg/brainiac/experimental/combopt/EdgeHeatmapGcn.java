package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * A supervised non-autoregressive TSP solver (Joshi et al.): a residual gated graph
 * convolutional network over the dense graph predicts, for every edge, the probability that
 * it belongs to the optimal tour, and a beam search turns the heatmap into a tour.
 *
 * <p>Nodes start from their coordinates, edges from their length and a flag for the
 * {@code k} nearest neighbours. Each layer updates edges by
 * {@code e' = e + relu(e A + h_i B + h_j C)} and nodes by
 * {@code h' = h + relu(h U + sum_j sigmoid(e'_ij) * h_j V / sum_j sigmoid(e'_ij))} (edge gates
 * act as a learned soft attention). A final linear map reads edge logits, symmetrised and
 * trained by class-weighted binary cross-entropy against the optimal tour's edges (the
 * weight {@code n/2 - 1} balances the two tour edges per node against the rest).
 *
 * <p>{@link #decode} runs beam search from node 0 over partial tours, ranking by the summed
 * log edge probabilities, and returns the shortest of the finished tours (Joshi et al.'s
 * "beam search with shortest tour" heuristic); beam 1 is greedy.
 *
 * <p>Paper: Joshi, Laurent and Bresson, "An Efficient Graph Convolutional Network Technique for
 * the Travelling Salesman Problem" (2019), <a href="https://arxiv.org/abs/1906.01227">arXiv:1906.01227</a>.
 */
public final class EdgeHeatmapGcn {

    private final int dim;
    private final int knn;
    private final Var nodeIn;
    private final Var edgeIn;
    private final List<Var[]> layers = new ArrayList<>();
    private final Var out;
    private final Var outBias;

    private EdgeHeatmapGcn(int dim, int layerCount, int knn, long seed) {
        java.util.SplittableRandom r = new java.util.SplittableRandom(seed);
        this.dim = dim;
        this.knn = knn;
        this.nodeIn = AttentionModel.matrix(r, 2, dim);
        this.edgeIn = AttentionModel.matrix(r, 2, dim);
        for (int l = 0; l < layerCount; l++) {
            layers.add(new Var[] {AttentionModel.matrix(r, dim, dim), AttentionModel.matrix(r, dim, dim),
                AttentionModel.matrix(r, dim, dim), AttentionModel.matrix(r, dim, dim),
                AttentionModel.matrix(r, dim, dim)});
        }
        this.out = AttentionModel.matrix(r, dim, 1);
        this.outBias = Var.param(Tensor.zeros(Shape.of(1)));
    }

    /**
     * Creates the network.
     *
     * @param dim hidden width
     * @param layers graph convolution layers
     * @param knn the nearest-neighbour count flagged on edges
     * @param seed the seed
     * @return the network
     */
    public static EdgeHeatmapGcn of(int dim, int layers, int knn, long seed) {
        return new EdgeHeatmapGcn(dim, layers, knn, seed);
    }

    /**
     * Edge logits for a batch of same-size instances.
     *
     * @param envs the instances
     * @return {@code [batch, n, n]} symmetric logits
     */
    public Var logits(List<RoutingEnv> envs) {
        int b = envs.size();
        int n = envs.get(0).nodes();
        float[] x = new float[b * n * 2];
        float[] e = new float[b * n * n * 2];
        for (int s = 0; s < b; s++) {
            double[][] c = envs.get(s).coords();
            double[][] d = envs.get(s).distances();
            for (int i = 0; i < n; i++) {
                x[(s * n + i) * 2] = (float) c[i][0];
                x[(s * n + i) * 2 + 1] = (float) c[i][1];
                Integer[] order = new Integer[n];
                for (int j = 0; j < n; j++) {
                    order[j] = j;
                }
                final int ii = i;
                Arrays.sort(order, Comparator.comparingDouble(j -> d[ii][j]));
                for (int j = 0; j < n; j++) {
                    e[((s * n + i) * n + j) * 2] = (float) d[i][j];
                }
                for (int k = 1; k <= Math.min(knn, n - 1); k++) {
                    e[((s * n + i) * n + order[k]) * 2 + 1] = 1;
                }
            }
        }
        Var h = Var.of(Tensor.of(Shape.of(b, n, 2), x)).matmul(nodeIn);
        Var ed = Var.of(Tensor.of(Shape.of(b, n, n, 2), e)).matmul(edgeIn);
        for (Var[] w : layers) {
            Var hi = h.matmul(w[1]).reshape(b, n, 1, dim);
            Var hj = h.matmul(w[2]).reshape(b, 1, n, dim);
            ed = ed.add(ed.matmul(w[0]).add(hi).add(hj).relu());
            Var gate = ed.sigmoid();
            Var msg = gate.mul(h.matmul(w[4]).reshape(b, 1, n, dim)).sum(2, false)
                    .div(gate.sum(2, false).add(1e-6f));
            h = h.add(h.matmul(w[3]).add(msg).relu());
        }
        Var logit = ed.matmul(out).reshape(b, n, n).add(outBias);
        return logit.add(logit.permute(0, 2, 1)).mul(0.5f);
    }

    /**
     * One supervised step.
     *
     * @param envs instances
     * @param tours their optimal tours
     * @param optimizer the optimizer
     * @return the loss
     */
    public float step(List<RoutingEnv> envs, int[][] tours, Optimizer optimizer) {
        int b = envs.size();
        int n = envs.get(0).nodes();
        float[] y = new float[b * n * n];
        float[] w = new float[b * n * n];
        float pos = n / 2f - 1;
        Arrays.fill(w, 1);
        for (int s = 0; s < b; s++) {
            for (int k = 0; k < n; k++) {
                int u = tours[s][k];
                int v = tours[s][(k + 1) % n];
                y[(s * n + u) * n + v] = 1;
                y[(s * n + v) * n + u] = 1;
                w[(s * n + u) * n + v] = pos;
                w[(s * n + v) * n + u] = pos;
            }
        }
        Var z = logits(envs);
        Tensor yt = Tensor.of(Shape.of(b, n, n), y);
        // BCE with logits: softplus(z) - y z
        Var loss = z.softplus().sub(z.mul(yt)).mul(Tensor.of(Shape.of(b, n, n), w)).mean();
        float v = loss.value().item();
        AttentionModel.step(loss, parameters(), optimizer, 1.0);
        return v;
    }

    /**
     * The edge probabilities of one instance.
     *
     * @param env the instance
     * @return {@code [n][n]}
     */
    public double[][] heatmap(RoutingEnv env) {
        int n = env.nodes();
        float[] z = Var.noGrad(() -> logits(List.of(env))).value().toArray();
        double[][] p = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p[i][j] = i == j ? 0 : 1 / (1 + Math.exp(-z[i * n + j]));
            }
        }
        return p;
    }

    /**
     * Beam search over the heatmap.
     *
     * @param env the instance
     * @param beam the beam width (1 is greedy)
     * @return the shortest finished tour
     */
    public int[] decode(RoutingEnv env, int beam) {
        return beamSearch(heatmap(env), env.distances(), beam);
    }

    /**
     * Beam search from node 0 by summed log edge probability, returning the shortest
     * finished tour.
     *
     * @param p edge probabilities
     * @param d distances
     * @param beam the width
     * @return the tour
     */
    public static int[] beamSearch(double[][] p, double[][] d, int beam) {
        int n = p.length;
        record Partial(int[] tour, int len, boolean[] used, double score) {
        }
        List<Partial> live = new ArrayList<>();
        boolean[] u0 = new boolean[n];
        u0[0] = true;
        int[] t0 = new int[n];
        live.add(new Partial(t0, 1, u0, 0));
        for (int step = 1; step < n; step++) {
            List<Partial> next = new ArrayList<>();
            for (Partial pa : live) {
                int cur = pa.tour()[pa.len() - 1];
                for (int j = 0; j < n; j++) {
                    if (!pa.used()[j]) {
                        int[] t = pa.tour().clone();
                        t[pa.len()] = j;
                        boolean[] u = pa.used().clone();
                        u[j] = true;
                        next.add(new Partial(t, pa.len() + 1, u, pa.score() + Math.log(p[cur][j] + 1e-9)));
                    }
                }
            }
            next.sort((a, c) -> Double.compare(c.score(), a.score()));
            live = new ArrayList<>(next.subList(0, Math.min(beam, next.size())));
        }
        int[] best = null;
        double bestLen = Double.POSITIVE_INFINITY;
        for (Partial pa : live) {
            double l = ClassicHeuristics.tourLength(d, pa.tour());
            if (l < bestLen) {
                bestLen = l;
                best = pa.tour();
            }
        }
        return best;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(nodeIn, edgeIn));
        for (Var[] w : layers) {
            p.addAll(Arrays.asList(w));
        }
        p.addAll(List.of(out, outBias));
        return p;
    }
}
