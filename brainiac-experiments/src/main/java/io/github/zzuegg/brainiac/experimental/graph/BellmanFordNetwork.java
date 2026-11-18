package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Neural Bellman-Ford networks for link prediction (Zhu et al., NeurIPS 2021,
 * <a href="https://arxiv.org/abs/2106.06935">arXiv:2106.06935</a>). A pair {@code (u, v)} is
 * represented by a learned generalised sum, over all paths from {@code u} to {@code v}, of the
 * product of the edges' relation representations, computed by a Bellman-Ford iteration:
 * <ul>
 *   <li>indicator: the source starts with the query relation's embedding, every other node with
 *       zeros;</li>
 *   <li>message: the neighbour's state times the edge relation's embedding (DistMult
 *       composition, as in CompGCN);</li>
 *   <li>aggregation: the sum of messages plus the boundary (indicator) state, then a linear map,
 *       RMS normalisation, ReLU and a shortcut;</li>
 *   <li>read-out: an MLP on {@code [h_v, q]} scores every node as the tail.</li>
 * </ul>
 * Nothing is tied to entity identities, so the model is inductive: it scores entities of a graph
 * it never saw. {@link #topEdge} reads out the most important edge for a prediction by the
 * gradient of the score with respect to edge weights, the paper's path interpretation.
 */
public final class BellmanFordNetwork {

    private final int relations;
    private final int dim;
    private final int layers;
    private final Var query;
    private final List<Var[]> layer = new ArrayList<>();
    private final Var mlp1;
    private final Var mlp1b;
    private final Var mlp2;
    private final List<Var> params = new ArrayList<>();

    /**
     * Creates a model.
     *
     * @param relations relation types (inverses are added)
     * @param dim the state width
     * @param layers Bellman-Ford iterations
     * @param seed the initialisation seed
     */
    public BellmanFordNetwork(int relations, int dim, int layers, long seed) {
        this.relations = relations;
        this.dim = dim;
        this.layers = layers;
        SplittableRandom r = new SplittableRandom(seed);
        int rr = 2 * relations;
        query = Var.param(Tensor.uniform(Shape.of(rr, dim), -1, 1, r));
        params.add(query);
        for (int l = 0; l < layers; l++) {
            Var rel = Var.param(Tensor.uniform(Shape.of(rr, 1, 1, dim), -1, 1, r));
            Var w = weight(r, dim, dim);
            Var b = Var.param(Tensor.zeros(Shape.of(dim)));
            layer.add(new Var[] {rel, w, b});
            params.addAll(List.of(rel, w, b));
        }
        mlp1 = weight(r, 2 * dim, dim);
        mlp1b = Var.param(Tensor.zeros(Shape.of(dim)));
        mlp2 = weight(r, dim, 1);
        params.addAll(List.of(mlp1, mlp1b, mlp2));
    }

    private static Var weight(SplittableRandom r, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -limit, limit, r));
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    /**
     * The unnormalised relational adjacency.
     *
     * @param triples {@code {head, relation, tail}}
     * @param entities the entity count
     * @param relations relation types (inverses are added)
     * @return {@code [2R, N, N]} 0/1 with {@code A[r][v][u] = 1} for each edge {@code u -r-> v}
     */
    public static float[] adjacency(List<int[]> triples, int entities, int relations) {
        float[] a = new float[2 * relations * entities * entities];
        for (int[] t : triples) {
            a[(t[1] * entities + t[2]) * entities + t[0]] = 1;
            a[((t[1] + relations) * entities + t[0]) * entities + t[2]] = 1;
        }
        return a;
    }

    /**
     * Scores every entity as the tail of each query.
     *
     * @param adjacency the graph {@code [2R, N, N]} (a {@code Var} so edge weights can be
     *     differentiated)
     * @param entities the entity count
     * @param heads query heads
     * @param rels query relations in {@code [0, 2R)}
     * @return {@code [queries, N]} scores
     */
    public Var scores(Var adjacency, int entities, int[] heads, int[] rels) {
        int b = heads.length;
        int n = entities;
        int rr = 2 * relations;
        float[] onehot = new float[b * n];
        for (int i = 0; i < b; i++) {
            onehot[i * n + heads[i]] = 1;
        }
        Var q = query.gather(0, RelationalGcn.ids(rels));                  // [B, d]
        Var boundary = Var.of(Tensor.of(Shape.of(b, n, 1), onehot)).mul(q.reshape(b, 1, dim));
        Var x = boundary;
        Var a = adjacency.reshape(rr * n, n);
        for (int l = 0; l < layers; l++) {
            Var[] p = layer.get(l);
            Var flat = x.permute(1, 0, 2).reshape(n, b * dim);
            Var msg = a.matmul(flat).reshape(rr, n, b, dim).mul(p[0]).sum(0, false);   // [N, B, d]
            Var agg = msg.permute(1, 0, 2).add(boundary);
            Var h = agg.matmul(p[1]).add(p[2]);
            h = h.div(h.square().mean(-1, true).add(1e-6f).sqrt()).relu();
            x = h.add(x);
        }
        Var feat = Var.concat(-1, x, q.reshape(b, 1, dim).expand(Shape.of(b, n, dim)));
        return feat.matmul(mlp1).add(mlp1b).relu().matmul(mlp2).reshape(b, n);
    }

    /**
     * Trains with a softmax over all entities as the tail of each training triple and its
     * inverse; the batch's own edges are removed from the graph so they must be inferred from
     * other paths.
     *
     * @param triples training triples (also the graph)
     * @param entities the entity count
     * @param steps optimiser steps
     * @param batch triples per step
     * @param lr the learning rate
     * @param seed the sampling seed
     * @return the final loss
     */
    public double train(List<int[]> triples, int entities, int steps, int batch, float lr, long seed) {
        float[] full = adjacency(triples, entities, relations);
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int n = entities;
        double last = 0;
        for (int s = 0; s < steps; s++) {
            float[] a = full.clone();
            int[] heads = new int[2 * batch];
            int[] rels = new int[2 * batch];
            float[] tails = new float[2 * batch];
            for (int i = 0; i < batch; i++) {
                int[] t = triples.get(r.nextInt(triples.size()));
                heads[2 * i] = t[0];
                rels[2 * i] = t[1];
                tails[2 * i] = t[2];
                heads[2 * i + 1] = t[2];
                rels[2 * i + 1] = t[1] + relations;
                tails[2 * i + 1] = t[0];
                a[(t[1] * n + t[2]) * n + t[0]] = 0;
                a[((t[1] + relations) * n + t[0]) * n + t[2]] = 0;
            }
            Var adj = Var.of(Tensor.of(Shape.of(2 * relations, n, n), a));
            Var loss = Functional.crossEntropy(scores(adj, n, heads, rels), Tensor.of(Shape.of(2 * batch), tails));
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * @param adjacency the graph from {@link #adjacency}
     * @param entities the entity count
     * @param head the head
     * @param rel the relation
     * @return one score per entity
     */
    public float[] scoreTails(float[] adjacency, int entities, int head, int rel) {
        return Var.noGrad(() -> scores(Var.of(Tensor.of(Shape.of(2 * relations, entities, entities), adjacency)),
                entities, new int[] {head}, new int[] {rel}).value().toArray());
    }

    /**
     * The edge whose weight most increases the score of {@code (head, rel, tail)}: the gradient
     * of the score with respect to a weight on every existing edge.
     *
     * @param adjacency the graph from {@link #adjacency}
     * @param entities the entity count
     * @param head the head
     * @param rel the relation
     * @param tail the tail
     * @return {@code {relation (in [0, 2R)), source u, target v}} of the top edge {@code u -> v}
     */
    public int[] topEdge(float[] adjacency, int entities, int head, int rel, int tail) {
        int rr = 2 * relations;
        int n = entities;
        Var weights = Var.param(Tensor.full(Shape.of(rr, n, n), 1f));
        Var adj = Var.of(Tensor.of(Shape.of(rr, n, n), adjacency)).mul(weights);
        Var s = scores(adj, n, new int[] {head}, new int[] {rel});
        float[] pick = new float[n];
        pick[tail] = 1;
        s.mul(Tensor.of(Shape.of(1, n), pick)).sum().backward();
        float[] g = weights.grad().toArray();
        int best = -1;
        for (int i = 0; i < g.length; i++) {
            if (adjacency[i] > 0 && (best < 0 || g[i] > g[best])) {
                best = i;
            }
        }
        int r = best / (n * n);
        int v = best / n % n;
        int u = best % n;
        return new int[] {r, u, v};
    }
}
