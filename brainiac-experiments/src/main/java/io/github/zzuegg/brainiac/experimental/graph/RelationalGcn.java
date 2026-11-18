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
 * Relational message passing on multi-relational graphs, as an encoder in front of a DistMult
 * decoder for link prediction (or a classifier head for nodes):
 * <ul>
 *   <li>R-GCN (Schlichtkrull et al., ESWC 2018, <a href="https://arxiv.org/abs/1703.06103">
 *       arXiv:1703.06103</a>): {@code h_v' = relu(W_0 h_v + sum_r sum_(u in N_r(v)) W_r h_u / c_(v,r))}
 *       with one matrix per relation and direction, regularised by <em>basis decomposition</em>
 *       ({@code W_r = sum_b a_rb V_b}) or <em>block-diagonal</em> matrices;</li>
 *   <li>CompGCN (Vashishth et al., ICLR 2020, <a href="https://arxiv.org/abs/1911.03082">
 *       arXiv:1911.03082</a>): relations get embeddings {@code z_r} too, and each message composes
 *       the neighbour with its relation, {@code phi(h_u, z_r)}, by subtraction (TransE), product
 *       (DistMult) or circular correlation (HolE), then one matrix per direction (original,
 *       inverse, self-loop); relation embeddings are updated by their own matrix, so the
 *       parameter count does not grow with the number of relations.</li>
 * </ul>
 * Every relation is used in both directions (relation {@code r + R} is the inverse of {@code r}).
 * Entities either have learned input embeddings (transductive) or a constant input
 * ({@code featureless}), which makes the encoder applicable to unseen entities. The adjacency is
 * a dense {@code [2R, N, N]} tensor normalised by in-degree per relation, fine for toy graphs.
 */
public final class RelationalGcn {

    /** The layer type. */
    public enum Kind {
        /** R-GCN with basis decomposition. */
        RGCN_BASIS,
        /** R-GCN with block-diagonal relation matrices. */
        RGCN_BLOCK,
        /** CompGCN with subtraction. */
        COMPGCN_SUB,
        /** CompGCN with the element-wise product. */
        COMPGCN_MULT,
        /** CompGCN with circular correlation. */
        COMPGCN_CORR
    }

    private final Kind kind;
    private final int relations;
    private final int dim;
    private final int layers;
    private final boolean featureless;
    private final Var entity;
    private final List<Var[]> weights = new ArrayList<>();
    private final Var relation0;      // CompGCN input relation embeddings [2R, d]
    private final Var decoder;        // R-GCN DistMult relation diagonals [2R, d]
    private final Tensor blockMask;
    private final Tensor circulantIndex;
    private final List<Var> params = new ArrayList<>();

    /**
     * Creates an encoder.
     *
     * @param kind the layer type
     * @param entities entities with learned inputs; 0 for a featureless (inductive) encoder
     * @param relations relation types (inverses are added)
     * @param dim the width
     * @param layers message-passing layers
     * @param bases bases (R-GCN basis) or blocks (R-GCN block)
     * @param seed the initialisation seed
     */
    public RelationalGcn(Kind kind, int entities, int relations, int dim, int layers, int bases, long seed) {
        this.kind = kind;
        this.relations = relations;
        this.dim = dim;
        this.layers = layers;
        this.featureless = entities == 0;
        SplittableRandom r = new SplittableRandom(seed);
        int rr = 2 * relations;
        entity = featureless ? null : Var.param(Tensor.uniform(Shape.of(entities, dim), -0.5f, 0.5f, r));
        if (entity != null) {
            params.add(entity);
        }
        boolean comp = kind.name().startsWith("COMPGCN");
        float[] mask = new float[dim * dim];
        int block = Math.max(1, dim / Math.max(1, bases));
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                mask[i * dim + j] = i / block == j / block ? 1 : 0;
            }
        }
        blockMask = Tensor.of(Shape.of(1, dim, dim), mask);
        float[] circ = new float[dim * dim];
        for (int i = 0; i < dim; i++) {
            for (int k = 0; k < dim; k++) {
                circ[i * dim + k] = (i + k) % dim;
            }
        }
        circulantIndex = Tensor.of(Shape.of(dim * dim), circ);
        for (int l = 0; l < layers; l++) {
            Var[] w;
            if (comp) {
                w = new Var[] {weight(r, dim, dim), weight(r, dim, dim), weight(r, dim, dim), weight(r, dim, dim)};
            } else if (kind == Kind.RGCN_BASIS) {
                w = new Var[] {weight(r, dim, dim), Var.param(Tensor.uniform(Shape.of(rr, bases), -1, 1, r)),
                    Var.param(Tensor.uniform(Shape.of(bases, dim * dim), -scale(dim), scale(dim), r))};
            } else {
                w = new Var[] {weight(r, dim, dim),
                    Var.param(Tensor.uniform(Shape.of(rr, dim, dim), -scale(dim) * 2, scale(dim) * 2, r))};
            }
            weights.add(w);
            params.addAll(List.of(w));
        }
        relation0 = comp ? Var.param(Tensor.uniform(Shape.of(rr, dim), -0.5f, 0.5f, r)) : null;
        decoder = comp ? null : Var.param(Tensor.uniform(Shape.of(rr, dim), -0.5f, 0.5f, r));
        params.add(comp ? relation0 : decoder);
    }

    private static float scale(int d) {
        return (float) Math.sqrt(3.0 / d);
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
     * The normalised relational adjacency.
     *
     * @param triples {@code {head, relation, tail}}
     * @param entities the number of entities
     * @param relations relation types (inverses are added)
     * @return {@code [2R, N, N]} with {@code A[r][v][u] = 1 / |N_r(v)|} for each edge {@code u -r-> v}
     */
    public static Tensor adjacency(List<int[]> triples, int entities, int relations) {
        int rr = 2 * relations;
        float[] a = new float[rr * entities * entities];
        int[] deg = new int[rr * entities];
        for (int[] t : triples) {
            a[(t[1] * entities + t[2]) * entities + t[0]] = 1;
            a[((t[1] + relations) * entities + t[0]) * entities + t[2]] = 1;
        }
        for (int r = 0; r < rr; r++) {
            for (int v = 0; v < entities; v++) {
                int base = (r * entities + v) * entities;
                for (int u = 0; u < entities; u++) {
                    deg[r * entities + v] += (int) a[base + u];
                }
            }
        }
        for (int r = 0; r < rr; r++) {
            for (int v = 0; v < entities; v++) {
                int d = deg[r * entities + v];
                if (d > 1) {
                    int base = (r * entities + v) * entities;
                    for (int u = 0; u < entities; u++) {
                        a[base + u] /= d;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(rr, entities, entities), a);
    }

    /** Aggregates per-relation messages {@code [2R, N, d]} over in-neighbours: {@code [2R, N, d]}. */
    private static Var aggregate(Tensor adjacency, Var messages) {
        return Var.of(adjacency).matmul(messages);
    }

    private Var expand(Var h, int n) {
        return h.reshape(1, n, dim).expand(Shape.of(2 * relations, n, dim));
    }

    /**
     * Encodes every entity.
     *
     * @param adjacency from {@link #adjacency}
     * @param entities the number of entities
     * @return entity states {@code [N, d]} and, for CompGCN, relation states {@code [2R, d]}
     */
    public Var[] encode(Tensor adjacency, int entities) {
        int n = entities;
        int rr = 2 * relations;
        Var h = featureless ? Var.of(Tensor.full(Shape.of(n, dim), 1f)) : entity;
        Var z = relation0;
        for (int l = 0; l < layers; l++) {
            Var[] w = weights.get(l);
            Var next;
            if (z != null) {
                Var hb = expand(h, n);
                Var zb = z.reshape(rr, 1, dim);
                Var msg = switch (kind) {
                    case COMPGCN_SUB -> hb.sub(zb);
                    case COMPGCN_MULT -> hb.mul(zb);
                    default -> hb.matmul(z.gather(1, circulantIndex).reshape(rr, dim, dim));
                };
                Var agg = aggregate(adjacency, msg);
                Var out = agg.slice(0, 0, relations).sum(0, false).matmul(w[0]);
                Var in = agg.slice(0, relations, rr).sum(0, false).matmul(w[1]);
                next = out.add(in).add(h.matmul(w[2])).mul(1f / 3);
                z = z.matmul(w[3]);
            } else {
                Var wr = kind == Kind.RGCN_BASIS ? w[1].matmul(w[2]).reshape(rr, dim, dim)
                        : w[1].mul(blockMask);
                Var agg = aggregate(adjacency, expand(h, n)).matmul(wr).sum(0, false);
                next = agg.add(h.matmul(w[0]));
            }
            h = l + 1 < layers ? next.relu() : next;
        }
        return z == null ? new Var[] {h} : new Var[] {h, z};
    }

    /**
     * DistMult scores of every entity as the tail of each query.
     *
     * @param encoded the output of {@link #encode}
     * @param heads query heads
     * @param rels query relations in {@code [0, 2R)}
     * @return {@code [queries, N]} scores
     */
    public Var scores(Var[] encoded, int[] heads, int[] rels) {
        Var h = encoded[0];
        Var relTable = encoded.length > 1 ? encoded[1] : decoder;
        Var q = h.gather(0, ids(heads)).mul(relTable.gather(0, ids(rels)));
        return q.matmul(h.transpose());
    }

    static Tensor ids(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    /**
     * Trains for link prediction with 1-N scoring: a softmax over all entities as the tail of each
     * training triple and of its inverse.
     *
     * @param triples training triples (also the message-passing graph)
     * @param entities the number of entities
     * @param steps optimiser steps
     * @param batch triples per step
     * @param lr the learning rate
     * @param seed the sampling seed
     * @return the final loss
     */
    public double train(List<int[]> triples, int entities, int steps, int batch, float lr, long seed) {
        Tensor adj = adjacency(triples, entities, relations);
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        double last = 0;
        for (int s = 0; s < steps; s++) {
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
            }
            Var loss = Functional.crossEntropy(scores(encode(adj, entities), heads, rels),
                    Tensor.of(Shape.of(2 * batch), tails));
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Scores every entity as the tail of {@code (head, relation, ?)} on a graph.
     *
     * @param adjacency the graph from {@link #adjacency}
     * @param entities its entity count
     * @param head the head
     * @param rel the relation
     * @return one score per entity
     */
    public float[] scoreTails(Tensor adjacency, int entities, int head, int rel) {
        return Var.noGrad(() -> scores(encode(adjacency, entities), new int[] {head}, new int[] {rel}).value()
                .toArray());
    }

    /** @return the layer type */
    public Kind kind() {
        return kind;
    }
}
