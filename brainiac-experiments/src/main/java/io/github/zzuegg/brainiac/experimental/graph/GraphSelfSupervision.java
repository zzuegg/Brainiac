package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Self-supervised pretraining objectives for graph encoders, three ways to learn node
 * embeddings without labels:
 * <ul>
 *   <li>{@link #dgi} Deep Graph Infomax (Veličković et al., 2019, <a href="https://arxiv.org/abs/1809.10341">arXiv:1809.10341</a>):
 *       a bilinear discriminator tells node embeddings of the real graph, paired with the
 *       graph summary {@code s = sigmoid(mean(H))}, from those of a corrupted graph whose
 *       feature rows are shuffled; maximising this local-global mutual information makes
 *       embeddings capture what is shared across the graph (the communities);</li>
 *   <li>{@link #graphCl} GraphCL (You et al., 2020, <a href="https://arxiv.org/abs/2010.13902">arXiv:2010.13902</a>),
 *       here at node level: two views from edge dropping and feature masking, a projection
 *       head, and the InfoNCE loss ({@link SelfSupervisedLosses#ntXent}) that pulls a node's
 *       two views together and pushes other nodes away;</li>
 *   <li>{@link #graphMae} GraphMAE (Hou et al., 2022, <a href="https://arxiv.org/abs/2205.10803">arXiv:2205.10803</a>):
 *       replace a fraction of the nodes' features by a learned mask token, encode,
 *       re-mask the encoded masked nodes, decode with one graph convolution, and score the
 *       reconstruction of the masked features by the scaled cosine error
 *       {@code (1 - cos)^gamma}.</li>
 * </ul>
 * The encoder is anything that maps features and an adjacency to node embeddings.
 */
public final class GraphSelfSupervision {

    /** A graph encoder. */
    public interface Encoder {
        /**
         * Embeds the nodes.
         *
         * @param x node features {@code [n, in]}
         * @param adjacency {@code [n, n]}
         * @param training whether the encoder is training
         * @return {@code [n, width]}
         */
        Var encode(Var x, Tensor adjacency, boolean training);

        /** @return the encoder's parameters */
        List<Var> parameters();
    }

    /** The objective. */
    public enum Method {
        /** Deep Graph Infomax */
        DGI,
        /** contrastive two-view InfoNCE */
        GRAPH_CL,
        /** masked feature reconstruction */
        GRAPH_MAE
    }

    private final Method method;
    private final Encoder encoder;
    private final SplittableRandom random;
    private final float rate;
    private final List<Var> own = new ArrayList<>();

    private GraphSelfSupervision(Method method, Encoder encoder, float rate, SplittableRandom random) {
        this.method = method;
        this.encoder = Objects.requireNonNull(encoder);
        this.rate = rate;
        this.random = random;
    }

    /**
     * Deep Graph Infomax.
     *
     * @param encoder the encoder
     * @param width its embedding width
     * @param random the initialization and corruption source
     * @return the objective
     */
    public static GraphSelfSupervision dgi(Encoder encoder, int width, SplittableRandom random) {
        GraphSelfSupervision s = new GraphSelfSupervision(Method.DGI, encoder, 0, random);
        s.own.add(GraphMath.weight(width, width, random));
        return s;
    }

    /**
     * Node-level GraphCL with edge dropping and feature masking.
     *
     * @param encoder the encoder
     * @param width its embedding width
     * @param rate the edge-drop and feature-mask rate of each view
     * @param random the initialization and augmentation source
     * @return the objective
     */
    public static GraphSelfSupervision graphCl(Encoder encoder, int width, float rate, SplittableRandom random) {
        GraphSelfSupervision s = new GraphSelfSupervision(Method.GRAPH_CL, encoder, rate, random);
        s.own.add(GraphMath.weight(width, width, random));
        s.own.add(GraphMath.bias(width));
        s.own.add(GraphMath.weight(width, width, random));
        return s;
    }

    /**
     * GraphMAE.
     *
     * @param encoder the encoder
     * @param in the input feature width
     * @param width its embedding width
     * @param rate the fraction of nodes masked
     * @param random the initialization and masking source
     * @return the objective
     */
    public static GraphSelfSupervision graphMae(Encoder encoder, int in, int width, float rate,
            SplittableRandom random) {
        GraphSelfSupervision s = new GraphSelfSupervision(Method.GRAPH_MAE, encoder, rate, random);
        s.own.add(Var.param(Tensor.zeros(Shape.of(1, in))));
        s.own.add(GraphMath.weight(width, in, random));
        return s;
    }

    /** @return the objective */
    public Method method() {
        return method;
    }

    /** @return the encoder's and the objective's own parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.addAll(own);
        return p;
    }

    /**
     * The self-supervised loss on one graph.
     *
     * @param x node features {@code [n, in]}
     * @param adjacency {@code [n, n]}
     * @return the loss
     */
    public Var loss(Var x, Tensor adjacency) {
        int n = x.shape().dim(0);
        return switch (method) {
            case DGI -> {
                Var h = encoder.encode(x, adjacency, true);
                float[] perm = new float[n];
                int[] p = permutation(n);
                for (int i = 0; i < n; i++) {
                    perm[i] = p[i];
                }
                Var hc = encoder.encode(x.gather(0, Tensor.of(Shape.of(n), perm)), adjacency, true);
                Var summary = h.mean(0, true).sigmoid().transpose();
                Var w = own.get(0);
                Var pos = h.matmul(w).matmul(summary);
                Var neg = hc.matmul(w).matmul(summary);
                yield pos.softplus().sub(pos).mean().add(neg.softplus().mean());
            }
            case GRAPH_CL -> {
                Var z1 = project(encoder.encode(maskFeatures(x), DropEdge.drop(adjacency, rate, random), true));
                Var z2 = project(encoder.encode(maskFeatures(x), DropEdge.drop(adjacency, rate, random), true));
                yield SelfSupervisedLosses.ntXent(z1, z2, 0.5f);
            }
            case GRAPH_MAE -> {
                float[] keep = new float[n];
                for (int i = 0; i < n; i++) {
                    keep[i] = random.nextDouble() < rate ? 0 : 1;
                }
                Tensor keepT = Tensor.of(Shape.of(n, 1), keep);
                Tensor maskT = keepT.neg().add(1f);
                Var masked = x.mul(keepT).add(own.get(0).mul(maskT));
                Var h = encoder.encode(masked, adjacency, true).mul(keepT);
                Var recon = Var.of(GraphOps.gcnNormalize(adjacency)).matmul(h).matmul(own.get(1));
                Var cos = SelfSupervisedLosses.normalize(recon).mul(SelfSupervisedLosses.normalize(x)).sum(1, true);
                float count = Math.max(1, n - keepT.sum().toArray()[0]);
                yield cos.neg().add(1f).square().mul(maskT).sum().div(count);
            }
        };
    }

    private Var project(Var h) {
        return GraphMath.linear(h, own.get(0), own.get(1)).relu().matmul(own.get(2));
    }

    private Var maskFeatures(Var x) {
        int f = x.shape().dim(1);
        float[] keep = new float[f];
        for (int i = 0; i < f; i++) {
            keep[i] = random.nextDouble() < rate ? 0 : 1;
        }
        return x.mul(Tensor.of(Shape.of(1, f), keep));
    }

    private int[] permutation(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    @Override
    public String toString() {
        return "GraphSelfSupervision(" + method + ")";
    }
}
