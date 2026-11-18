package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.SparseMatrix;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * LightGCN (He et al., 2020, <a href="https://arxiv.org/abs/2002.02126">arXiv:2002.02126</a>):
 * collaborative filtering on the user-item bipartite graph with every transform removed. Users
 * and items get one learned embedding each ({@code E0}); layers only propagate them,
 * {@code E(k+1) = A_hat E(k)} with the symmetric normalisation
 * {@code A_hat = D^-1/2 A D^-1/2}; the final embedding is the mean over layers {@code 0..K}, and
 * a user-item score is the dot product. Trained with the BPR loss. The paper's ablation shows
 * that the feature transforms and non-linearities of NGCF hurt collaborative filtering; with
 * {@link #ngcf} the same class adds a weight matrix and a leaky ReLU per layer (NGCF-lite) for
 * that comparison. {@code K = 0} is plain BPR matrix factorization.
 */
public final class LightGcn {

    private final int users;
    private final int items;
    private final int layers;
    private final SparseMatrix adjacency;
    private final Var embedding;
    private final List<Var> transforms = new ArrayList<>();
    private Optimizer optimizer = Optimizer.adam(0.01f);
    private float l2 = 1e-4f;
    private Tensor cached;

    private LightGcn(int users, int items, int[] edgeUsers, int[] edgeItems, int dim, int layers, boolean ngcf,
            long seed) {
        if (edgeUsers.length != edgeItems.length || edgeUsers.length == 0) {
            throw new IllegalArgumentException("need matching, non-empty edge lists");
        }
        this.users = users;
        this.items = items;
        this.layers = layers;
        SplittableRandom r = new SplittableRandom(seed);
        int nodes = users + items;
        double[] degree = new double[nodes];
        for (int e = 0; e < edgeUsers.length; e++) {
            degree[edgeUsers[e]]++;
            degree[users + edgeItems[e]]++;
        }
        int nnz = 2 * edgeUsers.length;
        int[] rows = new int[nnz];
        int[] cols = new int[nnz];
        float[] vals = new float[nnz];
        for (int e = 0; e < edgeUsers.length; e++) {
            int u = edgeUsers[e];
            int i = users + edgeItems[e];
            float w = (float) (1 / Math.sqrt(degree[u] * degree[i]));
            rows[2 * e] = u;
            cols[2 * e] = i;
            vals[2 * e] = w;
            rows[2 * e + 1] = i;
            cols[2 * e + 1] = u;
            vals[2 * e + 1] = w;
        }
        adjacency = SparseMatrix.of(nodes, nodes, rows, cols, vals, Device.cpu());
        embedding = RecsysOps.normal(r, 0.1f, nodes, dim);
        if (ngcf) {
            for (int k = 0; k < layers; k++) {
                transforms.add(RecsysOps.matrix(r, dim, dim));
            }
        }
    }

    /**
     * LightGCN.
     *
     * @param users the number of users
     * @param items the number of items
     * @param edgeUsers the user of each observed interaction
     * @param edgeItems the item of each observed interaction
     * @param dim the embedding size
     * @param layers the propagation depth {@code K}
     * @param seed the initialization seed
     * @return the model
     */
    public static LightGcn of(int users, int items, int[] edgeUsers, int[] edgeItems, int dim, int layers,
            long seed) {
        return new LightGcn(users, items, edgeUsers, edgeItems, dim, layers, false, seed);
    }

    /**
     * NGCF-lite: the same propagation with a weight matrix and a leaky ReLU in every layer, the
     * ablation that LightGCN removes.
     *
     * @param users the number of users
     * @param items the number of items
     * @param edgeUsers the user of each observed interaction
     * @param edgeItems the item of each observed interaction
     * @param dim the embedding size
     * @param layers the propagation depth
     * @param seed the initialization seed
     * @return the model
     */
    public static LightGcn ngcf(int users, int items, int[] edgeUsers, int[] edgeItems, int dim, int layers,
            long seed) {
        return new LightGcn(users, items, edgeUsers, edgeItems, dim, layers, true, seed);
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public LightGcn optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /**
     * @param weight the L2 penalty on the layer-0 embeddings of a batch
     * @return this model
     */
    public LightGcn l2(float weight) {
        this.l2 = weight;
        return this;
    }

    /**
     * Propagates the embeddings.
     *
     * @return the final embeddings of all users then all items, {@code [users + items, dim]}
     */
    public Var propagate() {
        Var e = embedding;
        Var sum = e;
        for (int k = 0; k < layers; k++) {
            e = adjacency.matmul(e);
            if (!transforms.isEmpty()) {
                Var h = e.matmul(transforms.get(k));
                e = h.maximum(h.mul(0.2f));
            }
            sum = sum.add(e);
        }
        return sum.div(layers + 1);
    }

    /**
     * One BPR step on a batch of triples.
     *
     * @param user user ids
     * @param positive an item each user interacted with
     * @param negative an item each user did not
     * @return the loss
     */
    public double trainStep(int[] user, int[] positive, int[] negative) {
        cached = null;
        Var all = propagate();
        Tensor u = RecsysOps.ids(user);
        Tensor p = RecsysOps.ids(offset(positive));
        Tensor n = RecsysOps.ids(offset(negative));
        Var eu = all.gather(0, u);
        Var diff = eu.mul(all.gather(0, p).sub(all.gather(0, n))).sum(1, true);
        Var reg = embedding.gather(0, u).square().add(embedding.gather(0, p).square())
                .add(embedding.gather(0, n).square()).sum().mul(l2 / user.length);
        Var loss = diff.neg().softplus().mean().add(reg);
        loss.backward();
        optimizer.step(parameters());
        return loss.value().item();
    }

    private int[] offset(int[] item) {
        int[] o = new int[item.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = users + item[i];
        }
        return o;
    }

    /**
     * @param user a user id
     * @return the score of every item for the user
     */
    public float[] scores(int user) {
        if (cached == null) {
            cached = Var.noGrad(this::propagate).value();
        }
        Tensor u = cached.slice(0, user, user + 1);
        return u.matmul(cached.slice(0, users, users + items).transpose()).toArray();
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding));
        p.addAll(transforms);
        return p;
    }

    /** @return the propagation depth */
    public int layers() {
        return layers;
    }

    /** @return the number of users and items, as {@code {users, items}} */
    public int[] size() {
        return new int[] {users, items};
    }

    @Override
    public String toString() {
        return (transforms.isEmpty() ? "LightGcn" : "NgcfLite") + "(K=" + layers + ")";
    }
}
