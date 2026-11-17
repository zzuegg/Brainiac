package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A two-tower (dual encoder) retrieval model (Yi et al., 2019,
 * <a href="https://doi.org/10.1145/3298689.3346996">RecSys 2019</a>; Covington et al., 2016): a
 * user tower and an item tower embed their own features into one space, and relevance is
 * the cosine similarity of the two embeddings. Because items are embedded independently
 * of users, all item embeddings can be precomputed and searched with a nearest-neighbour
 * index, which is how candidate generation scales to millions of items.
 *
 * <p>Training uses the <b>in-batch sampled softmax</b>: for a batch of positive
 * (user, item) pairs, every other item in the batch is a negative, and each user must pick
 * its own item among them (cross-entropy over similarities divided by a temperature).
 */
public final class TwoTowerRetrieval {

    private final Model userTower;
    private final Model itemTower;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float temperature = 0.1f;

    private TwoTowerRetrieval(Model userTower, Model itemTower) {
        this.userTower = userTower;
        this.itemTower = itemTower;
        if (userTower.outputSize() != itemTower.outputSize()) {
            throw new IllegalArgumentException("both towers must output embeddings of the same size");
        }
    }

    /**
     * @param userTower user features to an embedding
     * @param itemTower item features to an embedding of the same size
     * @return the model
     */
    public static TwoTowerRetrieval of(Model userTower, Model itemTower) {
        return new TwoTowerRetrieval(Objects.requireNonNull(userTower), Objects.requireNonNull(itemTower));
    }

    /** @param t the softmax temperature @return this model */
    public TwoTowerRetrieval temperature(float t) {
        this.temperature = t;
        return this;
    }

    /** @param o the optimizer @return this model */
    public TwoTowerRetrieval optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    private static Var normalize(Var e) {
        return e.div(e.square().sum(1, true).add(1e-8f).sqrt());
    }

    /**
     * One step of in-batch softmax training.
     *
     * @param users {@code [batch, userFeatures]}
     * @param items {@code [batch, itemFeatures]}: row {@code i} is user {@code i}'s positive
     * @return the loss
     */
    public double trainStep(Tensor users, Tensor items) {
        int b = users.shape().dim(0);
        Var u = normalize(userTower.forward(Var.of(users), true));
        Var v = normalize(itemTower.forward(Var.of(items), true));
        Var logits = u.matmul(v.transpose()).div(temperature);
        Var shifted = logits.sub(logits.value().max(1, true));
        Var logp = shifted.sub(shifted.exp().sum(1, true).log());
        float[] eye = new float[b * b];
        for (int i = 0; i < b; i++) {
            eye[i * b + i] = 1;
        }
        Var loss = logp.mul(Tensor.of(Shape.of(b, b), eye)).sum().neg().div(b);
        loss.backward();
        List<Var> params = new ArrayList<>(userTower.parameters());
        params.addAll(itemTower.parameters());
        optimizer.step(params);
        return loss.value().item();
    }

    /**
     * @param users {@code [n, userFeatures]}
     * @return unit-length user embeddings
     */
    public Tensor embedUsers(Tensor users) {
        return normalize(userTower.forward(Var.of(users), false)).value();
    }

    /**
     * @param items {@code [m, itemFeatures]}
     * @return unit-length item embeddings
     */
    public Tensor embedItems(Tensor items) {
        return normalize(itemTower.forward(Var.of(items), false)).value();
    }

    /**
     * Retrieves the best items for every user by exhaustive similarity search, with an exact
     * core {@link VectorIndex#flat} index over the item embeddings (use {@link #embedItems}
     * with {@link VectorIndex#hnsw} for large catalogues).
     *
     * @param users {@code [n, userFeatures]}
     * @param items {@code [m, itemFeatures]}
     * @param k how many items per user
     * @return {@code [n][k]} item indices, best first
     */
    public int[][] topK(Tensor users, Tensor items, int k) {
        Tensor itemEmbeddings = embedItems(items).to(DType.FLOAT32);
        Tensor userEmbeddings = embedUsers(users).to(DType.FLOAT32);
        int dim = itemEmbeddings.shape().dim(1);
        VectorIndex index = VectorIndex.flat(dim, VectorIndex.Metric.DOT, itemEmbeddings.device());
        index.add(itemEmbeddings);
        int n = users.shape().dim(0);
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = index.search(userEmbeddings.slice(0, i, i + 1).reshape(dim), k).stream()
                    .mapToInt(VectorIndex.Hit::id).toArray();
        }
        return out;
    }
}
