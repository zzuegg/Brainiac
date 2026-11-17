package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Matrix factorization trained with Bayesian personalized ranking (Rendle et al., 2009,
 * <a href="https://arxiv.org/abs/1205.2618">arXiv:1205.2618</a>), the standard collaborative
 * filter for implicit feedback (clicks, plays, purchases). Users and items get learned
 * embeddings and a bias per item; for a user, an item they interacted with should score
 * higher than one they did not: the loss is {@code -log sigmoid(s(u, i) - s(u, j))} over
 * sampled (user, positive, negative) triples, optimizing the ranking (AUC) directly
 * rather than reconstructing a rating matrix full of unknowns.
 */
public final class BprMatrixFactorization {

    private final Var users;
    private final Var items;
    private final Var itemBias;
    private final int factors;
    private Optimizer optimizer = Optimizer.adam(0.02f);
    private float l2 = 1e-4f;

    /**
     * Creates the model.
     *
     * @param users the number of users
     * @param items the number of items
     * @param factors the embedding size
     * @param seed the initialization seed
     */
    public BprMatrixFactorization(int users, int items, int factors, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.factors = factors;
        this.users = Var.param(Tensor.normal(Shape.of(users, factors), 0, 0.1f, r));
        this.items = Var.param(Tensor.normal(Shape.of(items, factors), 0, 0.1f, r));
        this.itemBias = Var.param(Tensor.zeros(Shape.of(items, 1)));
    }

    /** @param o the optimizer @return this model */
    public BprMatrixFactorization optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /** @param weight the L2 penalty on the embeddings used in a step @return this model */
    public BprMatrixFactorization l2(float weight) {
        this.l2 = weight;
        return this;
    }

    private static Tensor indices(int[] ids) {
        float[] f = new float[ids.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = ids[i];
        }
        return Tensor.of(Shape.of(ids.length), f);
    }

    /**
     * One step on a batch of triples.
     *
     * @param user user ids
     * @param positive an item each user interacted with
     * @param negative an item each user did not
     * @return the loss
     */
    public double trainStep(int[] user, int[] positive, int[] negative) {
        Var u = users.gather(0, indices(user));
        Var i = items.gather(0, indices(positive));
        Var j = items.gather(0, indices(negative));
        Var bi = itemBias.gather(0, indices(positive));
        Var bj = itemBias.gather(0, indices(negative));
        Var diff = u.mul(i.sub(j)).sum(1, true).add(bi).sub(bj);
        Var loss = diff.neg().softplus().mean()
                .add(u.square().add(i.square()).add(j.square()).sum().mul(l2 / user.length));
        loss.backward();
        optimizer.step(List.of(users, items, itemBias));
        return loss.value().item();
    }

    /**
     * @param user a user id
     * @return the score of every item for the user
     */
    public float[] scores(int user) {
        Tensor u = users.value().slice(0, user, user + 1);
        return u.matmul(items.value().transpose()).add(itemBias.value().reshape(1, items.shape().dim(0)))
                .toArray();
    }

    /** @return the embedding size */
    public int factors() {
        return factors;
    }
}
