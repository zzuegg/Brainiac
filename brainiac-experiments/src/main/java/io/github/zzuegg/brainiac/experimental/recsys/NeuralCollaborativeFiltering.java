package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Neural collaborative filtering, NeuMF (He et al., 2017,
 * <a href="https://arxiv.org/abs/1708.05031">arXiv:1708.05031</a>). Two branches score a
 * (user, item) pair: generalised matrix factorization (GMF), the elementwise product of a user
 * and an item embedding, and an MLP over the concatenation of a second, separate pair of
 * embeddings. A final linear layer fuses both into one interaction logit, trained with binary
 * cross-entropy on observed pairs and sampled negatives.
 *
 * <p>As a {@link Layer} it maps {@code [batch, 2]} rows of (user id, item id) to logits
 * {@code [batch, 1]}.
 */
public final class NeuralCollaborativeFiltering implements Layer {

    private final int items;
    private final Var gmfUsers;
    private final Var gmfItems;
    private final Var mlpUsers;
    private final Var mlpItems;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var fuse;
    private final Var bias;
    private final SplittableRandom random;
    private Optimizer optimizer = Optimizer.adam(0.005f);

    /**
     * Creates the model.
     *
     * @param users the number of users
     * @param items the number of items
     * @param factors the GMF embedding size (the MLP embeddings are twice that)
     * @param seed the seed
     */
    public NeuralCollaborativeFiltering(int users, int items, int factors, long seed) {
        this.items = items;
        this.random = new SplittableRandom(seed);
        gmfUsers = RecsysOps.normal(random, 0.1f, users, factors);
        gmfItems = RecsysOps.normal(random, 0.1f, items, factors);
        mlpUsers = RecsysOps.normal(random, 0.1f, users, 2 * factors);
        mlpItems = RecsysOps.normal(random, 0.1f, items, 2 * factors);
        w1 = RecsysOps.matrix(random, 4 * factors, 2 * factors);
        b1 = RecsysOps.zeros(2 * factors);
        w2 = RecsysOps.matrix(random, 2 * factors, factors);
        b2 = RecsysOps.zeros(factors);
        fuse = RecsysOps.matrix(random, 2 * factors, 1);
        bias = RecsysOps.zeros(1);
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public NeuralCollaborativeFiltering optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    @Override
    public Var forward(Var input) {
        Tensor u = input.value().slice(1, 0, 1).reshape(input.shape().dim(0));
        Tensor i = input.value().slice(1, 1, 2).reshape(input.shape().dim(0));
        Var gmf = gmfUsers.gather(0, u).mul(gmfItems.gather(0, i));
        Var mlp = Var.concat(1, mlpUsers.gather(0, u), mlpItems.gather(0, i)).matmul(w1).add(b1).relu().matmul(w2)
                .add(b2).relu();
        return Var.concat(1, gmf, mlp).matmul(fuse).add(bias);
    }

    /**
     * One binary cross-entropy step: each observed pair against {@code negatives} random items of
     * the same user.
     *
     * @param user user ids
     * @param positive an item each user interacted with
     * @param negatives sampled negatives per positive
     * @return the loss
     */
    public double trainStep(int[] user, int[] positive, int negatives) {
        int n = user.length * (1 + negatives);
        float[] x = new float[2 * n];
        float[] y = new float[n];
        int k = 0;
        for (int b = 0; b < user.length; b++) {
            for (int j = 0; j <= negatives; j++) {
                x[2 * k] = user[b];
                x[2 * k + 1] = j == 0 ? positive[b] : random.nextInt(items);
                y[k++] = j == 0 ? 1 : 0;
            }
        }
        Var logit = forward(Var.of(Tensor.of(Shape.of(n, 2), x)));
        Var loss = logit.softplus().sub(logit.mul(Tensor.of(Shape.of(n, 1), y))).mean();
        loss.backward();
        optimizer.step(parameters());
        return loss.value().item();
    }

    /**
     * @param user a user id
     * @return the logit of every item for the user
     */
    public float[] scores(int user) {
        float[] x = new float[2 * items];
        for (int i = 0; i < items; i++) {
            x[2 * i] = user;
            x[2 * i + 1] = i;
        }
        return Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(items, 2), x)))).value().toArray();
    }

    @Override
    public List<Var> parameters() {
        return List.of(gmfUsers, gmfItems, mlpUsers, mlpItems, w1, b1, w2, b2, fuse, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    @Override
    public String toString() {
        return "NeuMF(" + items + " items)";
    }
}
