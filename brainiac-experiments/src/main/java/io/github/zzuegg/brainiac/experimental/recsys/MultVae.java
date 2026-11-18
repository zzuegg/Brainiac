package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Mult-VAE, a variational autoencoder for collaborative filtering (Liang et al., 2018,
 * <a href="https://arxiv.org/abs/1802.05814">arXiv:1802.05814</a>). The encoder reads a user's
 * whole interaction row (L2-normalised, with input dropout while training) and outputs a Gaussian
 * over a latent taste vector; the decoder maps a sample to logits over all items, read as a
 * <em>multinomial</em> distribution: the reconstruction loss is
 * {@code -sum_i x_i log softmax(logits)_i}, which suits top-k ranking better than Gaussian or
 * logistic likelihoods. The KL term is weighted by {@code beta}, annealed from 0 up to a cap
 * (the paper finds {@code beta} well below 1 best).
 *
 * <p>As a {@link Layer} it maps rows {@code [batch, items]} to logits {@code [batch, items]} and
 * reports {@code beta * KL} of its last training pass as {@link #auxiliaryLoss()}.
 */
public final class MultVae implements Layer {

    private final int items;
    private final Var w1;
    private final Var b1;
    private final Var wMu;
    private final Var wLogVar;
    private final Var wd1;
    private final Var bd1;
    private final Var wd2;
    private final Var bd2;
    private final SplittableRandom random;
    private final float dropout;
    private final float betaCap;
    private final int annealSteps;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int steps;
    private Var lastKl;

    /**
     * Creates the model.
     *
     * @param items the number of items
     * @param hidden the hidden width of encoder and decoder
     * @param latent the latent size
     * @param dropout the input dropout rate while training
     * @param betaCap the largest KL weight
     * @param annealSteps training steps over which the KL weight rises linearly to the cap
     * @param seed the seed
     */
    public MultVae(int items, int hidden, int latent, float dropout, float betaCap, int annealSteps, long seed) {
        this.items = items;
        this.random = new SplittableRandom(seed);
        this.dropout = dropout;
        this.betaCap = betaCap;
        this.annealSteps = Math.max(1, annealSteps);
        w1 = RecsysOps.matrix(random, items, hidden);
        b1 = RecsysOps.zeros(hidden);
        wMu = RecsysOps.matrix(random, hidden, latent);
        wLogVar = RecsysOps.matrix(random, hidden, latent);
        wd1 = RecsysOps.matrix(random, latent, hidden);
        bd1 = RecsysOps.zeros(hidden);
        wd2 = RecsysOps.matrix(random, hidden, items);
        bd2 = RecsysOps.zeros(items);
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public MultVae optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /** @return the KL weight of the next training step */
    public float beta() {
        return Math.min(betaCap, betaCap * steps / (float) annealSteps);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var x = RecsysOps.normalize(input);
        if (training && dropout > 0) {
            float[] keep = new float[(int) x.shape().size()];
            for (int i = 0; i < keep.length; i++) {
                keep[i] = random.nextDouble() < dropout ? 0 : 1 / (1 - dropout);
            }
            x = x.mul(Tensor.of(x.shape(), keep));
        }
        Var h = x.matmul(w1).add(b1).tanh();
        Var mu = h.matmul(wMu);
        Var logVar = h.matmul(wLogVar);
        Var z = training ? Functional.reparameterize(mu, logVar, random) : mu;
        lastKl = training ? Functional.gaussianKl(mu, logVar).mul(beta()) : null;
        return z.matmul(wd1).add(bd1).tanh().matmul(wd2).add(bd2);
    }

    @Override
    public Var auxiliaryLoss() {
        return lastKl;
    }

    /**
     * The multinomial log-likelihood loss plus {@code beta * KL}.
     *
     * @param rows {@code [batch, items]} interaction rows
     * @return the loss
     */
    public Var loss(Tensor rows) {
        Var logits = forward(Var.of(rows), true);
        Var nll = Activation.LOG_SOFTMAX.forward(logits).mul(rows).sum(1, false).mean().neg();
        return nll.add(lastKl);
    }

    /**
     * One optimizer step on a batch of rows; advances the KL annealing.
     *
     * @param rows {@code [batch, items]} interaction rows
     * @return the loss
     */
    public double trainStep(Tensor rows) {
        Var loss = loss(rows);
        loss.backward();
        optimizer.step(parameters());
        steps++;
        return loss.value().item();
    }

    /**
     * @param row one user's interaction row
     * @return the score (logit) of every item
     */
    public float[] scores(double[] row) {
        float[] f = new float[row.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) row[i];
        }
        return Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, items), f)), false)).value().toArray();
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1, b1, wMu, wLogVar, wd1, bd1, wd2, bd2);
    }

    @Override
    public String toString() {
        return "MultVae(" + items + " items)";
    }
}
