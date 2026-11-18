package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * An epinet (Osband et al. 2023, <a href="https://arxiv.org/abs/2107.08924">Epistemic neural
 * networks</a>): a small network added to a base model that turns it into an "ensemble" indexed
 * by a random vector {@code z ~ N(0, I)}.
 *
 * <p>It reads the base network's features {@code phi(x)} (gradient stopped) and the index
 * {@code z}, and outputs {@code sigma_L(phi, z) + sigma_P(phi, z)}: a learned term plus a fixed,
 * randomly initialised prior network of the same shape. Both are linear in {@code z}
 * ({@code sigma(phi, z) = MLP([phi, z]) z}), so different indices give different functions;
 * the prior keeps them apart where there is no data, and training (averaging the loss over a
 * few indices per batch) only pulls them together where the data constrains them. Sampling
 * {@code z} at test time gives an ensemble of correlated predictions whose disagreement is the
 * epistemic uncertainty, at the price of one base pass plus a few tiny epinet passes, and its
 * joint predictions are what the paper's joint log-loss scores.
 *
 * <p>As a {@link Layer} it maps features to the epinet's additive logits with a fresh index
 * drawn per call; {@link #forward(Var, Tensor)} takes an explicit index.
 */
public final class Epinet implements Layer {

    private final int features;
    private final int outputs;
    private final int indexDim;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Tensor p1;
    private final Tensor pb1;
    private final Tensor p2;
    private final float priorScale;
    private final SplittableRandom random;

    /**
     * Creates an epinet.
     *
     * @param features the base feature width
     * @param outputs the number of outputs (classes or regression targets)
     * @param indexDim the index dimension, for example 8
     * @param hidden the hidden width of both the learned and the prior network
     * @param priorScale the weight of the fixed prior network, for example 1
     * @param random the initialisation and index source
     */
    public Epinet(int features, int outputs, int indexDim, int hidden, float priorScale, SplittableRandom random) {
        this.features = features;
        this.outputs = outputs;
        this.indexDim = indexDim;
        this.priorScale = priorScale;
        this.random = random;
        int in = features + indexDim;
        float a1 = (float) Math.sqrt(2.0 / in);
        float a2 = (float) Math.sqrt(1.0 / hidden);
        w1 = Var.param(Tensor.normal(Shape.of(in, hidden), 0, a1, random));
        b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        w2 = Var.param(Tensor.zeros(Shape.of(hidden, outputs * indexDim)));
        b2 = Var.param(Tensor.zeros(Shape.of(1, outputs * indexDim)));
        p1 = Tensor.normal(Shape.of(in, hidden), 0, a1, random);
        pb1 = Tensor.normal(Shape.of(1, hidden), 0, 0.1f, random);
        p2 = Tensor.normal(Shape.of(hidden, outputs * indexDim), 0, a2, random);
    }

    /**
     * Draws an index.
     *
     * @return {@code z ~ N(0, I)} of the index dimension
     */
    public Tensor index() {
        return Tensor.normal(Shape.of(indexDim), 0, 1, random);
    }

    /**
     * The epinet's additive output for one index.
     *
     * @param features the base features {@code [B, features]} (detached here)
     * @param z the index {@code [indexDim]}
     * @return {@code sigma_L + priorScale sigma_P}, {@code [B, outputs]}
     */
    public Var forward(Var features, Tensor z) {
        int b = features.shape().dim(0);
        float[] zv = z.toArray();
        float[] rep = new float[b * indexDim];
        for (int i = 0; i < b; i++) {
            System.arraycopy(zv, 0, rep, i * indexDim, indexDim);
        }
        Var in = Var.concat(1, features.detach(), Var.of(Tensor.of(Shape.of(b, indexDim), rep)));
        Var zCol = Var.of(Tensor.of(Shape.of(indexDim, 1), zv));
        Var learned = in.matmul(w1).add(b1).relu().matmul(w2).add(b2);
        Var prior = in.detach().matmul(Var.of(p1)).add(Var.of(pb1)).relu().matmul(Var.of(p2));
        Var both = learned.add(prior.mul(priorScale));
        return both.reshape(b * outputs, indexDim).matmul(zCol).reshape(b, outputs);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, index());
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1, b1, w2, b2);
    }

    /**
     * The feature width this epinet reads.
     *
     * @return the feature width
     */
    public int features() {
        return features;
    }

    @Override
    public String toString() {
        return "Epinet(" + features + " -> " + outputs + ", index " + indexDim + ")";
    }
}
