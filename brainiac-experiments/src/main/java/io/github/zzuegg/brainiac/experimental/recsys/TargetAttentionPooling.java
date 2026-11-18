package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Target attention, the pooling of the Deep Interest Network (DIN; Zhou et al., 2018,
 * <a href="https://arxiv.org/abs/1706.06978">arXiv:1706.06978</a>). Instead of compressing a
 * user's behaviour history into one fixed vector, each history item is weighted by its relevance
 * to the candidate being scored: a small "activation unit" MLP reads
 * {@code [h_i, c, h_i - c, h_i * c]} and outputs a weight {@code w_i}, and the pooled interest is
 * {@code sum_i w_i h_i}. As in the paper the weights are not softmax-normalised, so their sum can
 * also express how strongly the history supports the candidate.
 *
 * <p>As a {@link MultiInputLayer} it takes the history {@code [batch, steps, features]}, the
 * candidate {@code [batch, features]} and optionally a {@code [batch, steps]} 0/1 mask of real
 * steps, and returns {@code [batch, features]}.
 */
public final class TargetAttentionPooling implements MultiInputLayer {

    private final int features;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    /**
     * Creates the layer.
     *
     * @param features the embedding width
     * @param hidden the activation unit's hidden width
     * @param random the initialization source
     */
    public TargetAttentionPooling(int features, int hidden, SplittableRandom random) {
        this.features = features;
        w1 = RecsysOps.matrix(random, 4 * features, hidden);
        b1 = RecsysOps.zeros(hidden);
        w2 = RecsysOps.matrix(random, hidden, 1);
        b2 = RecsysOps.zeros(1);
    }

    /**
     * The attention weights of each history step.
     *
     * @param history {@code [batch, steps, features]}
     * @param candidate {@code [batch, features]}
     * @return {@code [batch, steps, 1]}
     */
    public Var weights(Var history, Var candidate) {
        int b = history.shape().dim(0);
        int n = history.shape().dim(1);
        Var c = candidate.reshape(b, 1, features).expand(Shape.of(b, n, features));
        Var in = Var.concat(2, history, c, history.sub(c), history.mul(c)).reshape(b * n, 4 * features);
        return in.matmul(w1).add(b1).sigmoid().matmul(w2).add(b2).reshape(b, n, 1);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var history = inputs.get(0);
        Var w = weights(history, inputs.get(1));
        if (inputs.size() > 2) {
            Var mask = inputs.get(2);
            w = w.mul(mask.value().reshape(mask.shape().dim(0), mask.shape().dim(1), 1));
        }
        return history.mul(w).sum(1, false);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(1);
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1, b1, w2, b2);
    }

    @Override
    public String toString() {
        return "TargetAttentionPooling(" + features + ")";
    }
}
