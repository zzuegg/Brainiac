package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Explanation-guided training, "right for the right reasons" (Ross, Hughes and Doshi-Velez,
 * IJCAI 2017, <a href="https://arxiv.org/abs/1703.03717">arXiv:1703.03717</a>): given an
 * annotation mask {@code A} marking inputs that should not matter, the loss adds
 * {@code lambda ||A * d/dx sum_k log p_k(x)||^2} to the cross-entropy, penalising input-gradient
 * attribution on the annotated region by double backpropagation, so the model is pushed to other
 * features. {@link Penalty#INTEGRATED_GRADIENTS} penalises a few-step integrated-gradients
 * attribution {@code (x - x0) * mean_a grad log p_y(x0 + a (x - x0))} instead, in the spirit of
 * the attribution-based penalties of Rieger et al. (ICML 2020,
 * <a href="https://arxiv.org/abs/1909.13584">arXiv:1909.13584</a>).
 *
 * <p>Masks may be given for all examples, or only for a few annotated ones (rows of zeros for
 * the rest). The penalty works with any {@link Layer} (a {@code Model} included) whose ops
 * support second derivatives.
 */
public final class ExplanationRegularizer {

    /** The penalised attribution. */
    public enum Penalty {
        /** The input gradient of the summed log-probabilities. */
        INPUT_GRADIENT,
        /** Integrated gradients of the true class's log-probability from a zero baseline. */
        INTEGRATED_GRADIENTS
    }

    private final Penalty penalty;
    private final float lambda;
    private final int steps;

    /**
     * Creates the regulariser.
     *
     * @param penalty the attribution
     * @param lambda the penalty weight
     * @param steps integration steps (integrated gradients only)
     */
    public ExplanationRegularizer(Penalty penalty, float lambda, int steps) {
        this.penalty = penalty;
        this.lambda = lambda;
        this.steps = steps;
    }

    /**
     * The penalty of a batch.
     *
     * @param model the classifier, inputs to logits
     * @param x the inputs {@code [b, d]}
     * @param labels the labels {@code [b]}
     * @param mask the annotation {@code [b, d]}: 1 where the input should not matter
     * @return {@code lambda} times the mean masked squared attribution
     */
    public Var penalty(Layer model, Tensor x, Tensor labels, Tensor mask) {
        int b = x.shape().dim(0);
        if (penalty == Penalty.INPUT_GRADIENT) {
            Var in = Var.param(x);
            Var lp = model.forward(in, true).logSoftmax(1).sum();
            Var g = Var.gradients(lp, List.of(in), true).get(0);
            return g.mul(mask).square().sum().mul(lambda / b);
        }
        int classes = model.forward(Var.of(x.slice(0, 0, 1))).shape().dim(1);
        Tensor oneHot = labels.reshape(b, 1).equalTo(Tensor.arange(classes).reshape(1, classes));
        Var total = null;
        for (int s = 1; s <= steps; s++) {
            float a = s / (float) steps;
            Var in = Var.param(x.mul(a));
            Var lp = model.forward(in, true).logSoftmax(1).mul(oneHot).sum();
            Var g = Var.gradients(lp, List.of(in), true).get(0);
            total = total == null ? g : total.add(g);
        }
        Var ig = total.mul(x).div((float) steps);
        return ig.mul(mask).square().sum().mul(lambda / b);
    }

    /**
     * Cross-entropy plus the penalty.
     *
     * @param model the classifier
     * @param x the inputs
     * @param labels the labels
     * @param mask the annotation
     * @return the training loss
     */
    public Var loss(Layer model, Tensor x, Tensor labels, Tensor mask) {
        return Functional.crossEntropy(model.forward(Var.of(x), true), labels).add(penalty(model, x, labels, mask));
    }
}
