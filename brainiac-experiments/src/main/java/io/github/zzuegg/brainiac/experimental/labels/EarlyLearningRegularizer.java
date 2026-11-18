package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Early-learning regularisation (ELR; Liu, Niles-Weed, Razavian, Fernandez-Granda, NeurIPS 2020,
 * <a href="https://arxiv.org/abs/2007.00151">arXiv:2007.00151</a>). It keeps a temporal-ensemble
 * target {@code t_i} per training example, an EMA {@code t_i <- beta t_i + (1 - beta) p_i} of the
 * example's past (clamped, renormalised) predictions, and adds {@code lambda log(1 - <p_i, t_i>)}
 * to the loss. Early in training the targets are mostly right, so the term pulls the model
 * towards its own early predictions, and its gradient cancels the push towards memorising wrong
 * labels without detecting the noisy examples explicitly.
 */
public final class EarlyLearningRegularizer {

    private final float[][] targets;
    private final double beta;
    private final float lambda;

    /**
     * Creates the regulariser.
     *
     * @param examples the training-set size
     * @param classes the number of classes
     * @param beta the target EMA momentum, 0.7 on CIFAR-10
     * @param lambda the regularisation weight, 3 on CIFAR-10
     */
    public EarlyLearningRegularizer(int examples, int classes, double beta, float lambda) {
        this.targets = new float[examples][classes];
        this.beta = beta;
        this.lambda = lambda;
    }

    /**
     * Updates the targets of a batch and returns the regulariser.
     *
     * @param logits the batch's logits, with gradient
     * @param indices the batch's training-set indices
     * @return {@code lambda * mean_i log(1 - <p_i, t_i>)}
     */
    public Var loss(Var logits, int[] indices) {
        float[][] p = SoftTargets.probabilities(logits.value(), 1f);
        int classes = p[0].length;
        float[] t = new float[indices.length * classes];
        for (int i = 0; i < indices.length; i++) {
            float[] target = targets[indices[i]];
            double sum = 0;
            for (int c = 0; c < classes; c++) {
                sum += Math.clamp(p[i][c], 1e-4f, 1 - 1e-4f);
            }
            for (int c = 0; c < classes; c++) {
                float pc = (float) (Math.clamp(p[i][c], 1e-4f, 1 - 1e-4f) / sum);
                target[c] = (float) (beta * target[c] + (1 - beta) * pc);
                t[i * classes + c] = target[c];
            }
        }
        Var prob = SoftTargets.softmax(logits).clamp(1e-4f, 1 - 1e-4f);
        Var inner = prob.mul(Tensor.of(Shape.of(indices.length, classes), t)).sum(-1, false);
        return inner.neg().add(1f).log().mean().mul(lambda);
    }

    /**
     * @param index a training example
     * @return its current target; a copy
     */
    public float[] target(int index) {
        return targets[index].clone();
    }
}
