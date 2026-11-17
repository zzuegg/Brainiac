package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Cross-entropy with label smoothing (Szegedy et al. 2016,
 * <a href="https://arxiv.org/abs/1512.00567">arXiv:1512.00567</a>): the target keeps
 * {@code 1 - epsilon} on the true class and spreads {@code epsilon} evenly over all classes,
 * so the model is not pushed to infinitely confident logits; better calibration and
 * generalisation. Like the core {@link Loss#CROSS_ENTROPY} it takes probabilities from a
 * softmax output and one-hot (or soft) targets, or class ids.
 */
public final class SmoothedCrossEntropy implements Loss {

    private final float epsilon;

    /**
     * @param epsilon the smoothing, for example 0.1
     */
    public SmoothedCrossEntropy(float epsilon) {
        if (!(epsilon >= 0 && epsilon < 1)) {
            throw new IllegalArgumentException("epsilon must be in [0, 1)");
        }
        this.epsilon = epsilon;
    }

    /**
     * Smooths targets.
     *
     * @param target {@code [rows, classes]} one-hot or soft targets
     * @param epsilon the smoothing
     * @return {@code (1 - epsilon) target + epsilon / classes}
     */
    public static Tensor smooth(Tensor target, float epsilon) {
        return target.mul(1 - epsilon).add(epsilon / target.shape().dim(-1));
    }

    @Override
    public Var compute(Var prediction, Tensor target) {
        Tensor t = Loss.classIds(prediction.shape(), target.shape()) ? Loss.oneHot(target, prediction.shape()) : target;
        return Loss.CROSS_ENTROPY.compute(prediction, smooth(t, epsilon));
    }

    @Override
    public String toString() {
        return "SmoothedCrossEntropy(" + epsilon + ")";
    }
}
