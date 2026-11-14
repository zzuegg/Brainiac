package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Measures how far predictions are from targets; training minimizes it. All losses
 * return the mean over the batch.
 */
@FunctionalInterface
public interface Loss {

    /** Mean squared error: for predicting numbers. */
    Loss MSE = (p, t) -> checked(p, t).sub(t).square().mean();

    /**
     * Cross-entropy for one-hot targets and probabilities from a softmax output:
     * {@code -mean over rows of sum(target * log(prediction))}.
     */
    Loss CROSS_ENTROPY = (p, t) -> {
        int rows = checked(p, t).shape().dim(0);
        return p.add(Loss.EPSILON).log().mul(t).sum().mul(-1f / rows);
    };

    /**
     * Binary cross-entropy for 0/1 targets and probabilities from a sigmoid output.
     */
    Loss BINARY_CROSS_ENTROPY = (p, t) -> {
        Var pos = checked(p, t).add(Loss.EPSILON).log().mul(t);
        Var neg = p.neg().add(1 + Loss.EPSILON).log().mul(t.neg().add(1));
        return pos.add(neg).mean().neg();
    };

    /** Added inside logarithms so a probability of exactly 0 does not give infinity. */
    float EPSILON = 1e-7f;

    /**
     * Computes the loss.
     *
     * @param prediction the model output, shape {@code [batch, outputs]}
     * @param target the expected output, same shape
     * @return a scalar
     */
    Var compute(Var prediction, Tensor target);

    private static Var checked(Var prediction, Tensor target) {
        if (!prediction.shape().equals(target.shape())) {
            throw new IllegalArgumentException(
                    "prediction shape " + prediction.shape() + " does not match target shape " + target.shape());
        }
        return prediction;
    }
}
