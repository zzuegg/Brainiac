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
     * Cross-entropy for probabilities from a softmax output:
     * {@code -mean over rows of sum(target * log(prediction))}. Targets are one-hot, or
     * class ids with one value per prediction row (the last axis left out), which saves
     * memory when there are many classes, such as the tokens of a language model.
     */
    Loss CROSS_ENTROPY = (p, t) -> {
        Tensor target = classIds(p.shape(), t.shape()) ? oneHot(t, p.shape()) : t;
        int rows = checked(p, target).shape().dim(0);
        return p.add(Loss.EPSILON).log().mul(target).sum().mul(-1f / rows);
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

    /**
     * The loss of each example rather than their mean, for weighting, mixup, per-sample
     * gradients or picking hard examples: a vector {@code [batch]} whose mean is
     * {@code loss.compute(prediction, target)}. {@link #MSE}, {@link #CROSS_ENTROPY} and
     * {@link #BINARY_CROSS_ENTROPY} are computed in one pass; other losses one example at a time.
     *
     * @param loss the loss
     * @param prediction the model output, examples on axis 0
     * @param target the expected output
     * @return the loss of each example
     */
    static Var perSample(Loss loss, Var prediction, Tensor target) {
        int batch = prediction.shape().dim(0);
        int[] rows = {batch, Math.toIntExact(prediction.shape().size() / batch)};
        if (loss == MSE) {
            return checked(prediction, target).sub(target).square().reshape(rows).mean(1, false);
        }
        if (loss == BINARY_CROSS_ENTROPY) {
            Var pos = checked(prediction, target).add(EPSILON).log().mul(target);
            Var neg = prediction.neg().add(1 + EPSILON).log().mul(target.neg().add(1));
            return pos.add(neg).reshape(rows).mean(1, false).neg();
        }
        if (loss == CROSS_ENTROPY) {
            Tensor t = classIds(prediction.shape(), target.shape()) ? oneHot(target, prediction.shape()) : target;
            return checked(prediction, t).add(EPSILON).log().mul(t).reshape(rows).sum(1, false).neg();
        }
        Var[] each = new Var[batch];
        for (int i = 0; i < batch; i++) {
            each[i] = loss.compute(prediction.slice(0, i, i + 1), target.slice(0, i, i + 1)).reshape(1);
        }
        return Var.concat(0, each);
    }

    /**
     * Whether targets are class ids for these predictions: one value per row of the
     * prediction's last axis, rather than a full one-hot row.
     *
     * @param prediction the prediction shape, classes on the last axis
     * @param target the target shape
     * @return {@code true} for class ids
     */
    static boolean classIds(io.github.zzuegg.brainiac.tensor.Shape prediction,
                            io.github.zzuegg.brainiac.tensor.Shape target) {
        int classes = prediction.dim(-1);
        return classes > 1 && !prediction.equals(target) && target.size() * classes == prediction.size();
    }

    /**
     * Turns class ids into one-hot rows, on the ids' device. Inside a loss the comparison
     * fuses with the rest, so the one-hot rows are never stored.
     *
     * @param ids whole-number class ids
     * @param shape the one-hot shape: the ids' rows followed by the classes
     * @return 1 at each id and 0 elsewhere, shaped {@code shape}
     */
    static Tensor oneHot(Tensor ids, io.github.zzuegg.brainiac.tensor.Shape shape) {
        int classes = shape.dim(-1);
        int[] dims = shape.toArray();
        dims[dims.length - 1] = 1;
        return ids.reshape(dims).equalTo(Tensor.arange(classes).to(ids.device()));
    }

    private static Var checked(Var prediction, Tensor target) {
        if (!prediction.shape().equals(target.shape())) {
            throw new IllegalArgumentException(
                    "prediction shape " + prediction.shape() + " does not match target shape " + target.shape());
        }
        return prediction;
    }
}
