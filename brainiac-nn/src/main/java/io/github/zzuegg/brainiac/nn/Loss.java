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

    /** Mean absolute error: for numbers, less swayed by outliers than {@link #MSE}. */
    Loss L1 = (p, t) -> checked(p, t).sub(t).abs().mean();

    /**
     * Kullback-Leibler divergence {@code KL(target || prediction)} between probability rows,
     * summed over the last axis and averaged over the rows: for distillation.
     */
    Loss KL_DIVERGENCE = (p, t) -> {
        int rows = Math.toIntExact(checked(p, t).shape().size() / p.shape().dim(-1));
        Tensor tLogT = t.mul(t.add(Loss.EPSILON).log());
        return Var.of(tLogT).sub(p.add(Loss.EPSILON).log().mul(t)).sum().mul(1f / rows);
    };

    /** One minus the cosine similarity of each row with its target, averaged over the rows. */
    Loss COSINE = (p, t) -> {
        Var dot = checked(p, t).mul(t).sum(-1, false);
        Var norm = p.square().sum(-1, false).add(Loss.EPSILON).sqrt().mul(Var.of(t.square().sum(-1, false).add(Loss.EPSILON).sqrt()));
        return dot.div(norm).neg().add(1).mean();
    };

    /** Hinge loss for targets of -1 and 1 and raw scores: {@code mean(max(0, 1 - t * p))}, as SVMs and GANs use. */
    Loss HINGE = (p, t) -> checked(p, t).mul(t).neg().add(1).maximum(0).mean();

    /**
     * The Huber (smooth L1) loss: squared below {@code delta}, linear above.
     *
     * @param delta where the loss turns linear, positive
     * @return the loss
     */
    static Loss huber(float delta) {
        if (!(delta > 0)) {
            throw new IllegalArgumentException("delta must be positive: " + delta);
        }
        return (p, t) -> {
            Var d = checked(p, t).sub(t).abs();
            Var small = d.minimum(delta);   // the quadratic part, capped at delta
            return small.square().mul(0.5f).add(d.sub(small).mul(delta)).mean();
        };
    }

    /**
     * Cross-entropy from raw scores (logits), with a log-softmax inside for stability: the
     * loss for classifiers without a softmax layer. Targets are class ids, one per row
     * (negative ids are skipped, like padding), or full probability rows.
     *
     * @param labelSmoothing the share of each target spread evenly over all classes, in
     *     {@code [0, 1)}; 0 for plain cross-entropy
     * @return the loss, averaged over the rows that count
     */
    static Loss crossEntropyFromLogits(float labelSmoothing) {
        if (!(labelSmoothing >= 0 && labelSmoothing < 1)) {
            throw new IllegalArgumentException("label smoothing must be in [0, 1): " + labelSmoothing);
        }
        return (p, t) -> {
            int classes = p.shape().dim(-1);
            Var logp = Activation.logSoftmax(-1).forward(p);
            Tensor target;
            float count;
            if (classIds(p.shape(), t.shape())) {
                Tensor valid = t.greater(Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(), -0.5f, t.device()))
                        .to(p.dtype());
                count = Math.max(1, valid.sum().item());
                Tensor oneHot = oneHot(t, p.shape()).to(p.dtype());
                int[] dims = p.shape().toArray();
                dims[dims.length - 1] = 1;
                target = oneHot.mul(1 - labelSmoothing).add(labelSmoothing / classes).mul(valid.reshape(dims));
            } else {
                count = Math.toIntExact(checked(p, t).shape().size() / classes);
                target = labelSmoothing == 0 ? t : t.mul(1 - labelSmoothing).add(labelSmoothing / classes);
            }
            return logp.mul(target).sum().mul(-1f / count);
        };
    }

    /**
     * Focal loss (Lin et al., 2017) for 0/1 targets and probabilities from a sigmoid: binary
     * cross-entropy with each example scaled by {@code (1 - p_t)^gamma}, so easy examples
     * count less; for detection and other imbalanced problems.
     *
     * @param gamma the focusing power; 0 is binary cross-entropy
     * @param alpha the weight of positives, in {@code [0, 1]}, negatives get {@code 1 - alpha};
     *     0.5 weighs them equally (and halves the loss)
     * @return the loss
     */
    static Loss focal(float gamma, float alpha) {
        return (p, t) -> {
            Var pt = checked(p, t).mul(t).add(p.neg().add(1).mul(t.neg().add(1)));   // probability of the truth
            Tensor weight = t.mul(alpha).add(t.neg().add(1).mul(1 - alpha));
            Var ce = pt.add(EPSILON).log().neg();
            return pt.neg().add(1).maximum(0).pow(gamma).mul(ce).mul(weight).mean();
        };
    }

    /**
     * Dice loss for segmentation: one minus the soft Dice coefficient
     * {@code (2 |p t| + smooth) / (|p| + |t| + smooth)} of each example, averaged.
     *
     * @param smooth added to both sides, so empty masks give 0; usually 1
     * @return the loss
     */
    static Loss dice(float smooth) {
        return (p, t) -> {
            int batch = checked(p, t).shape().dim(0);
            int per = Math.toIntExact(p.shape().size() / batch);
            Var pr = p.reshape(batch, per);
            Tensor tr = t.reshape(batch, per);
            Var inter = pr.mul(tr).sum(1, false).mul(2).add(smooth);
            Var total = pr.sum(1, false).add(Var.of(tr.sum(1, false))).add(smooth);
            return inter.div(total).neg().add(1).mean();
        };
    }

    /**
     * The quantile (pinball) loss for probabilistic forecasts: prediction column {@code j}
     * estimates quantile {@code quantiles[j]} of the target, under-predictions costing
     * {@code q} and over-predictions {@code 1 - q} per unit.
     *
     * @param quantiles the quantile of each prediction column, each in {@code (0, 1)}
     * @return the loss over predictions {@code [batch, quantiles]} and targets {@code [batch, 1]}
     */
    static Loss quantile(float... quantiles) {
        for (float q : quantiles) {
            if (!(q > 0 && q < 1)) {
                throw new IllegalArgumentException("quantiles must be in (0, 1): " + q);
            }
        }
        float[] qs = quantiles.clone();
        return (p, t) -> {
            int batch = p.shape().dim(0);
            if (p.shape().rank() != 2 || p.shape().dim(1) != qs.length || t.size() != batch) {
                throw new IllegalArgumentException("quantile loss needs predictions [batch, " + qs.length
                        + "] and one target per row, got " + p.shape() + " and " + t.shape());
            }
            Tensor q = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(1, qs.length), qs, t.device())
                    .to(p.value().dtype());
            Var err = Var.of(t.reshape(batch, 1)).sub(p);   // positive: predicted too low
            return err.mul(q).maximum(err.mul(q.sub(1))).mean();
        };
    }

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
