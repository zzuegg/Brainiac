package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Self-paced learning (Kumar, Packer and Koller 2010; Jiang et al. 2015): a curriculum
 * chosen by the model itself. Each sample's loss decides whether it is "easy" enough to
 * learn from now: with the hard regulariser its weight is {@code 1} when its loss is below
 * the threshold {@code lambda} and {@code 0} otherwise; with the linear (soft) one it is
 * {@code max(0, 1 - loss / lambda)}. The threshold grows by a factor every epoch until
 * every sample counts. Outliers and mislabeled samples, whose losses stay high, are
 * ignored the longest.
 */
public final class SelfPacedLearning {

    private final boolean soft;
    private final float growth;
    private float threshold;

    /**
     * @param threshold the starting loss threshold {@code lambda}
     * @param growth the factor applied each epoch, for example 1.3
     * @param soft whether to use linear (soft) weights instead of 0/1
     */
    public SelfPacedLearning(float threshold, float growth, boolean soft) {
        this.threshold = threshold;
        this.growth = growth;
        this.soft = soft;
    }

    /** @return the current threshold */
    public float threshold() {
        return threshold;
    }

    /** Raises the threshold at the end of an epoch. */
    public void nextEpoch() {
        threshold *= growth;
    }

    /**
     * Cross-entropy per sample, without the mean.
     *
     * @param logits {@code [rows, classes]}
     * @param ids {@code [rows]} class ids
     * @return {@code [rows]} losses
     */
    public static Var perSampleCrossEntropy(Var logits, Tensor ids) {
        int rows = logits.shape().dim(0);
        Tensor target = LearningMath.oneHot(ids, logits.shape().dim(1));
        return Activation.LOG_SOFTMAX.forward(logits).mul(target).sum(1, false).neg().reshape(rows);
    }

    /**
     * @param losses {@code [rows]} per-sample losses
     * @return the self-paced weights, {@code [rows]}
     */
    public Tensor weights(Tensor losses) {
        if (soft) {
            return losses.div(threshold).neg().add(1).maximum(Tensor.full(Shape.of(), 0f));
        }
        return Tensor.full(Shape.of(), threshold).greater(losses);
    }

    /**
     * The weighted mean loss: weights are constants, so samples above the threshold get no
     * gradient.
     *
     * @param losses {@code [rows]} per-sample losses
     * @return a scalar
     */
    public Var weightedLoss(Var losses) {
        Tensor w = weights(losses.value());
        float count = Math.max(1e-6f, w.sum().item());
        return losses.mul(w).sum().div(count);
    }
}
