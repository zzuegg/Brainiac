package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Outlier exposure: train the classifier on auxiliary outliers (never the test OOD set) so it
 * is unconfident off the data manifold.
 *
 * <ul>
 *   <li>{@link #uniformLoss}: cross-entropy from the outliers' softmax to the uniform
 *       distribution, {@code logsumexp(z) - mean(z)} (Hendrycks et al., ICLR 2019,
 *       <a href="https://arxiv.org/abs/1812.04606">arXiv:1812.04606</a>; weight 0.5).</li>
 *   <li>{@link #energyLoss}: squared hinges that push the in-distribution energy
 *       {@code E = -logsumexp(z)} below {@code mIn} and the outlier energy above {@code mOut}
 *       (Liu et al., NeurIPS 2020, <a href="https://arxiv.org/abs/2010.03759">arXiv:2010.03759</a>;
 *       weight 0.1).</li>
 * </ul>
 *
 * <p>{@link #loss} wraps either as a core {@link Loss} on logits: targets are class ids, and a
 * negative id marks an outlier row, so a mixed batch drops into any training loop that takes a
 * {@code Loss}.
 */
public final class OutlierExposure {

    /** The outlier term. */
    public enum Variant {
        /** cross-entropy to uniform */
        UNIFORM,
        /** energy margins */
        ENERGY
    }

    private OutlierExposure() {
    }

    /**
     * @param outlierLogits {@code [m, classes]}
     * @return the mean cross-entropy from the softmax to the uniform distribution
     */
    public static Var uniformLoss(Var outlierLogits) {
        return outlierLogits.logsumexp(1, false).sub(outlierLogits.mean(1, false)).mean();
    }

    /**
     * @param inLogits in-distribution logits {@code [n, classes]}
     * @param outlierLogits outlier logits {@code [m, classes]}
     * @param mIn the in-distribution energy margin, such as -7 for 10 classes (scale with the logits)
     * @param mOut the outlier energy margin, such as -2
     * @return {@code mean(relu(E_in - mIn)^2) + mean(relu(mOut - E_out)^2)}
     */
    public static Var energyLoss(Var inLogits, Var outlierLogits, float mIn, float mOut) {
        Var eIn = inLogits.logsumexp(1, false).neg();
        Var eOut = outlierLogits.logsumexp(1, false).neg();
        return eIn.sub(mIn).relu().square().mean().add(eOut.neg().add(mOut).relu().square().mean());
    }

    /**
     * A core {@link Loss} on logits: cross-entropy on rows with a class id, plus the weighted
     * outlier term on rows with a negative id.
     *
     * @param variant the outlier term
     * @param weight its weight
     * @param mIn the in-distribution energy margin ({@link Variant#ENERGY})
     * @param mOut the outlier energy margin ({@link Variant#ENERGY})
     * @return the loss
     */
    public static Loss loss(Variant variant, float weight, float mIn, float mOut) {
        return (logits, target) -> {
            float[] t = target.toArray();
            int in = 0;
            for (float v : t) {
                in += v >= 0 ? 1 : 0;
            }
            float[] inRows = new float[in];
            float[] outRows = new float[t.length - in];
            float[] ids = new float[in];
            int a = 0;
            int b = 0;
            for (int i = 0; i < t.length; i++) {
                if (t[i] >= 0) {
                    ids[a] = t[i];
                    inRows[a++] = i;
                } else {
                    outRows[b++] = i;
                }
            }
            Var inLogits = logits.gather(0, Tensor.of(Shape.of(in), inRows));
            Var ce = Functional.crossEntropy(inLogits, Tensor.of(Shape.of(in), ids));
            if (outRows.length == 0) {
                return ce;
            }
            Var outLogits = logits.gather(0, Tensor.of(Shape.of(outRows.length), outRows));
            Var extra = variant == Variant.UNIFORM ? uniformLoss(outLogits) : energyLoss(inLogits, outLogits, mIn, mOut);
            return ce.add(extra.mul(weight));
        };
    }
}
