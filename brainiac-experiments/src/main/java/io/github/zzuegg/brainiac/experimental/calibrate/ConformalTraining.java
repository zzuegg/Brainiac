package io.github.zzuegg.brainiac.experimental.calibrate;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Conformal training (ConfTr; Stutz et al., ICLR 2022,
 * <a href="https://arxiv.org/abs/2110.09192">arXiv:2110.09192</a>): split conformal
 * prediction simulated inside every mini-batch so the model learns probabilities whose
 * conformal sets are small. The first half of the batch calibrates a threshold {@code τ} on
 * the true-class probabilities with a differentiable quantile, the second half forms soft sets
 * {@code C_k = sigmoid((p_k - τ) / T)}, and the loss is
 * <pre>
 *   sizeWeight · mean max(0, Σ_k C_k - κ)
 * + coverageWeight · mean (1 - C_y)
 * + ceWeight · cross-entropy(batch)
 * </pre>
 * The guarantee still comes from an ordinary post-hoc calibration step (any
 * {@code ConformalClassifier} or {@link ConformalSets} predictor) after training.
 *
 * <p>The quantile is the linearly interpolated order statistic of the sorted true-class
 * probabilities ({@link Var#sort} passes the gradient back to the two neighbouring samples),
 * a simpler stand-in for the paper's smooth sorting with the same first-order behaviour.
 */
public final class ConformalTraining {

    /**
     * Loss settings.
     *
     * @param alpha the target miscoverage
     * @param temperature the sigmoid temperature of the soft sets
     * @param sizeWeight the weight of the size loss
     * @param coverageWeight the weight of the coverage loss
     * @param ceWeight the weight of cross-entropy
     * @param kappa set sizes up to {@code κ} are free
     */
    public record Settings(float alpha, float temperature, float sizeWeight, float coverageWeight, float ceWeight,
            float kappa) {

        /** @return the paper-like defaults at α = 0.1: T 0.1, size 1, coverage 0, CE 0.1, κ 1 */
        public static Settings defaults() {
            return new Settings(0.1f, 0.1f, 1f, 0f, 0.1f, 1f);
        }
    }

    private ConformalTraining() {
    }

    /**
     * The ConfTr loss of a batch.
     *
     * @param logits the batch logits {@code [n, k]}, {@code n >= 4}
     * @param ids the class ids {@code [n]}
     * @param settings the settings
     * @return the scalar loss
     */
    public static Var loss(Var logits, Tensor ids, Settings settings) {
        int n = logits.shape().dim(0);
        int k = logits.shape().dim(1);
        int h = n / 2;
        float[] y = ids.toArray();
        Tensor oneHot = oneHot(y, k);
        Var p = logits.softmax(1);
        Var calTrue = p.slice(0, 0, h).mul(oneHot.slice(0, 0, h)).sum(1, false);
        Var sorted = calTrue.sort(0, false);
        double pos = Math.max(0, Math.min(h - 1, settings.alpha() * (h + 1) - 1));
        int lo = (int) Math.floor(pos);
        int hi = Math.min(h - 1, lo + 1);
        float f = (float) (pos - lo);
        Var tau = sorted.slice(0, lo, lo + 1).mul(1 - f).add(sorted.slice(0, hi, hi + 1).mul(f));
        Var pred = p.slice(0, h, n);
        Var soft = pred.sub(tau).div(settings.temperature()).sigmoid();
        Var size = soft.sum(1, false).sub(settings.kappa()).relu().mean();
        Var loss = size.mul(settings.sizeWeight());
        if (settings.coverageWeight() > 0) {
            Var cover = soft.mul(oneHot.slice(0, h, n)).sum(1, false).neg().add(1f).mean();
            loss = loss.add(cover.mul(settings.coverageWeight()));
        }
        if (settings.ceWeight() > 0) {
            loss = loss.add(Functional.crossEntropy(logits, ids).mul(settings.ceWeight()));
        }
        return loss;
    }

    /**
     * One-hot rows.
     *
     * @param ids class ids
     * @param classes the number of classes
     * @return {@code [n, classes]}
     */
    public static Tensor oneHot(float[] ids, int classes) {
        float[] v = new float[ids.length * classes];
        for (int i = 0; i < ids.length; i++) {
            v[i * classes + (int) ids[i]] = 1;
        }
        return Tensor.of(Shape.of(ids.length, classes), v);
    }
}
