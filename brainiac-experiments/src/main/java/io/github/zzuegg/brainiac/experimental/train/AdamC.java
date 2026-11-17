package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Map;
import java.util.function.Predicate;

/**
 * AdamC (Defazio et al. 2025, "Why Gradients Rapidly Increase Near the End of Training",
 * <a href="https://arxiv.org/abs/2506.02285">arXiv:2506.02285</a>): AdamW with a weight
 * decay that follows the learning-rate schedule. For a weight followed by a normalisation
 * layer the loss does not depend on the weight's scale, so its norm settles where decay
 * balances the update; at that steady state the gradient norm is proportional to
 * {@code sqrt(2 wd / lr)}. With AdamW's fixed {@code wd} and a decaying {@code lr} the
 * ratio grows, which is why gradient norms climb at the end of cosine or WSD runs. AdamC
 * decays normalised weights with {@code wd * lr_t / lr_max} instead, so the ratio stays
 * fixed and the gradient norm stays flat.
 *
 * <p>The peak rate is the largest learning rate set so far (call {@link #peakRate(float)}
 * to fix it up front when the schedule starts below its peak, as with a warm-up).
 * {@code normalized} chooses the parameters that get the corrected decay; the default
 * takes matrices. Others get AdamW's plain decoupled decay.
 */
public final class AdamC extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final Predicate<Var> normalized;
    private float peak;

    /**
     * @param learningRate the step size
     * @param beta1 decay of the gradient average
     * @param beta2 decay of the squared-gradient average
     * @param epsilon added to the denominator
     * @param weightDecay the decoupled weight decay at the peak rate
     * @param normalized the parameters that get the corrected decay
     */
    public AdamC(float learningRate, float beta1, float beta2, float epsilon, float weightDecay,
            Predicate<Var> normalized) {
        super(learningRate, "m", "v");
        if (weightDecay < 0) {
            throw new IllegalArgumentException("weightDecay must not be negative");
        }
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.normalized = normalized;
        this.peak = learningRate;
    }

    /**
     * AdamC with Adam's usual betas and the corrected decay on every matrix.
     *
     * @param learningRate the peak step size
     * @param weightDecay the weight decay
     * @return the optimizer
     */
    public static AdamC of(float learningRate, float weightDecay) {
        return new AdamC(learningRate, 0.9f, 0.999f, 1e-8f, weightDecay, p -> p.shape().rank() == 2);
    }

    /**
     * Fixes the peak learning rate the decay is measured against.
     *
     * @param rate the peak rate
     * @return this optimizer
     */
    public AdamC peakRate(float rate) {
        this.peak = rate;
        return this;
    }

    /** @return the weight decay in effect for normalised parameters now */
    public float correctedDecay() {
        return weightDecay * learningRate() / peak;
    }

    @Override
    public void learningRate(float learningRate) {
        super.learningRate(learningRate);
        peak = Math.max(peak, learningRate);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor v = s[1] == null ? g.square().mul(1 - beta2) : s[1].mul(beta2).add(g.square().mul(1 - beta2));
        float c1 = 1 - (float) Math.pow(beta1, t);
        float c2 = 1 - (float) Math.pow(beta2, t);
        float lr = learningRate();
        float wd = normalized.test(p) ? correctedDecay() : weightDecay;
        Tensor w = wd > 0 ? p.value().mul(1 - lr * wd) : p.value();
        p.assign(w.sub(m.div(c1).div(v.div(c2).sqrt().add(epsilon)).mul(lr)));
        return new Tensor[] {m, v};
    }

    @Override
    protected Map<String, Tensor> extraState() {
        return Map.of("peakRate", Tensor.of(Shape.of(1), peak));
    }

    @Override
    protected void loadExtraState(Map<String, Tensor> saved) {
        Tensor p = saved.get("peakRate");
        if (p != null) {
            peak = p.item();
        }
    }

    @Override
    public String toString() {
        return "AdamC(lr=" + learningRate() + ", weightDecay=" + weightDecay + ")";
    }
}
