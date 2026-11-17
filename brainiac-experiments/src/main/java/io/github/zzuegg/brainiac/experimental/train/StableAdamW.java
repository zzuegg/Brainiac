package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Two fixes for AdamW's instability at high learning rates, where a stale second moment
 * (one that has not yet seen a sudden larger gradient) lets steps blow up:
 *
 * <ul>
 *   <li>{@link Mode#STABLE}: StableAdamW (Wortsman et al. 2023,
 *       <a href="https://arxiv.org/abs/2304.13013">arXiv:2304.13013</a>) clips each tensor's
 *       learning rate by the root mean square of {@code g / sqrt(v)}: {@code lr_t = lr /
 *       max(1, RMS(g / sqrt(max(v, eps^2))))}. When the current gradient is much larger than
 *       {@code v} expects, the step shrinks for that tensor instead of jumping. It is
 *       AdaFactor's update clipping, carried over to AdamW.</li>
 *   <li>{@link Mode#ATAN2}: Adam-atan2 (Everett et al. 2024,
 *       <a href="https://arxiv.org/abs/2407.05872">arXiv:2407.05872</a>) replaces
 *       {@code m / (sqrt(v) + eps)} by {@code a * atan2(m, b * sqrt(v))}: bounded by
 *       {@code a * pi / 2} however small {@code v} is, scale-invariant, and free of the
 *       epsilon hyperparameter.</li>
 * </ul>
 */
public final class StableAdamW extends SlotOptimizer {

    /** The variant. */
    public enum Mode {
        /** Per-tensor update clipping. */
        STABLE,
        /** {@code atan2} in place of the division, no epsilon. */
        ATAN2
    }

    private static final float ATAN2_A = 1.2732395f;   // 4 / pi, the paper's a = b = 1.27
    private static final float ATAN2_B = 1f;

    private final Mode mode;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;

    /**
     * @param learningRate the step size
     * @param mode the variant
     * @param beta1 the first-moment decay
     * @param beta2 the second-moment decay
     * @param epsilon the epsilon (ignored by {@link Mode#ATAN2})
     * @param weightDecay the decoupled weight decay
     */
    public StableAdamW(float learningRate, Mode mode, float beta1, float beta2, float epsilon, float weightDecay) {
        super(learningRate, "m", "v");
        if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || weightDecay < 0) {
            throw new IllegalArgumentException("bad StableAdamW settings");
        }
        this.mode = mode;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    /**
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return StableAdamW with AdamW's usual betas
     */
    public static StableAdamW stable(float learningRate, float weightDecay) {
        return new StableAdamW(learningRate, Mode.STABLE, 0.9f, 0.999f, 1e-8f, weightDecay);
    }

    /**
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return Adam-atan2 with AdamW's usual betas
     */
    public static StableAdamW atan2(float learningRate, float weightDecay) {
        return new StableAdamW(learningRate, Mode.ATAN2, 0.9f, 0.999f, 0, weightDecay);
    }

    /**
     * Element-wise {@code atan2(y, x)} for {@code x >= 0}, from tensor operations only: the
     * half-angle identity {@code atan2(y, x) = 2 atan(y / (x + sqrt(x^2 + y^2)))} brings the
     * argument into {@code [-1, 1]}, where a degree-9 odd polynomial (Abramowitz and Stegun
     * 4.4.49) is accurate to {@code 1e-5}. {@code atan2(0, 0) = 0}.
     *
     * @param y the numerators, any sign
     * @param x the denominators, not negative
     * @return the angles, in {@code [-pi/2, pi/2]}
     */
    public static Tensor atan2(Tensor y, Tensor x) {
        Tensor r = x.square().add(y.square()).sqrt();
        Tensor u = y.div(x.add(r).add(1e-30f));
        Tensor u2 = u.square();
        Tensor poly = u2.mul(0.0208351f).add(-0.0851330f).mul(u2).add(0.1801410f).mul(u2).add(-0.3302995f).mul(u2)
                .add(0.9998660f);
        return u.mul(poly).mul(2f);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor v = s[1] == null ? g.square().mul(1 - beta2) : s[1].mul(beta2).add(g.square().mul(1 - beta2));
        float c1 = 1 - (float) Math.pow(beta1, t);
        float c2 = 1 - (float) Math.pow(beta2, t);
        Tensor mHat = m.div(c1);
        Tensor vHat = v.div(c2);
        float lr = learningRate();
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        Tensor step;
        if (mode == Mode.ATAN2) {
            step = atan2(mHat, vHat.sqrt().mul(ATAN2_B)).mul(ATAN2_A * lr);
        } else {
            Tensor floor = Tensor.full(Shape.of(), Math.max(epsilon * epsilon, 1e-30f));
            Tensor rms = g.square().div(vHat.maximum(floor)).mean().sqrt();
            Tensor rate = rms.maximum(Tensor.full(Shape.of(), 1f)).pow(-1f).mul(lr);
            step = mHat.div(vHat.sqrt().add(epsilon)).mul(rate);
        }
        p.assign(w.sub(step));
        return new Tensor[] {m, v};
    }

    @Override
    public String toString() {
        return (mode == Mode.STABLE ? "StableAdamW" : "Adam-atan2") + "(lr=" + learningRate() + ")";
    }
}
