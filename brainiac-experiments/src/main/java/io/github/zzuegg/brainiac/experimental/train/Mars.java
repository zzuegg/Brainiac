package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * MARS (Yuan et al. 2024, <a href="https://arxiv.org/abs/2411.10438">arXiv:2411.10438</a>):
 * variance reduction for preconditioned optimizers. A STORM-style correction adds the
 * scaled change of the gradient to the gradient,
 * {@code c = g_t + gamma * beta1 / (1 - beta1) * (g_t - g_{t-1})}, clips {@code c} to unit
 * norm, and feeds it to AdamW's (or Lion's) moments in place of the raw gradient. The
 * correction cancels much of the mini-batch noise that momentum alone only averages, which
 * lets GPT-2 models reach a given loss in fewer tokens than AdamW.
 *
 * <p>This is the practical "MARS-approx" variant: {@code g_{t-1}} is the previous batch's
 * gradient at the previous weights, so no second gradient evaluation is needed. One extra
 * tensor per parameter holds it.
 */
public final class Mars extends SlotOptimizer {

    /** Which preconditioner the corrected gradient feeds. */
    public enum Base {
        /** AdamW moments, the paper's default MARS-AdamW. */
        ADAMW,
        /** Lion's sign step, MARS-Lion. */
        LION
    }

    private final Base base;
    private final float beta1;
    private final float beta2;
    private final float gamma;
    private final float weightDecay;

    /**
     * @param learningRate the step size
     * @param base the preconditioner
     * @param beta1 the first-moment decay, 0.95
     * @param beta2 the second-moment decay, 0.99
     * @param gamma the strength of the correction, 0.025
     * @param weightDecay the decoupled weight decay
     */
    public Mars(float learningRate, Base base, float beta1, float beta2, float gamma, float weightDecay) {
        super(learningRate, "m", "v", "previous");
        if (beta1 <= 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || gamma < 0 || weightDecay < 0) {
            throw new IllegalArgumentException("bad MARS settings");
        }
        this.base = base;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.gamma = gamma;
        this.weightDecay = weightDecay;
    }

    /**
     * MARS-AdamW with the paper's settings ({@code betas = (0.95, 0.99), gamma = 0.025}).
     *
     * @param learningRate the step size, as for AdamW
     * @param weightDecay the decoupled weight decay
     * @return the optimizer
     */
    public static Mars adamW(float learningRate, float weightDecay) {
        return new Mars(learningRate, Base.ADAMW, 0.95f, 0.99f, 0.025f, weightDecay);
    }

    /**
     * MARS-Lion ({@code betas = (0.9, 0.99), gamma = 0.025}).
     *
     * @param learningRate the step size, as for Lion
     * @param weightDecay the decoupled weight decay
     * @return the optimizer
     */
    public static Mars lion(float learningRate, float weightDecay) {
        return new Mars(learningRate, Base.LION, 0.9f, 0.99f, 0.025f, weightDecay);
    }

    /**
     * The variance-reduced gradient, clipped to unit norm.
     *
     * @param g this step's gradient
     * @param previous the previous step's gradient, or {@code null}
     * @param gamma the correction strength
     * @param beta1 the first-moment decay
     * @return {@code c}
     */
    public static Tensor corrected(Tensor g, Tensor previous, float gamma, float beta1) {
        Tensor c = previous == null ? g : g.add(g.sub(previous).mul(gamma * beta1 / (1 - beta1)));
        Tensor norm = c.square().sum().sqrt();
        return c.div(norm.maximum(Tensor.full(Shape.of(), 1f)));
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor c = corrected(g, s[2], gamma, beta1);
        float lr = learningRate();
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        Tensor previous = g.add(0f);   // a copy, in case the caller reuses the gradient's memory
        if (base == Base.LION) {
            Tensor mixed = s[0] == null ? c.mul(1 - beta1) : s[0].mul(beta1).add(c.mul(1 - beta1));
            p.assign(w.sub(mixed.sign().mul(lr)));
            Tensor m = s[0] == null ? c.mul(1 - beta2) : s[0].mul(beta2).add(c.mul(1 - beta2));
            return new Tensor[] {m, null, previous};
        }
        Tensor m = s[0] == null ? c.mul(1 - beta1) : s[0].mul(beta1).add(c.mul(1 - beta1));
        Tensor v = s[1] == null ? c.square().mul(1 - beta2) : s[1].mul(beta2).add(c.square().mul(1 - beta2));
        float c1 = 1 - (float) Math.pow(beta1, t);
        float c2 = 1 - (float) Math.pow(beta2, t);
        p.assign(w.sub(m.div(c1).div(v.div(c2).sqrt().add(1e-8f)).mul(lr)));
        return new Tensor[] {m, v, previous};
    }

    @Override
    public String toString() {
        return "MARS-" + base + "(lr=" + learningRate() + ")";
    }
}
