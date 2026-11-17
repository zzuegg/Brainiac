package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * ADOPT (Taniguchi et al. 2024, <a href="https://arxiv.org/abs/2411.02853">arXiv:2411.02853</a>):
 * an Adam variant that converges at the optimal rate for <em>any</em> {@code beta2}. Adam can
 * diverge because its second moment includes the current gradient, which correlates the
 * step size with the step direction. ADOPT removes the correlation by two changes:
 *
 * <ul>
 *   <li>the gradient is normalised by the <em>previous</em> second moment,
 *       {@code n = g / max(sqrt(v_{t-1}), eps)}, and the second moment is updated after;</li>
 *   <li>the normalisation happens before the momentum, {@code m = beta1 m + (1 - beta1) n},
 *       and the step is simply {@code w -= lr * m}.</li>
 * </ul>
 * The first step only initialises {@code v = g^2}. As in the authors' code, {@code n} is
 * clipped to {@code ±t^0.25}, which keeps the first steps (when {@code v} is a single
 * sample) stable.
 */
public final class Adopt extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final boolean clip;

    /**
     * @param learningRate the step size, as for Adam
     * @param beta1 the momentum, 0.9
     * @param beta2 the second-moment decay, 0.9999 in the paper; any value in {@code [0, 1)} converges
     * @param epsilon the smallest denominator, {@code 1e-6}
     * @param weightDecay the decoupled weight decay
     * @param clip whether to clip the normalised gradient to {@code ±t^0.25}
     */
    public Adopt(float learningRate, float beta1, float beta2, float epsilon, float weightDecay, boolean clip) {
        super(learningRate, "m", "v");
        if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || !(epsilon > 0) || weightDecay < 0) {
            throw new IllegalArgumentException("bad ADOPT settings");
        }
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.clip = clip;
    }

    /**
     * @param learningRate the step size
     * @param beta2 the second-moment decay
     * @return ADOPT with {@code beta1 = 0.9}, {@code eps = 1e-6}, clipping and no weight decay
     */
    public static Adopt of(float learningRate, float beta2) {
        return new Adopt(learningRate, 0.9f, beta2, 1e-6f, 0, true);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        if (s[1] == null) {
            return new Tensor[] {null, g.square()};   // step 1 only measures the gradient's scale
        }
        Tensor n = g.div(s[1].sqrt().maximum(Tensor.full(Shape.of(), epsilon)));
        if (clip) {
            float c = (float) Math.pow(t - 1, 0.25);
            n = n.maximum(Tensor.full(Shape.of(), -c)).minimum(Tensor.full(Shape.of(), c));
        }
        Tensor m = s[0] == null ? n.mul(1 - beta1) : s[0].mul(beta1).add(n.mul(1 - beta1));
        float lr = learningRate();
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        p.assign(w.sub(m.mul(lr)));
        Tensor v = s[1].mul(beta2).add(g.square().mul(1 - beta2));
        return new Tensor[] {m, v};
    }

    @Override
    public String toString() {
        return "ADOPT(lr=" + learningRate() + ", beta2=" + beta2 + ")";
    }
}
