package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Adan (Xie et al. 2022, <a href="https://arxiv.org/abs/2208.06677">arXiv:2208.06677</a>):
 * adaptive Nesterov momentum without the extra gradient evaluation. Besides Adam's
 * averages it tracks the average gradient <em>difference</em> {@code g_k - g_{k-1}},
 * which estimates the look-ahead gradient, and uses it both in the step and in the
 * second moment. Strong on vision and language models with the same settings.
 *
 * <p>The betas follow the paper's convention ({@code m = (1 - b1) m + b1 g}), so the
 * defaults are small: {@code b1 = 0.02, b2 = 0.08, b3 = 0.01}.
 */
public final class Adan extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float beta3;
    private final float epsilon;
    private final float weightDecay;

    /**
     * @param learningRate the step size
     * @param beta1 weight of the new gradient in the first moment
     * @param beta2 weight of the new difference in its average
     * @param beta3 weight of the new value in the second moment
     * @param epsilon added to the denominator
     * @param weightDecay the proximal weight decay
     */
    public Adan(float learningRate, float beta1, float beta2, float beta3, float epsilon, float weightDecay) {
        super(learningRate, "m", "v", "n", "previous");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.beta3 = beta3;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    /**
     * @param learningRate the step size, for example {@code 1e-3f}
     * @param weightDecay the weight decay, for example {@code 0.02f}
     * @return Adan with the paper's betas
     */
    public static Adan of(float learningRate, float weightDecay) {
        return new Adan(learningRate, 0.02f, 0.08f, 0.01f, 1e-8f, weightDecay);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor diff = s[3] == null ? g.mul(0) : g.sub(s[3]);
        Tensor m = s[0] == null ? g : s[0].mul(1 - beta1).add(g.mul(beta1));
        Tensor v = s[1] == null ? diff : s[1].mul(1 - beta2).add(diff.mul(beta2));
        Tensor u = g.add(diff.mul(1 - beta2)).square();
        Tensor n = s[2] == null ? u : s[2].mul(1 - beta3).add(u.mul(beta3));
        // bias correction as in the reference implementation
        float c1 = 1 - (float) Math.pow(1 - beta1, t);
        float c2 = 1 - (float) Math.pow(1 - beta2, t);
        float c3 = 1 - (float) Math.pow(1 - beta3, t);
        Tensor step = m.div(c1).add(v.div(c2).mul(1 - beta2)).div(n.div(c3).sqrt().add(epsilon));
        Tensor w = p.value().sub(step.mul(learningRate()));
        if (weightDecay > 0) {
            w = w.div(1 + learningRate() * weightDecay);
        }
        p.assign(w);
        return new Tensor[] {m, v, n, g};
    }

    @Override
    public String toString() {
        return "Adan(lr=" + learningRate() + ")";
    }
}
