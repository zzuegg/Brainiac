package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * AdEMAMix (Pagliardini, Ablin and Grangier 2024,
 * <a href="https://arxiv.org/abs/2409.03137">arXiv:2409.03137</a>): AdamW plus a second,
 * very slow gradient average ({@code beta3 = 0.9999}) mixed into the step with weight
 * {@code alpha}. A single EMA cannot both react to recent gradients and remember old ones;
 * two can, and models forget their training data more slowly. {@code alpha} ramps up
 * linearly over {@code warmup} steps for stability.
 */
public final class AdEMAMix extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float beta3;
    private final float alpha;
    private final int warmup;
    private final float weightDecay;

    /**
     * @param learningRate the step size
     * @param beta1 decay of the fast average, 0.9
     * @param beta2 decay of the squared-gradient average, 0.999
     * @param beta3 decay of the slow average, 0.9999
     * @param alpha weight of the slow average, for example 5
     * @param warmup steps over which {@code alpha} ramps up from 0
     * @param weightDecay the decoupled weight decay
     */
    public AdEMAMix(float learningRate, float beta1, float beta2, float beta3, float alpha, int warmup,
            float weightDecay) {
        super(learningRate, "m1", "m2", "v");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.beta3 = beta3;
        this.alpha = alpha;
        this.warmup = warmup;
        this.weightDecay = weightDecay;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m1 = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor m2 = s[1] == null ? g.mul(1 - beta3) : s[1].mul(beta3).add(g.mul(1 - beta3));
        Tensor v = s[2] == null ? g.square().mul(1 - beta2) : s[2].mul(beta2).add(g.square().mul(1 - beta2));
        float a = warmup > 0 ? alpha * Math.min(1f, (float) t / warmup) : alpha;
        Tensor num = m1.div(1 - (float) Math.pow(beta1, t)).add(m2.mul(a));
        Tensor step = num.div(v.div(1 - (float) Math.pow(beta2, t)).sqrt().add(1e-8f));
        Tensor w = weightDecay > 0 ? p.value().mul(1 - learningRate() * weightDecay) : p.value();
        p.assign(w.sub(step.mul(learningRate())));
        return new Tensor[] {m1, m2, v};
    }

    @Override
    public String toString() {
        return "AdEMAMix(lr=" + learningRate() + ", alpha=" + alpha + ")";
    }
}
