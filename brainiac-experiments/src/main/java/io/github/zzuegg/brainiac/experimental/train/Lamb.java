package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * LAMB (You et al. 2020, <a href="https://arxiv.org/abs/1904.00962">arXiv:1904.00962</a>):
 * AdamW whose step for every parameter tensor is rescaled by the <em>trust ratio</em>
 * {@code |w| / |update|}, so each layer moves by a fixed fraction of its own size. Lets
 * very large batches train with one learning rate for all layers.
 */
public final class Lamb extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;

    /**
     * @param learningRate the step size, for example {@code 1e-3f}
     * @param beta1 decay of the gradient average
     * @param beta2 decay of the squared-gradient average
     * @param epsilon added to the denominator
     * @param weightDecay the decoupled weight decay
     */
    public Lamb(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        super(learningRate, "m", "v");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    /**
     * @param learningRate the step size
     * @param weightDecay the weight decay
     * @return LAMB with {@code beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6}
     */
    public static Lamb of(float learningRate, float weightDecay) {
        return new Lamb(learningRate, 0.9f, 0.999f, 1e-6f, weightDecay);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor v = s[1] == null ? g.square().mul(1 - beta2) : s[1].mul(beta2).add(g.square().mul(1 - beta2));
        Tensor mHat = m.div(1 - (float) Math.pow(beta1, t));
        Tensor vHat = v.div(1 - (float) Math.pow(beta2, t));
        Tensor r = mHat.div(vHat.sqrt().add(epsilon));
        if (weightDecay > 0) {
            r = r.add(p.value().mul(weightDecay));
        }
        float wn = LearningMath.norm(p.value());
        float rn = LearningMath.norm(r);
        float trust = wn > 0 && rn > 0 ? wn / rn : 1;
        p.assign(p.value().sub(r.mul(learningRate() * trust)));
        return new Tensor[] {m, v};
    }

    @Override
    public String toString() {
        return "LAMB(lr=" + learningRate() + ", weightDecay=" + weightDecay + ")";
    }
}
