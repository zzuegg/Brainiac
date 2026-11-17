package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * LARS (You, Gitman and Ginsburg 2017, <a href="https://arxiv.org/abs/1708.03888">arXiv:1708.03888</a>):
 * momentum SGD with a layer-wise learning rate {@code trust * |w| / (|g| + weightDecay |w|)},
 * so that every layer's step is a fixed fraction of its weight norm. The large-batch
 * optimizer for convolutional networks.
 */
public final class Lars extends SlotOptimizer {

    private final float momentum;
    private final float weightDecay;
    private final float trust;

    /**
     * @param learningRate the global step size
     * @param momentum the momentum, for example 0.9
     * @param weightDecay the weight decay, for example {@code 1e-4f}
     * @param trust the trust coefficient, for example {@code 0.001f} (or 1 for small models)
     */
    public Lars(float learningRate, float momentum, float weightDecay, float trust) {
        super(learningRate, "velocity");
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.trust = trust;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        float wn = LearningMath.norm(p.value());
        float gn = LearningMath.norm(g);
        float local = wn > 0 && gn > 0 ? trust * wn / (gn + weightDecay * wn) : 1;
        Tensor d = weightDecay > 0 ? g.add(p.value().mul(weightDecay)) : g;
        Tensor v = d.mul(learningRate() * local);
        if (s[0] != null) {
            v = s[0].mul(momentum).add(v);
        }
        p.assign(p.value().sub(v));
        return new Tensor[] {v};
    }

    @Override
    public String toString() {
        return "LARS(lr=" + learningRate() + ", momentum=" + momentum + ")";
    }
}
