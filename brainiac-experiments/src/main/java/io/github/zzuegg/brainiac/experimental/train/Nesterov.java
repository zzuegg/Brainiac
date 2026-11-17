package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * SGD with Nesterov momentum (Sutskever et al. 2013): {@code v = mu v + g},
 * {@code w -= lr (g + mu v)}, the "look-ahead" form PyTorch uses. The core has heavy-ball
 * momentum only; DiLoCo's outer optimizer needs this one.
 */
public final class Nesterov extends SlotOptimizer {

    private final float momentum;

    /**
     * @param learningRate the step size
     * @param momentum the momentum, for example 0.9
     */
    public Nesterov(float learningRate, float momentum) {
        super(learningRate, "velocity");
        if (momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException("momentum must be in [0, 1)");
        }
        this.momentum = momentum;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor v = s[0] == null ? g : s[0].mul(momentum).add(g);
        p.assign(p.value().sub(g.add(v.mul(momentum)).mul(learningRate())));
        return new Tensor[] {v};
    }

    @Override
    public String toString() {
        return "Nesterov(lr=" + learningRate() + ", momentum=" + momentum + ")";
    }
}
