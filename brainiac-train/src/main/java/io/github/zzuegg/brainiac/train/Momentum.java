package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** SGD with optional momentum, plain or Nesterov's. */
final class Momentum extends StatefulOptimizer {
    private final float momentum;
    private final boolean nesterov;

    Momentum(float learningRate, float momentum, boolean nesterov) {
        super(learningRate, "velocity");
        if (momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException("momentum must be in [0, 1), got " + momentum);
        }
        this.momentum = momentum;
        this.nesterov = nesterov;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor velocity = momentum == 0 || s[0] == null ? g : s[0].mul(momentum).add(g);
        Tensor update = nesterov && momentum > 0 ? g.add(velocity.mul(momentum)) : velocity;
        p.assign(p.value().sub(update.mul(learningRate())));
        return new Tensor[] {momentum > 0 ? velocity : null};
    }

    @Override
    public String toString() {
        return momentum == 0 ? "SGD(lr=" + learningRate() + ")"
                : (nesterov ? "Nesterov" : "Momentum") + "(lr=" + learningRate() + ", momentum=" + momentum + ")";
    }
}
