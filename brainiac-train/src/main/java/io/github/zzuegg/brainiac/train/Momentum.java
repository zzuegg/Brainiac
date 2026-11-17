package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** SGD with optional momentum. */
final class Momentum extends StatefulOptimizer {
    private final float momentum;

    Momentum(float learningRate, float momentum) {
        super(learningRate, "velocity");
        if (momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException("momentum must be in [0, 1), got " + momentum);
        }
        this.momentum = momentum;
    }

    @Override
    Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor update = momentum == 0 || s[0] == null ? g : s[0].mul(momentum).add(g);
        p.assign(p.value().sub(update.mul(learningRate())));
        return new Tensor[] {momentum > 0 ? update : null};
    }

    @Override
    public String toString() {
        return momentum == 0 ? "SGD(lr=" + learningRate() + ")"
                : "Momentum(lr=" + learningRate() + ", momentum=" + momentum + ")";
    }
}
