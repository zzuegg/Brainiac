package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** SGD with optional momentum. */
final class Momentum implements Optimizer {
    private final float learningRate;
    private final float momentum;
    private final Map<Var, Tensor> velocity = new IdentityHashMap<>();

    Momentum(float learningRate, float momentum) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got " + learningRate);
        }
        if (momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException("momentum must be in [0, 1), got " + momentum);
        }
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var p : parameters) {
            Tensor g = p.grad();
            if (g == null) {
                continue;
            }
            Tensor update = g;
            if (momentum > 0) {
                Tensor v = velocity.get(p);
                update = v == null ? g : v.mul(momentum).add(g);
                velocity.put(p, update);
            }
            p.assign(p.value().sub(update.mul(learningRate)));
            p.zeroGrad();
        }
    }

    @Override
    public String toString() {
        return momentum == 0 ? "SGD(lr=" + learningRate + ")" : "Momentum(lr=" + learningRate + ", momentum=" + momentum + ")";
    }
}
