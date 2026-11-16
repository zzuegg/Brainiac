package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** SGD with optional momentum. */
final class Momentum implements Optimizer {
    private float learningRate;
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
        List<Tensor> updated = new ArrayList<>(2 * parameters.size());
        List<Tensor> used = new ArrayList<>(2 * parameters.size());
        List<Tensor> created = Tensor.created(() -> {
            for (Var p : parameters) {
                Tensor g = p.grad();
                if (g == null) {
                    continue;
                }
                used.add(g);
                Tensor update = g;
                if (momentum > 0) {
                    Tensor v = velocity.get(p);
                    if (v != null) {
                        used.add(v);
                    }
                    update = v == null ? g : v.mul(momentum).add(g);
                    velocity.put(p, update);
                    updated.add(update);
                }
                p.assign(p.value().sub(update.mul(learningRate)));
                p.zeroGrad();
                updated.add(p.value());
            }
        });
        Tensor.evaluate(updated);   // on a GPU, the whole step as one kernel
        // the gradients, old velocities and values on the way: only this optimizer saw them
        used.addAll(created);
        Tensor.release(used, updated);
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got " + learningRate);
        }
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return momentum == 0 ? "SGD(lr=" + learningRate + ")" : "Momentum(lr=" + learningRate + ", momentum=" + momentum + ")";
    }
}
