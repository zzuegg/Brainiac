package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Keeps a fixed sparsity pattern during training: an {@link Optimizer} wrapper that masks the
 * gradients of pruned weights before the inner step and re-applies the 0/1 masks after it, so a
 * pruned network can be fine-tuned (recovery after one-shot pruning, training after pruning at
 * initialisation) without pruned weights growing back. The masks come from a pruner or from the
 * weights' current zeros ({@link #ofZeros}).
 */
public final class FixedSparsityOptimizer implements Optimizer {

    private final Optimizer inner;
    private final Map<Var, Tensor> masks;

    /**
     * @param inner the optimizer
     * @param masks a 0/1 mask per pruned weight; weights not in the map train freely
     */
    public FixedSparsityOptimizer(Optimizer inner, Map<Var, Tensor> masks) {
        this.inner = inner;
        this.masks = new IdentityHashMap<>(masks);
    }

    /**
     * @param inner the optimizer
     * @param weights pruned weights; their exact zeros stay zero
     * @return the wrapper
     */
    public static FixedSparsityOptimizer ofZeros(Optimizer inner, List<Var> weights) {
        return new FixedSparsityOptimizer(inner, zeroMasks(weights));
    }

    /**
     * @param weights some weights
     * @return the 0/1 mask of the non-zero entries of each
     */
    public static Map<Var, Tensor> zeroMasks(List<Var> weights) {
        Map<Var, Tensor> m = new IdentityHashMap<>();
        for (Var w : weights) {
            m.put(w, w.value().abs().greater(Tensor.zeros(w.shape())));
        }
        return m;
    }

    /**
     * Multiplies every weight by its mask.
     *
     * @param masks the masks
     */
    public static void apply(Map<Var, Tensor> masks) {
        masks.forEach((w, m) -> w.assign(w.value().mul(m)));
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var p : parameters) {
            Tensor m = masks.get(p);
            if (m != null && p.grad() != null) {
                p.setGrad(p.grad().mul(m));
            }
        }
        inner.step(parameters);
        apply(masks);
    }
}
