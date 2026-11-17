package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Elastic Weight Consolidation (Kirkpatrick et al. 2017, <a href="https://arxiv.org/abs/1612.00796">arXiv:1612.00796</a>):
 * after a task, estimate how important every weight was, the diagonal of the Fisher
 * information {@code F = E[g^2]}, and from then on penalise moving important weights away
 * from their old values: {@code lambda/2 sum F (w - w*)^2}. Consolidating again after the
 * next task adds its Fisher to the old one (online EWC with no decay), anchoring at the
 * newest weights.
 *
 * <p>Add the penalty to a loss by hand ({@link #penalty}) or wrap the loss given to a core
 * {@code Trainer} ({@link #wrap}).
 */
public final class ElasticWeightConsolidation {

    private final float lambda;
    private final List<Var> parameters;
    private final Map<Var, Tensor> fisher = new IdentityHashMap<>();
    private final Map<Var, Tensor> anchor = new IdentityHashMap<>();

    /**
     * @param parameters the parameters to protect
     * @param lambda the penalty strength, for example 100
     */
    public ElasticWeightConsolidation(List<Var> parameters, float lambda) {
        this.parameters = List.copyOf(parameters);
        this.lambda = lambda;
    }

    /**
     * Estimates the Fisher diagonal from the squared gradients of some small batches (one
     * sample each gives the exact empirical Fisher) and anchors the current weights.
     *
     * @param batchLosses each computes one batch's loss (not yet differentiated)
     */
    public void consolidate(List<Supplier<Var>> batchLosses) {
        Map<Var, Tensor> sum = new IdentityHashMap<>();
        for (Supplier<Var> loss : batchLosses) {
            for (Var p : parameters) {
                p.zeroGrad();
            }
            loss.get().backward();
            for (Var p : parameters) {
                if (p.grad() != null) {
                    Tensor g2 = p.grad().square();
                    sum.merge(p, g2, Tensor::add);
                    p.zeroGrad();
                }
            }
        }
        for (Var p : parameters) {
            Tensor f = sum.get(p);
            if (f != null) {
                fisher.merge(p, f.div(batchLosses.size()), Tensor::add);
            }
            anchor.put(p, p.value());
        }
    }

    /**
     * @param p a parameter
     * @return its accumulated Fisher diagonal, or {@code null} before consolidation
     */
    public Tensor fisher(Var p) {
        return fisher.get(p);
    }

    /** @return {@code lambda/2 sum F (w - w*)^2}, zero before the first consolidation */
    public Var penalty() {
        Var total = null;
        for (Var p : parameters) {
            Tensor f = fisher.get(p);
            if (f == null) {
                continue;
            }
            Var term = p.sub(anchor.get(p)).square().mul(f).sum();
            total = total == null ? term : total.add(term);
        }
        return total == null ? Var.of(Tensor.scalar(0)) : total.mul(lambda / 2);
    }

    /**
     * @param base the task loss
     * @return a loss that adds the EWC penalty, for a core {@code Trainer}
     */
    public Loss wrap(Loss base) {
        return (prediction, target) -> base.compute(prediction, target).add(penalty());
    }

    /** @return the protected parameters */
    public List<Var> parameters() {
        return new ArrayList<>(parameters);
    }
}
