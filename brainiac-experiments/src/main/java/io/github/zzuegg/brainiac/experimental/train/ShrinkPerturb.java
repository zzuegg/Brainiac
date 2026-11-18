package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.List;
import java.util.function.Supplier;

/**
 * Shrink and perturb (Ash and Adams 2020, <a href="https://arxiv.org/abs/1910.08475">arXiv:1910.08475</a>):
 * warm-starting from a model trained on part of the data generalizes worse than training from
 * scratch on all of it, even after convergence. Before training on new data, multiply every
 * weight by {@code lambda < 1} and add {@code p} times a fresh initialization,
 * {@code theta <- lambda theta + p theta_fresh}. Shrinking evens out the gradient contributions
 * of old and new examples; the noise restores plasticity. With {@code lambda = 0, p = 1} it is
 * training from scratch, with {@code lambda = 1, p = 0} a plain warm start.
 *
 * <p>Call {@link #apply} whenever new data arrives, or attach {@link #everyEpochs} to a core
 * trainer as a {@link TrainingListener}.
 */
public final class ShrinkPerturb {

    private final float shrink;
    private final float perturb;

    /**
     * @param shrink the factor {@code lambda} on the current weights, for example 0.4
     * @param perturb the factor {@code p} on the fresh initialization, for example 0.1
     */
    public ShrinkPerturb(float shrink, float perturb) {
        this.shrink = shrink;
        this.perturb = perturb;
    }

    /**
     * Shrinks the parameters and adds the scaled fresh initialization.
     *
     * @param parameters the parameters
     * @param fresh a fresh initialization of the same parameters, same order and shapes
     */
    public void apply(List<Var> parameters, List<Tensor> fresh) {
        for (int i = 0; i < parameters.size(); i++) {
            Var p = parameters.get(i);
            p.assign(p.value().mul(shrink).add(fresh.get(i).mul(perturb)));
        }
    }

    /**
     * A listener that shrinks and perturbs every {@code k} epochs, for training loops where new
     * data arrives on an epoch schedule.
     *
     * @param parameters the parameters
     * @param k the period in epochs
     * @param fresh draws a fresh initialization each time
     * @return the listener
     */
    public TrainingListener everyEpochs(List<Var> parameters, int k, Supplier<List<Tensor>> fresh) {
        return (EpochResult result) -> {
            if ((result.epoch() + 1) % k == 0) {
                apply(parameters, fresh.get());
            }
        };
    }

    @Override
    public String toString() {
        return "ShrinkPerturb(lambda=" + shrink + ", p=" + perturb + ")";
    }
}
