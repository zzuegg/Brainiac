package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Reptile: the simplest meta-learner. For each task, train a copy of the weights for a few
 * SGD steps, then move the initialization a fraction of the way towards the trained weights:
 * {@code w <- w + epsilon * (mean_tasks(w_task) - w)}. No second derivatives and no
 * support/query split are needed, yet to leading order it optimizes the same objective as
 * MAML. Works with any model through its parameter {@link Var}s.
 *
 * <p>Paper: Nichol, Achiam, Schulman, "On First-Order Meta-Learning Algorithms" (2018),
 * <a href="https://arxiv.org/abs/1803.02999">arXiv:1803.02999</a>.
 */
public final class Reptile {

    private final List<Var> parameters;
    private final float innerRate;
    private final int innerSteps;
    private final float metaRate;

    /**
     * Creates the meta-learner.
     *
     * @param parameters the model parameters
     * @param innerRate the SGD rate per task
     * @param innerSteps SGD steps per task
     * @param metaRate the outer interpolation step {@code epsilon}, in {@code (0, 1]}
     */
    public Reptile(List<Var> parameters, float innerRate, int innerSteps, float metaRate) {
        this.parameters = List.copyOf(parameters);
        this.innerRate = innerRate;
        this.innerSteps = innerSteps;
        this.metaRate = metaRate;
    }

    /**
     * Runs one meta step.
     *
     * @param tasks one loss per task, computed with the model's current parameters
     * @return the mean task loss after the inner steps
     */
    public float metaStep(List<? extends Supplier<Var>> tasks) {
        List<Tensor> start = new ArrayList<>();
        for (Var p : parameters) {
            start.add(p.value());
        }
        Tensor[] sum = new Tensor[parameters.size()];
        double total = 0;
        for (Supplier<Var> task : tasks) {
            for (int i = 0; i < parameters.size(); i++) {
                parameters.get(i).assign(start.get(i));
            }
            total += train(task, innerSteps);
            for (int i = 0; i < parameters.size(); i++) {
                Tensor w = parameters.get(i).value();
                sum[i] = sum[i] == null ? w : sum[i].add(w);
            }
        }
        for (int i = 0; i < parameters.size(); i++) {
            Tensor mean = sum[i].div(tasks.size());
            parameters.get(i).assign(start.get(i).add(mean.sub(start.get(i)).mul(metaRate)));
        }
        return (float) (total / tasks.size());
    }

    /**
     * Trains the current parameters on one task with plain SGD, in place.
     *
     * @param task the task loss
     * @param steps the SGD steps
     * @return the loss at the last step
     */
    public float train(Supplier<Var> task, int steps) {
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            for (Var p : parameters) {
                p.zeroGrad();
            }
            Var loss = task.get();
            last = loss.value().item();
            loss.backward();
            for (Var p : parameters) {
                if (p.grad() != null) {
                    p.assign(p.value().sub(p.grad().mul(innerRate)));
                    p.zeroGrad();
                }
            }
        }
        return last;
    }
}
