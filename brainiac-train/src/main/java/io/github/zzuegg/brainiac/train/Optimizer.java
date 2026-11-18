package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Updates parameters from their gradients. Optimizers keep per-parameter state (such as
 * momentum), so use one optimizer per model. All updates are tensor operations and run on
 * the parameters' device.
 */
public interface Optimizer {

    /**
     * Applies one update to every parameter that has a gradient, then clears the
     * gradients.
     *
     * @param parameters the parameters to update
     */
    void step(List<Var> parameters);

    /**
     * As {@link #step(List)}, for optimizers that need the loss and gradients again at other
     * weights within one step, such as SAM or zeroth-order methods. {@link TrainingLoop} passes it when it does not
     * accumulate micro-batches. The default ignores it.
     *
     * @param parameters the parameters to update
     * @param reevaluate computes the loss and gradients again
     */
    default void step(List<Var> parameters, Reevaluation reevaluate) {
        step(parameters);
    }

    /** Computes a batch's loss and gradients again, for {@link #step(List, Reevaluation)}. */
    @FunctionalInterface
    interface Reevaluation {

        /**
         * Clears the gradients, computes the loss at the parameters' current values and
         * back-propagates it.
         *
         * @return the loss
         */
        float lossAndGradients();
    }

    /**
     * Resets the optimizer's state for chosen entries of a parameter, as if they had never
     * been updated, for example when units are re-initialized. Optimizers without state
     * ignore it.
     *
     * @param parameter the parameter
     * @param keep 1 for entries to keep and 0 for entries to reset, broadcast to the parameter
     */
    default void resetState(Var parameter, Tensor keep) {
    }

    /**
     * Drops all state kept for a parameter that is no longer trained, such as one a
     * growing layer replaced. A parameter seen again later starts from fresh state. The
     * trainer calls it when the model's parameters change between steps; optimizers
     * without state ignore it.
     *
     * @param parameter the parameter
     */
    default void forget(Var parameter) {
    }

    /**
     * Returns the current learning rate.
     *
     * @return the learning rate
     */
    float learningRate();

    /**
     * Changes the learning rate, for example from a {@link LearningRateSchedule}.
     *
     * @param learningRate the new rate, positive
     */
    void learningRate(float learningRate);

    /**
     * One whole training step: clears the parameters' gradients, computes the loss,
     * back-propagates it and updates the parameters (which clears the gradients again). The
     * loop {@code for (...) optimizer.minimize(params, () -> loss(model, batch));} cannot
     * accumulate stale gradients by mistake.
     *
     * @param parameters the parameters to update
     * @param loss computes the loss at the parameters' current values
     * @return the loss before the update
     */
    default Var minimize(List<Var> parameters, java.util.function.Supplier<Var> loss) {
        for (Var p : parameters) {
            p.zeroGrad();
        }
        Var l = loss.get();
        l.backward();
        step(parameters, () -> {
            for (Var p : parameters) {
                p.zeroGrad();
            }
            Var again = loss.get();
            again.backward();
            return again.value().item();
        });
        return l;
    }

    /**
     * Plain stochastic gradient descent: {@code w -= learningRate * grad}.
     *
     * @param learningRate the step size, for example {@code 0.1f}
     * @return a new optimizer
     */
    static Optimizer sgd(float learningRate) {
        return new Momentum(learningRate, 0f, false);
    }

    /**
     * Gradient descent with momentum: {@code v = momentum * v + grad; w -= learningRate * v}.
     *
     * @param learningRate the step size, for example {@code 0.05f}
     * @param momentum how much of the previous update is kept, usually {@code 0.9f}
     * @return a new optimizer
     */
    static Optimizer momentum(float learningRate, float momentum) {
        return new Momentum(learningRate, momentum, false);
    }

    /**
     * Gradient descent with Nesterov momentum, as PyTorch's SGD with {@code nesterov=True}:
     * {@code v = momentum * v + grad; w -= learningRate * (grad + momentum * v)}.
     *
     * @param learningRate the step size
     * @param momentum how much of the previous update is kept, usually {@code 0.9f}
     * @return a new optimizer
     */
    static Optimizer nesterov(float learningRate, float momentum) {
        return new Momentum(learningRate, momentum, true);
    }

    /**
     * Adam with the usual settings ({@code beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8}).
     * A good default.
     *
     * @param learningRate the step size, for example {@code 0.001f}
     * @return a new optimizer
     */
    static Optimizer adam(float learningRate) {
        return adam(learningRate, 0.9f, 0.999f, 1e-8f);
    }

    /**
     * Adam: adaptive per-parameter step sizes from running averages of the gradient and
     * its square, with bias correction.
     *
     * @param learningRate the step size
     * @param beta1 decay of the gradient average
     * @param beta2 decay of the squared-gradient average
     * @param epsilon added to the denominator for stability
     * @return a new optimizer
     */
    static Optimizer adam(float learningRate, float beta1, float beta2, float epsilon) {
        return new Adam(learningRate, beta1, beta2, epsilon);
    }

    /**
     * The optimizer's state for these parameters (such as Adam's averages), for a
     * checkpoint. Keys are {@code "<index>.<name>"} with the parameter's index in the list,
     * plus {@code "step"}.
     *
     * @param parameters the parameters, in the order {@link #step} gets them
     * @return the state tensors
     */
    default Map<String, Tensor> state(List<Var> parameters) {
        return Map.of();
    }

    /**
     * Restores state that {@link #state} returned, for the same parameters in the same order.
     *
     * @param parameters the parameters
     * @param state the saved state
     */
    default void loadState(List<Var> parameters, Map<String, Tensor> state) {
        if (!state.isEmpty()) {
            throw new UnsupportedOperationException(this + " keeps no state");
        }
    }

    /**
     * Runs an optimizer on the CPU (ZeRO-offload): its state and a master copy of the
     * weights live in host memory, and each step moves the gradients there and the new
     * weights back. Frees GPU memory for larger models at the cost of two transfers per
     * step. The master copy also keeps float32 weights for models trained in float16.
     *
     * @param optimizer the optimizer to run on the host, a new one
     * @return the offloaded optimizer
     */
    static Optimizer offloaded(Optimizer optimizer) {
        return new Offloaded(optimizer, io.github.zzuegg.brainiac.tensor.Device.cpu());
    }

    /**
     * Adam with 8-bit state (Dettmers et al. 2022): the same steps as {@link #adam(float)}
     * from averages kept as 8-bit levels with a scale per block of 256, a quarter of the
     * memory. Small parameters (under 4096 values) keep float32 state.
     *
     * @param learningRate the step size
     * @return a new optimizer
     */
    static Optimizer adam8bit(float learningRate) {
        return adam8bit(learningRate, 0.9f, 0.999f, 1e-8f, 0);
    }

    /**
     * Adam, or AdamW with a weight decay, with 8-bit state and every setting.
     *
     * @param learningRate the step size
     * @param beta1 decay of the gradient average
     * @param beta2 decay of the squared-gradient average
     * @param epsilon added to the denominator
     * @param weightDecay the decoupled weight decay, 0 for none
     * @return a new optimizer
     */
    static Optimizer adam8bit(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        return new Adam8bit(learningRate, beta1, beta2, epsilon, weightDecay);
    }

    /**
     * AdamW (Loshchilov and Hutter 2019): Adam with weight decay applied to the weights
     * directly ({@code w -= lr * weightDecay * w}), not through the gradient.
     *
     * @param learningRate the step size, for example {@code 3e-4f}
     * @param weightDecay the decay, for example {@code 0.01f}
     * @return a new optimizer
     */
    static Optimizer adamW(float learningRate, float weightDecay) {
        return adamW(learningRate, 0.9f, 0.999f, 1e-8f, weightDecay);
    }

    /**
     * AdamW with every setting.
     *
     * @param learningRate the step size
     * @param beta1 decay of the gradient average
     * @param beta2 decay of the squared-gradient average
     * @param epsilon added to the denominator
     * @param weightDecay the decoupled weight decay
     * @return a new optimizer
     */
    static Optimizer adamW(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        return new Adam(learningRate, beta1, beta2, epsilon, weightDecay);
    }

    /**
     * Lion (Chen et al. 2023): steps of equal size in the direction of the sign of a
     * momentum, with half Adam's state. Use a learning rate 3-10 times smaller than AdamW's
     * and a weight decay 3-10 times larger.
     *
     * @param learningRate the step size, for example {@code 1e-4f}
     * @param weightDecay the decoupled weight decay
     * @return a new optimizer ({@code beta1 = 0.9, beta2 = 0.99})
     */
    static Optimizer lion(float learningRate, float weightDecay) {
        return lion(learningRate, 0.9f, 0.99f, weightDecay);
    }

    /**
     * Lion with every setting.
     *
     * @param learningRate the step size
     * @param beta1 how much momentum the step direction uses
     * @param beta2 how much momentum is kept
     * @param weightDecay the decoupled weight decay
     * @return a new optimizer
     */
    static Optimizer lion(float learningRate, float beta1, float beta2, float weightDecay) {
        return new Lion(learningRate, beta1, beta2, weightDecay);
    }

    /**
     * Adafactor with a fixed learning rate: Adam-like steps with a factored second moment
     * (a row and a column per matrix instead of a full copy), clipped by their size.
     *
     * @param learningRate the step size, for example {@code 1e-3f}
     * @return a new optimizer
     */
    static Optimizer adafactor(float learningRate) {
        return new Adafactor(learningRate, false, false, 0);
    }

    /**
     * Adafactor as the T5 paper trains: no learning rate, a step of {@code min(0.01,
     * 1/sqrt(t))} relative to each parameter's size. {@link #learningRate(float)} has no
     * effect.
     *
     * @return a new optimizer
     */
    static Optimizer adafactor() {
        return new Adafactor(1, true, true, 0);
    }

    /**
     * Adafactor with every setting.
     *
     * @param learningRate the step size, unless {@code relativeStep}
     * @param relativeStep whether to step {@code min(0.01, 1/sqrt(t))}
     * @param scaleParameter whether to scale the step by each parameter's root mean square
     * @param weightDecay the weight decay
     * @return a new optimizer
     */
    static Optimizer adafactor(float learningRate, boolean relativeStep, boolean scaleParameter,
            float weightDecay) {
        return new Adafactor(learningRate, relativeStep, scaleParameter, weightDecay);
    }

    /**
     * Muon (Jordan et al. 2024) with PyTorch's defaults (weight decay 0.1, momentum 0.95,
     * Nesterov, five Newton-Schulz steps): for the matrices of hidden layers. Combine it with
     * AdamW for the other parameters:
     *
     * <pre>{@code
     * Optimizer opt = Optimizer.groups(Optimizer.adamW(3e-4f, 0.01f),
     *         Optimizer.Group.matrices(Optimizer.muon(0.02f)));
     * }</pre>
     *
     * @param learningRate the step size, for example {@code 0.02f}
     * @return a new optimizer
     */
    static Optimizer muon(float learningRate) {
        return muon(learningRate, 0.1f, 0.95f, false);
    }

    /**
     * Muon with its main settings.
     *
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @param momentum the momentum
     * @param matchAdamW whether to scale steps to AdamW's typical size ({@code 0.2 *
     *     sqrt(max(rows, cols))}), so AdamW's learning rate can be reused; otherwise {@code
     *     sqrt(max(1, rows / cols))}
     * @return a new optimizer
     */
    static Optimizer muon(float learningRate, float weightDecay, float momentum, boolean matchAdamW) {
        return new Muon(learningRate, weightDecay, momentum, true, matchAdamW, 5);
    }

    /**
     * Muon with an orthogonalizer of one's own in place of five Newton-Schulz steps: such as
     * an exact polar factor from {@link io.github.zzuegg.brainiac.tensor.Linalg#svd} ({@code U V^T}),
     * other polynomial coefficients, or a cheaper approximation for large matrices.
     *
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @param momentum the momentum
     * @param matchAdamW as for {@link #muon(float, float, float, boolean)}
     * @param orthogonalizer maps each momentum matrix to its update direction, of the same shape
     * @return a new optimizer
     */
    static Optimizer muon(float learningRate, float weightDecay, float momentum, boolean matchAdamW,
            java.util.function.UnaryOperator<io.github.zzuegg.brainiac.tensor.Tensor> orthogonalizer) {
        return new Muon(learningRate, weightDecay, momentum, true, matchAdamW, orthogonalizer);
    }

    /**
     * Different optimizers (or settings) for different parameters: each parameter goes to
     * the first group that takes it, the rest to {@code rest}. Setting the learning rate
     * scales every group's rate by the same factor.
     *
     * @param rest the optimizer for parameters no group takes
     * @param groups the groups
     * @return a new optimizer
     */
    static Optimizer groups(Optimizer rest, Group... groups) {
        return new ParamGroups(rest, List.of(groups));
    }

    /**
     * Some parameters and the optimizer for them.
     *
     * @param takes which parameters belong to the group
     * @param optimizer their optimizer
     */
    record Group(Predicate<Var> takes, Optimizer optimizer) {

        /**
         * @param parameters the parameters
         * @param optimizer their optimizer
         * @return a group of exactly these parameters
         */
        public static Group of(Collection<Var> parameters, Optimizer optimizer) {
            Set<Var> set = Collections.newSetFromMap(new IdentityHashMap<>());
            set.addAll(parameters);
            return new Group(set::contains, optimizer);
        }

        /**
         * @param optimizer the optimizer
         * @return a group of every parameter with two dimensions
         */
        public static Group matrices(Optimizer optimizer) {
            return new Group(p -> p.shape().rank() == 2, optimizer);
        }

        /**
         * @param optimizer the optimizer
         * @return a group of every parameter with fewer than two dimensions (biases, norms),
         *     for example to exclude them from weight decay
         */
        public static Group vectors(Optimizer optimizer) {
            return new Group(p -> p.shape().rank() < 2, optimizer);
        }
    }
}
