package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;

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
     * Plain stochastic gradient descent: {@code w -= learningRate * grad}.
     *
     * @param learningRate the step size, for example {@code 0.1f}
     * @return a new optimizer
     */
    static Optimizer sgd(float learningRate) {
        return new Momentum(learningRate, 0f);
    }

    /**
     * Gradient descent with momentum: {@code v = momentum * v + grad; w -= learningRate * v}.
     *
     * @param learningRate the step size, for example {@code 0.05f}
     * @param momentum how much of the previous update is kept, usually {@code 0.9f}
     * @return a new optimizer
     */
    static Optimizer momentum(float learningRate, float momentum) {
        return new Momentum(learningRate, momentum);
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
}
