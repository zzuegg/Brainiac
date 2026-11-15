package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Batch normalization for inputs of shape {@code [batch, features]}.
 *
 * <p>While training, each feature is normalized with the mean and variance of the current
 * batch, and running averages of both are kept. For predictions the running averages are
 * used, so a single sample gives a stable answer. A learned scale ({@code gamma}) and
 * shift ({@code beta}) per feature follow the normalization.
 */
public final class BatchNorm implements Layer {

    private final int features;
    private final float momentum;
    private final float epsilon;
    private final Var gamma;
    private final Var beta;
    private final Var runningMean;
    private final Var runningVariance;

    /**
     * Creates batch normalization with momentum {@code 0.1} and epsilon {@code 1e-5}.
     *
     * @param features the number of features
     */
    public BatchNorm(int features) {
        this(features, 0.1f, 1e-5f);
    }

    /**
     * Creates batch normalization.
     *
     * @param features the number of features
     * @param momentum how much each training batch moves the running averages, in (0, 1]
     * @param epsilon added to the variance to avoid dividing by zero, positive
     */
    public BatchNorm(int features, float momentum, float epsilon) {
        if (features < 1) {
            throw new IllegalArgumentException("features must be at least 1, got " + features);
        }
        if (!(momentum > 0 && momentum <= 1)) {
            throw new IllegalArgumentException("momentum must be in (0, 1], got " + momentum);
        }
        if (!(epsilon > 0)) {
            throw new IllegalArgumentException("epsilon must be positive, got " + epsilon);
        }
        this.features = features;
        this.momentum = momentum;
        this.epsilon = epsilon;
        this.gamma = Var.param(Tensor.ones(Shape.of(features)));
        this.beta = Var.param(Tensor.zeros(Shape.of(features)));
        this.runningMean = Var.of(Tensor.zeros(Shape.of(features)));
        this.runningVariance = Var.of(Tensor.ones(Shape.of(features)));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape shape = input.shape();
        if (shape.rank() != 2 || shape.dim(1) != features) {
            throw new IllegalArgumentException("BatchNorm(" + features + ") expects [batch, " + features + "], got " + shape);
        }
        Var normalized;
        if (training) {
            long n = shape.dim(0);
            if (n < 2) {
                throw new IllegalArgumentException("batch normalization needs at least 2 samples per training batch");
            }
            Var mean = input.mean(0, false);
            Var centered = input.sub(mean);
            Var variance = centered.square().mean(0, false);
            normalized = centered.div(variance.add(epsilon).sqrt());
            Tensor unbiased = variance.value().mul(n / (float) (n - 1));
            runningMean.assign(runningMean.value().mul(1 - momentum).add(mean.value().mul(momentum)));
            runningVariance.assign(runningVariance.value().mul(1 - momentum).add(unbiased.mul(momentum)));
        } else {
            Tensor scale = runningVariance.value().add(epsilon).sqrt();
            normalized = input.sub(runningMean.value()).div(scale);
        }
        return normalized.mul(gamma).add(beta);
    }

    @Override
    public List<Var> parameters() {
        return List.of(gamma, beta);
    }

    @Override
    public List<Var> buffers() {
        return List.of(runningMean, runningVariance);
    }

    /** @return the running mean per feature, used for predictions */
    public Tensor runningMean() {
        return runningMean.value();
    }

    /** @return the running variance per feature, used for predictions */
    public Tensor runningVariance() {
        return runningVariance.value();
    }

    @Override
    public String toString() {
        return "BatchNorm(" + features + ")";
    }
}
