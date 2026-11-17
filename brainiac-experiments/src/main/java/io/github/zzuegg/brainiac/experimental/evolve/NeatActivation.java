package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.function.DoubleUnaryOperator;

/**
 * Activation functions of NEAT nodes and of HyperNEAT's compositional pattern producing
 * networks (CPPNs), where a mix of periodic, symmetric and monotone functions lets a small
 * network paint regular weight patterns: symmetry ({@link #GAUSS}, {@link #ABS}),
 * repetition ({@link #SIN}) and gradients ({@link #SIGMOID}, {@link #LINEAR}).
 */
public enum NeatActivation implements DoubleUnaryOperator {
    /** The steepened logistic of the original NEAT paper, {@code 1 / (1 + e^{-4.9x})}. */
    SIGMOID(x -> 1 / (1 + Math.exp(-4.9 * x))),
    /** Hyperbolic tangent. */
    TANH(Math::tanh),
    /** Sine, for repeating patterns. */
    SIN(Math::sin),
    /** A Gaussian bump {@code e^{-x^2}}, for symmetry and locality. */
    GAUSS(x -> Math.exp(-x * x)),
    /** Absolute value, for mirror symmetry. */
    ABS(Math::abs),
    /** Identity. */
    LINEAR(x -> x),
    /** Rectified linear unit. */
    RELU(x -> Math.max(0, x));

    private final DoubleUnaryOperator f;

    NeatActivation(DoubleUnaryOperator f) {
        this.f = f;
    }

    @Override
    public double applyAsDouble(double x) {
        return f.applyAsDouble(x);
    }
}
