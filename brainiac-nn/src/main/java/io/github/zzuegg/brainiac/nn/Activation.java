package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.function.UnaryOperator;

/**
 * An element-wise (or row-wise, for softmax) non-linearity. Activations are layers
 * without parameters, so they can also be added to a model on their own.
 */
public interface Activation extends Layer {

    /** Passes values through unchanged. */
    Activation IDENTITY = named("identity", x -> x);

    /** {@code max(0, x)}: the usual choice for hidden layers. */
    Activation RELU = named("relu", Var::relu);

    /** {@code 1 / (1 + e^-x)}: squashes to (0, 1); for yes/no outputs. */
    Activation SIGMOID = named("sigmoid", Var::sigmoid);

    /** Hyperbolic tangent: squashes to (-1, 1). */
    Activation TANH = named("tanh", Var::tanh);

    /**
     * Turns each row into probabilities that sum to 1; for picking one of several
     * classes. Computed stably by subtracting the row maximum first.
     */
    Activation SOFTMAX = named("softmax", x -> {
        Var shifted = x.sub(x.max(-1, true).detach());
        Var e = shifted.exp();
        return e.div(e.sum(-1, true));
    });

    /**
     * Creates an activation from a function.
     *
     * @param name the name shown in model summaries
     * @param f the function
     * @return the activation
     */
    static Activation named(String name, UnaryOperator<Var> f) {
        return new Activation() {
            @Override
            public Var forward(Var input) {
                return f.apply(input);
            }

            @Override
            public String toString() {
                return name;
            }
        };
    }
}
