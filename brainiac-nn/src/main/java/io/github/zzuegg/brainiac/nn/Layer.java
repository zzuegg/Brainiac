package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;

/**
 * One step of a network: turns a batch of inputs {@code [batch, inputSize]} into a batch
 * of outputs {@code [batch, outputSize(inputSize)]}.
 *
 * <p>Implement this interface to add your own layer type; express {@link #forward(Var)}
 * with {@link Var} operations and gradients come for free.
 */
public interface Layer {

    /**
     * Computes the layer's output.
     *
     * @param input a batch of inputs, shape {@code [batch, inputSize]}
     * @return a batch of outputs
     */
    Var forward(Var input);

    /**
     * Returns the trainable parameters, in a stable order.
     *
     * @return the parameters; empty for layers without any
     */
    default List<Var> parameters() {
        return List.of();
    }

    /**
     * Returns how many features each output row has, given the input width.
     *
     * @param inputSize the number of input features
     * @return the number of output features
     */
    default int outputSize(int inputSize) {
        return inputSize;
    }
}
