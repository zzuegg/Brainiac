package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.List;

/**
 * A layer that combines several inputs, such as cross-attention reading queries from one
 * node and keys and values from another. Wire it with {@link Node#merge}.
 */
public interface MultiInputLayer extends Layer {

    /**
     * Runs the layer.
     *
     * @param inputs one batch per input, in the order the node lists them
     * @param training whether the model is training
     * @return the output
     */
    Var forward(List<Var> inputs, boolean training);

    /**
     * Runs the layer on a single input, as if every input were the same.
     *
     * @param input the input
     * @param training whether the model is training
     * @return the output
     */
    @Override
    default Var forward(Var input, boolean training) {
        return forward(List.of(input), training);
    }

    @Override
    default Var forward(Var input) {
        return forward(input, false);
    }

    /**
     * Returns the output shape for the given input shapes, per sample.
     *
     * @param inputs the shape of one sample of each input
     * @return the shape of one output sample
     */
    Shape outputShape(List<Shape> inputs);

    /** A spec for a multi-input layer, created once the input shapes are known. */
    interface Spec {

        /**
         * Returns the output shape for the given input shapes.
         *
         * @param inputs the shape of one sample of each input
         * @return the output shape
         */
        Shape outputShape(List<Shape> inputs);

        /**
         * Creates the layer.
         *
         * @param inputs the shape of one sample of each input
         * @param random the source of initial weights
         * @return the layer
         */
        MultiInputLayer create(List<Shape> inputs, java.util.SplittableRandom random);
    }
}
