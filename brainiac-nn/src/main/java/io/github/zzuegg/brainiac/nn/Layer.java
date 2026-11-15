package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.List;

/**
 * One step of a network: turns a batch of inputs {@code [batch, inputSize]} into a batch
 * of outputs {@code [batch, outputSize(inputSize)]}.
 *
 * <p>Implement this interface to add your own layer type; express {@link #forward(Var)}
 * with {@link Var} operations and gradients come for free. To load saved models that use
 * it, {@linkplain #register(String, Class, LayerCodec) register} the class first.
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
     * Computes the layer's output in training or inference mode. Layers that behave
     * differently while training, such as {@link Dropout}, override this; the default
     * ignores the mode.
     *
     * @param input a batch of inputs, shape {@code [batch, inputSize]}
     * @param training {@code true} while training, {@code false} for predictions
     * @return a batch of outputs
     */
    default Var forward(Var input, boolean training) {
        return forward(input);
    }

    /**
     * Returns the trainable parameters, in a stable order.
     *
     * @return the parameters; empty for layers without any
     */
    default List<Var> parameters() {
        return List.of();
    }

    /**
     * Returns state that is not trained by gradients but belongs to the model, such as
     * the running statistics of {@link BatchNorm}. It moves with the model between
     * devices and is saved with it.
     *
     * @return the buffers, in a stable order; empty for layers without any
     */
    default List<Var> buffers() {
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

    /**
     * Returns the shape of one output sample for the shape of one input sample. The
     * default treats the last dimension as the features: it keeps the other dimensions and
     * maps the last one with {@link #outputSize(int)}.
     *
     * @param inputShape the shape of one input sample, without the batch dimension
     * @return the shape of one output sample
     */
    default Shape outputShape(Shape inputShape) {
        int[] dims = inputShape.toArray();
        dims[dims.length - 1] = outputSize(dims[dims.length - 1]);
        return Shape.of(dims);
    }

    /**
     * Makes a custom layer class loadable from model files. Built-in layers are
     * registered already; activations use {@link Activation#register(Activation)}.
     *
     * @param name the type name stored in files
     * @param type the layer class
     * @param codec turns the layer's settings into a {@link Config} and back
     * @param <L> the layer class
     */
    static <L extends Layer> void register(String name, Class<L> type, LayerCodec<L> codec) {
        Registry.layer(name, type, codec);
    }
}
