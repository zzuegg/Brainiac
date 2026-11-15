package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A description of a layer that is created once its input shape is known, such as
 * {@code Conv.of(32, 3, 3).stride(2)}. Model builders and {@link Node}s accept specs
 * through {@code add(spec)}, so every option of every built-in layer is available in both.
 *
 * <p>Shapes here are per sample, without the batch dimension.
 */
public interface LayerSpec {

    /**
     * Returns the shape this layer produces for an input shape.
     *
     * @param input the shape of one input sample
     * @return the shape of one output sample
     * @throws IllegalArgumentException if the layer cannot take that input
     */
    Shape outputShape(Shape input);

    /**
     * Creates the layer with fresh weights.
     *
     * @param input the shape of one input sample
     * @param random the source of random initial weights
     * @return the layer
     */
    Layer create(Shape input, SplittableRandom random);

    /**
     * Wraps an existing layer instance, which keeps (and shares) its weights.
     *
     * @param layer the layer
     * @return the spec
     */
    static LayerSpec of(Layer layer) {
        Objects.requireNonNull(layer, "layer");
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return layer.outputShape(input);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return layer;
            }
        };
    }
}
