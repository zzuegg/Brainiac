package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/** How a layer's weights are filled before training. */
@FunctionalInterface
public interface Init {

    /**
     * Glorot/Xavier uniform: {@code U(-a, a)} with {@code a = sqrt(6 / (fanIn + fanOut))}.
     * Good for tanh and sigmoid.
     */
    Init XAVIER = (shape, fanIn, fanOut, random) -> {
        float a = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        return Tensor.uniform(shape, -a, a, random);
    };

    /** He/Kaiming normal: {@code N(0, sqrt(2 / fanIn))}. Good for ReLU. */
    Init HE = (shape, fanIn, fanOut, random) ->
            Tensor.normal(shape, 0, (float) Math.sqrt(2.0 / fanIn), random);

    /** All zeros. */
    Init ZEROS = (shape, fanIn, fanOut, random) -> Tensor.zeros(shape);

    /**
     * Creates initial values.
     *
     * @param shape the shape to fill
     * @param fanIn the number of inputs per unit
     * @param fanOut the number of units
     * @param random the random source
     * @return the values, on the default device
     */
    Tensor create(Shape shape, int fanIn, int fanOut, RandomGenerator random);
}
