package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** A model nests inside another as a layer. */
class ModelAsLayerTest {

    @Test
    void aModelIsALayer() {
        Model inner = Model.sequential().input(3).dense(5, Activation.TANH).dense(2).seed(1).build();
        Model outer = Model.sequential().input(3).layer(inner).dense(1).seed(2).build();
        assertEquals(Shape.of(2), ((Layer) inner).outputShape(Shape.of(3)));
        assertTrue(outer.parameters().containsAll(inner.parameters()));
        Tensor x = Tensor.uniform(Shape.of(4, 3), -1, 1, new SplittableRandom(3));
        Dense last = (Dense) outer.layers().getLast();
        float[] expected = last.forward(inner.forward(Var.of(x))).value().toArray();
        assertArrayEquals(expected, outer.forward(Var.of(x)).value().toArray(), 1e-6f);
    }
}
