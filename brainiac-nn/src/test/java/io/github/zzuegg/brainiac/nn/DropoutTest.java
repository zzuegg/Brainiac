package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DropoutTest {

    @Test
    void passesThroughAtInference() {
        Dropout d = new Dropout(0.5f, new SplittableRandom(1));
        Var x = Var.of(Tensor.ones(Shape.of(4, 4)));
        assertArrayEquals(x.value().toArray(), d.forward(x).value().toArray());
        assertArrayEquals(x.value().toArray(), d.forward(x, false).value().toArray());
    }

    @Test
    void dropsAboutRateAndScalesTheRestWhileTraining() {
        Dropout d = new Dropout(0.25f, new SplittableRandom(7));
        float[] out = d.forward(Var.of(Tensor.ones(Shape.of(100, 100))), true).value().toArray();
        int zeros = 0;
        for (float v : out) {
            if (v == 0) {
                zeros++;
            } else {
                assertEquals(1f / 0.75f, v, 1e-6f);
            }
        }
        assertTrue(Math.abs(zeros / 10_000.0 - 0.25) < 0.02, "dropped share " + zeros / 10_000.0);
    }

    @Test
    void gradientFlowsOnlyThroughKeptValues() {
        Dropout d = new Dropout(0.5f, new SplittableRandom(3));
        Var x = Var.param(Tensor.ones(Shape.of(8)));
        Var y = d.forward(x, true);
        y.sum().backward();
        assertArrayEquals(y.value().toArray(), x.grad().toArray());
    }

    @Test
    void modelBuilderAddsDropoutWithoutParameters() {
        Model m = Model.sequential().input(4).dense(8, Activation.RELU).dropout(0.3f).dense(2).seed(1).build();
        assertEquals(3, m.layers().size());
        assertEquals(4 * 8 + 8 + 8 * 2 + 2, m.parameterCount());
        assertArrayEquals(m.predict(1, 2, 3, 4), m.predict(1, 2, 3, 4));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(1f, new SplittableRandom()));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(-0.1f, new SplittableRandom()));
    }
}
