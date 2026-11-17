package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AnalogDenseTest {
    @Test
    void adcQuantizesWithStraightThroughGradient() {
        AnalogDense a = AnalogDense.of(3, 2, 10, 0, 3, 1, 1);   // 3 bits: levels -1, -2/3, ... 1
        a.parameters().get(0).assign(Tensor.of(Shape.of(3, 2), 0.1f, -0.5f, 0.2f, 0.3f, 0.05f, 2f));
        Var x = Var.param(Tensor.of(Shape.of(1, 3), 1, 1, 1));
        Var y = a.forward(x);
        float[] v = y.value().toArray();
        assertEquals(1f / 3, v[0], 1e-6);   // 0.35 rounds to the 1/3 level
        assertEquals(1f, v[1], 1e-6);       // 1.8 clamps to full scale
        y.sum().backward();
        // the clamped output passes no gradient, the in-range one passes it unchanged
        assertArrayEquals(new float[] {0.1f, 0.2f, 0.05f}, x.grad().toArray(), 1e-6f);
    }

    @Test
    void programmingFixesOneNoisyChip() {
        AnalogDense a = AnalogDense.of(4, 3, 0.5f, 0.1f, 0, 1, 2);
        Var x = Var.of(Tensor.normal(Shape.of(2, 4), 0, 1, new SplittableRandom(3)));
        float[] ideal = a.forward(x).value().toArray();
        a.program(0.2f, new SplittableRandom(4));
        float[] chip = a.forward(x).value().toArray();
        assertArrayEquals(chip, a.forward(x).value().toArray(), 0);   // the chip is fixed once programmed
        assertFalse(java.util.Arrays.equals(ideal, chip));
        a.unprogram();
        assertArrayEquals(ideal, a.forward(x).value().toArray(), 0);
        // training passes draw fresh noise
        assertFalse(java.util.Arrays.equals(a.forward(x, true).value().toArray(), a.forward(x, true).value().toArray()));
        assertTrue(a.parameters().size() == 2);
    }
}
