package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class ConditioningTest {

    @Test
    void timestepEmbeddingIsCosThenSin() {
        Tensor e = Functional.timestepEmbedding(Tensor.of(new float[] {0, 3}), 4, 10000);
        assertEquals(Shape.of(2, 4), e.shape());
        float[] v = e.toArray();
        assertArrayEquals(new float[] {1, 1, 0, 0}, java.util.Arrays.copyOf(v, 4), 1e-6f);
        assertEquals(Math.cos(3), v[4], 1e-6);
        assertEquals(Math.cos(3 * 0.01), v[5], 1e-6);   // frequency 10000^(-1/2)
        assertEquals(Math.sin(3), v[6], 1e-6);
    }

    @Test
    void filmAdaLnAndGate() {
        Var x = Var.of(Tensor.of(Shape.of(2, 3, 2), 1, 3, 2, 4, 0, 2, 1, 1, 5, 7, 2, 2));
        Var scale = Var.of(Tensor.of(Shape.of(2, 2), 0, 1, -1, 0));
        Var shift = Var.of(Tensor.of(Shape.of(2, 2), 0, 0, 10, 0));
        float[] f = Functional.film(x, scale, shift).value().toArray();
        assertArrayEquals(new float[] {1, 6, 2, 8, 0, 4, 10, 1, 10, 7, 10, 2}, f);
        // each row of two normalizes to (-1, 1), or (0, 0) for equal values
        float[] n = Functional.adaLayerNorm(x, Var.of(Tensor.zeros(Shape.of(2, 2))), Var.of(Tensor.zeros(Shape.of(2, 2))),
                1e-6f).value().toArray();
        assertArrayEquals(new float[] {-1, 1, -1, 1, -1, 1, 0, 0, -1, 1, 0, 0}, n, 1e-3f);
        Var zero = Var.of(Tensor.zeros(Shape.of(2, 2)));
        assertArrayEquals(x.value().toArray(), Functional.gate(x, x.mul(5), zero).value().toArray());
    }
}
