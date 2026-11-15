package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class SliceConcatTest {

    private static final Tensor M = Tensor.of(Shape.of(2, 3), 1, 2, 3, 4, 5, 6);

    @Test
    void slicesAlongEitherAxis() {
        assertArrayEquals(new float[] {2, 3, 5, 6}, M.slice(1, 1, 3).toArray());
        assertEquals(Shape.of(2, 2), M.slice(-1, 1, 3).shape());
        assertArrayEquals(new float[] {4, 5, 6}, M.slice(0, 1, 2).toArray());
        assertThrows(IllegalArgumentException.class, () -> M.slice(1, 2, 2));
        assertThrows(IllegalArgumentException.class, () -> M.slice(1, 2, 4));
    }

    @Test
    void concatenatesAlongEitherAxis() {
        Tensor c = Tensor.concat(1, M, Tensor.of(Shape.of(2, 1), 7, 8));
        assertEquals(Shape.of(2, 4), c.shape());
        assertArrayEquals(new float[] {1, 2, 3, 7, 4, 5, 6, 8}, c.toArray());
        Tensor r = Tensor.concat(0, M, M.slice(0, 0, 1));
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 1, 2, 3}, r.toArray());
        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(1, M, Tensor.ones(Shape.of(3, 1))));
    }

    @Test
    void sliceAndConcatAreInverse() {
        Tensor t = Tensor.of(Shape.of(2, 3, 2), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        Tensor back = Tensor.concat(1, t.slice(1, 0, 1), t.slice(1, 1, 3));
        assertArrayEquals(t.toArray(), back.toArray());
    }
}
