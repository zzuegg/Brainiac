package io.github.zzuegg.brainiac.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ShapeTest {

    @Test
    void reportsRankDimsAndSize() {
        Shape s = Shape.of(4, 2, 3);
        assertEquals(3, s.rank());
        assertEquals(4, s.dim(0));
        assertEquals(3, s.dim(-1));
        assertEquals(24, s.size());
        assertEquals("[4, 2, 3]", s.toString());
    }

    @Test
    void scalarHasRankZeroAndSizeOne() {
        Shape s = Shape.of();
        assertEquals(0, s.rank());
        assertEquals(1, s.size());
    }

    @Test
    void comparesByValue() {
        assertEquals(Shape.of(2, 3), Shape.of(2, 3));
        assertEquals(Shape.of(2, 3).hashCode(), Shape.of(2, 3).hashCode());
        assertNotEquals(Shape.of(2, 3), Shape.of(3, 2));
    }

    @Test
    void isNotChangedByCallerArrays() {
        int[] dims = {2, 3};
        Shape s = Shape.of(dims);
        dims[0] = 9;
        s.toArray()[1] = 9;
        assertEquals(Shape.of(2, 3), s);
    }

    @Test
    void rejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> Shape.of(2, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> Shape.of(2).dim(1));
    }
}
