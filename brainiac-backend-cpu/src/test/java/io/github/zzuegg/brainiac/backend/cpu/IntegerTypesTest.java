package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class IntegerTypesTest {

    private static final Device CPU = Device.getDefault();

    @Test
    void wholeNumbersStayExact() {
        long big = (1L << 40) + 7;
        Tensor t = Tensor.of(Shape.of(3), new long[] {big, -3, 0}, DType.INT64, CPU);
        assertEquals(DType.INT64, t.dtype());
        assertArrayEquals(new long[] {big, -3, 0}, t.toLongArray());
        // float32 could not hold it: the integer type is what keeps token ids and offsets exact
        assertFalse((long) (float) big == big);
    }

    @Test
    void arithmeticAndPromotion() {
        Tensor a = Tensor.ofInts(Shape.of(4), new int[] {1, 2, 3, 4}, CPU);
        Tensor b = Tensor.ofInts(Shape.of(4), new int[] {10, 20, 30, 40}, CPU);
        Tensor sum = a.add(b);
        assertEquals(DType.INT32, sum.dtype());
        assertArrayEquals(new int[] {11, 22, 33, 44}, sum.toIntArray());
        assertArrayEquals(new int[] {10, 40, 90, 160}, a.mul(b).toIntArray());
        Tensor wide = Tensor.of(Shape.of(4), new long[] {1, 1, 1, 1}, DType.INT64, CPU);
        assertEquals(DType.INT64, a.add(wide).dtype());
        Tensor half = Tensor.of(Shape.of(4), 0.5f, 0.5f, 0.5f, 0.5f);
        Tensor mixed = a.add(half);
        assertEquals(DType.FLOAT32, mixed.dtype());
        assertArrayEquals(new float[] {1.5f, 2.5f, 3.5f, 4.5f}, mixed.toArray());
        // conversions round to the integer type's range
        Tensor bytes = Tensor.of(Shape.of(3), new float[] {-4f, 7.9f, 300f}, CPU).to(DType.UINT8);
        assertArrayEquals(new long[] {0, 7, 255}, bytes.toLongArray());
        Tensor flags = Tensor.of(Shape.of(3), new float[] {0f, 2f, -1f}, CPU).to(DType.BOOL);
        assertArrayEquals(new long[] {0, 1, 1}, flags.toLongArray());
    }

    @Test
    void integerIndicesGather() {
        Tensor table = Tensor.of(Shape.of(3, 2), 0f, 1f, 10f, 11f, 20f, 21f);
        Tensor ids = Tensor.of(Shape.of(2), new long[] {2, 0}, DType.INT64, CPU);
        assertArrayEquals(new float[] {20f, 21f, 0f, 1f}, table.gather(0, ids).toArray());
        assertArrayEquals(new float[] {2f, 0f}, ids.to(DType.FLOAT32).toArray());
    }
}
