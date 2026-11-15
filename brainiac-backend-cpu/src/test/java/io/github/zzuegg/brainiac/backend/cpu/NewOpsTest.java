package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

/** Error function, softplus, masks and element-wise selection. */
class NewOpsTest {

    @Test
    void erfMatchesKnownValues() {
        float[] x = {0f, 0.5f, -1f, 2f, 3f, -6f};
        float[] expected = {0f, 0.5204999f, -0.8427008f, 0.9953223f, 0.9999779f, -1f};
        assertArrayEquals(expected, Tensor.of(x).erf().toArray(), 2e-7f);
    }

    @Test
    void softplusNeverOverflows() {
        float[] y = Tensor.of(-100f, 0f, 1f, 100f, 1000f).softplus().toArray();
        assertArrayEquals(new float[] {0f, (float) Math.log(2), (float) Math.log1p(Math.E), 100f, 1000f}, y, 1e-6f);
    }

    @Test
    void maskDropsInfinitiesAndNaN() {
        Tensor a = Tensor.of(Float.POSITIVE_INFINITY, Float.NaN, 2f);
        assertArrayEquals(new float[] {0f, 0f, 2f}, a.mask(Tensor.of(0f, 0f, 1f)).toArray());
    }

    @Test
    void wherePicksPerElementWithBroadcasting() {
        Tensor cond = Tensor.of(Shape.of(2, 1), 1, 0);
        Tensor t = Tensor.of(Shape.of(2, 2), 1, 2, 3, 4);
        Tensor f = Tensor.of(Float.POSITIVE_INFINITY, -1);
        assertArrayEquals(new float[] {1, 2, Float.POSITIVE_INFINITY, -1}, Tensor.where(cond, t, f).toArray());
    }
}
