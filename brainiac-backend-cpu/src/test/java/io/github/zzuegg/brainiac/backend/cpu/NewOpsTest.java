package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

/** Error function, softplus, rounding, masks and element-wise selection. */
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
    void roundGoesToTheNearestWholeNumberTiesToEven() {
        float[] y = Tensor.of(-2.5f, -1.6f, -0.5f, 0.4f, 0.5f, 1.5f, 2.5f, 126.7f).round().toArray();
        assertArrayEquals(new float[] {-2f, -2f, -0f, 0f, 0f, 2f, 2f, 127f}, y);
        float[] wide = Tensor.of(-2.5f, 1.5f, 2.5f).to(io.github.zzuegg.brainiac.tensor.DType.FLOAT64).round().toArray();
        assertArrayEquals(new float[] {-2f, 2f, 2f}, wide);
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

    @Test
    void sortIsStableAndPutsNanLargest() {
        Tensor x = Tensor.of(Shape.of(2, 5), 3f, 1f, Float.NaN, 1f, -2f, 0f, -0f, 5f, 5f, -1f);
        Tensor[] up = x.sort(-1, false);
        assertArrayEquals(new float[] {-2f, 1f, 1f, 3f, Float.NaN, -1f, 0f, -0f, 5f, 5f}, up[0].toArray());
        assertArrayEquals(new float[] {4, 1, 3, 0, 2, 4, 0, 1, 2, 3}, up[1].toArray());
        Tensor[] down = x.sort(1, true);
        assertArrayEquals(new float[] {Float.NaN, 3f, 1f, 1f, -2f, 5f, 5f, 0f, -0f, -1f}, down[0].toArray());
        assertArrayEquals(new float[] {2, 0, 1, 3, 4, 2, 3, 0, 1, 4}, down[1].toArray());
        // along the first axis
        assertArrayEquals(new float[] {1, 1, 1, 0, 0, 0, 0, 0, 1, 1}, x.argsort(0, false).toArray());
    }

    @Test
    void customKernelsRunTheirJavaVersionOnTheCpu() {
        io.github.zzuegg.brainiac.tensor.Kernel axpb = io.github.zzuegg.brainiac.tensor.Kernel.of("axpb", 2, null,
                (in, out, ints, floats) -> {
                    for (int i = 0; i < out.length; i++) {
                        out[i] = floats[0] * in[0][i] + in[1][i] + ints[0];
                    }
                });
        Tensor x = Tensor.of(1f, 2f, 3f);
        Tensor y = Tensor.of(10f, 20f, 30f);
        Tensor z = Tensor.custom(axpb, x.shape(), io.github.zzuegg.brainiac.tensor.DType.FLOAT32, new int[] {1},
                new float[] {2}, x, y);
        assertArrayEquals(new float[] {13f, 25f, 37f}, z.toArray());
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> Tensor.custom(axpb, x.shape(), io.github.zzuegg.brainiac.tensor.DType.FLOAT32, new int[0],
                        new float[0], x));
    }
}
