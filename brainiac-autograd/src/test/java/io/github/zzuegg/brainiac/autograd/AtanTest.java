package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

/** atan, atan2, less, ones on a device and copy. */
class AtanTest {

    private static final float[] YS = {0, 0, 0, 1, -1, 1, -1, 2.5f, -0.3f, 1e-3f, 7, -7};
    private static final float[] XS = {0, 1, -1, 0, 0, -1, -1, 0.4f, -2, -5, 1e-4f, -1e-4f};

    @Test
    void atanMatchesJava() {
        float[] v = {-1e6f, -3, -1, -0.2f, 0, 0.2f, 1, 3, 1e6f};
        float[] got = Tensor.of(v).atan().toArray();
        for (int i = 0; i < v.length; i++) {
            assertEquals(Math.atan(v[i]), got[i], 1e-6, "atan " + v[i]);
        }
        double[] d = Tensor.of(v).to(DType.FLOAT64).atan().toDoubleArray();
        for (int i = 0; i < v.length; i++) {
            assertEquals(Math.atan(v[i]), d[i], 1e-14, "float64 atan " + v[i]);
        }
    }

    @Test
    void atan2MatchesJavaInEveryQuadrant() {
        float[] got = Tensor.of(YS).atan2(Tensor.of(XS)).toArray();
        for (int i = 0; i < YS.length; i++) {
            assertEquals(Math.atan2(YS[i], XS[i]), got[i], 2e-6, "atan2(" + YS[i] + ", " + XS[i] + ")");
        }
    }

    @Test
    void lessOnesAndCopy() {
        assertEquals(java.util.List.of(1f, 0f, 0f), toList(Tensor.of(1, 2, 3).less(Tensor.of(2, 2, 2))));
        Tensor ones = Tensor.ones(Shape.of(2, 2), Device.getDefault());
        assertEquals(4f, ones.sum().item());
        Tensor t = Tensor.of(1, 2, 3);
        Tensor c = t.copy();
        assertNotSame(t, c);
        c.write(0, 0, Tensor.of(9));
        assertEquals(java.util.List.of(1f, 2f, 3f), toList(t));
        assertEquals(java.util.List.of(9f, 2f, 3f), toList(c));
        assertEquals(5f, Tensor.of(5).reshape().copy().item());
    }

    private static java.util.List<Float> toList(Tensor t) {
        java.util.List<Float> out = new java.util.ArrayList<>();
        for (float f : t.toArray()) {
            out.add(f);
        }
        return out;
    }
}
