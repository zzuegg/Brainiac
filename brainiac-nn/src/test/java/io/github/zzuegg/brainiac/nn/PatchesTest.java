package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class PatchesTest {

    @Test
    void patchifyLaysOutPatchesAndInverts() {
        float[] v = new float[2 * 3 * 4 * 6];
        for (int i = 0; i < v.length; i++) {
            v[i] = i;
        }
        Var img = Var.of(Tensor.of(Shape.of(2, 3, 4, 6), v));
        Var p = Functional.patchify(img, 2);
        assertArrayEquals(new int[] {2, 6, 12}, p.shape().toArray());
        // batch 1, patch (row 1, col 2), channel 2, in-patch (1, 0)
        float got = p.value().toArray()[(1 * 6 + 1 * 3 + 2) * 12 + 2 * 4 + 1 * 2 + 0];
        float want = v[((1 * 3 + 2) * 4 + (1 * 2 + 1)) * 6 + (2 * 2 + 0)];
        assertEquals(want, got);
        assertArrayEquals(v, Functional.unpatchify(p, 3, 4, 6, 2).value().toArray());
    }

    @Test
    void rotary2dDependsOnRelativeGridOffsets() {
        SplittableRandom r = new SplittableRandom(3);
        int rows = 3;
        int cols = 4;
        int d = 8;
        float[] q = new float[d];
        float[] k = new float[d];
        for (int i = 0; i < d; i++) {
            q[i] = (float) r.nextGaussian();
            k[i] = (float) r.nextGaussian();
        }
        float[] qs = new float[rows * cols * d];
        float[] ks = new float[rows * cols * d];
        for (int t = 0; t < rows * cols; t++) {
            System.arraycopy(q, 0, qs, t * d, d);
            System.arraycopy(k, 0, ks, t * d, d);
        }
        float[] rq = Functional.rotary2d(Var.of(Tensor.of(Shape.of(rows * cols, d), qs)), rows, cols, 100).value()
                .toArray();
        float[] rk = Functional.rotary2d(Var.of(Tensor.of(Shape.of(rows * cols, d), ks)), rows, cols, 100).value()
                .toArray();
        // (0,0)·(1,1) equals (1,2)·(2,3): same offset
        assertEquals(dot(rq, 0, rk, 5, d), dot(rq, 6, rk, 11, d), 1e-4f);
        // token (0,0) is unrotated
        assertArrayEquals(q, java.util.Arrays.copyOf(rq, d), 1e-6f);
    }

    private static float dot(float[] a, int i, float[] b, int j, int d) {
        float s = 0;
        for (int x = 0; x < d; x++) {
            s += a[i * d + x] * b[j * d + x];
        }
        return s;
    }
}
