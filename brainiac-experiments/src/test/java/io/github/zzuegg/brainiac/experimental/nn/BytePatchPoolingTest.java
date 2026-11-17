package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class BytePatchPoolingTest {

    @Test
    void poolsPatchMeansAndUnpoolsThePreviousPatch() {
        BytePatchPooling bridge = new BytePatchPooling(3, 3, new SplittableRandom(1));
        int[] ids = {0, 0, 1, 1, 1, 2};
        Tensor bytes = Tensor.uniform(Shape.of(6, 3), -1, 1, new SplittableRandom(2));
        Var down = bridge.parameters().get(0);
        Var pooled = bridge.pool(Var.of(bytes), ids);
        assertEquals(Shape.of(3, 3), pooled.shape());
        float[] b = bytes.toArray();
        float[] mean1 = new float[3];
        for (int i = 2; i < 5; i++) {
            for (int j = 0; j < 3; j++) {
                mean1[j] += b[i * 3 + j] / 3;
            }
        }
        float[] expected = Var.of(Tensor.of(Shape.of(1, 3), mean1)).matmul(down).value().toArray();
        assertArrayEquals(expected, pooled.value().slice(0, 1, 2).toArray(), 1e-5f);
        Var un = bridge.unpool(pooled, ids);
        assertEquals(Shape.of(6, 3), un.shape());
        // bytes of patch 1 see patch 0's latent, not their own
        float[] u = un.value().toArray();
        float[] p0 = pooled.slice(0, 0, 1).matmul(bridge.parameters().get(1)).value().toArray();
        assertArrayEquals(p0, java.util.Arrays.copyOfRange(u, 2 * 3, 3 * 3), 1e-5f);
        GradCheck.Result r = GradCheck.check(v -> bridge.pool(v.get(0), ids).square().sum(), List.of(bytes));
        assertTrue(r.passed(), r.toString());
    }
}
