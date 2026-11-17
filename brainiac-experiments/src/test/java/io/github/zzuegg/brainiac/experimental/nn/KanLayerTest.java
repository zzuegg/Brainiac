package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KanLayerTest {
    @Test
    void basisIsAPartitionOfUnityInsideTheGrid() {
        KanLayer k = KanLayer.of(3, 2, 5, -1, 1, 1);
        Tensor x = Tensor.uniform(Shape.of(10, 3), -0.99f, 0.99f, new SplittableRandom(2));
        Tensor b = k.basis(Var.of(x)).value();
        assertEquals(8, b.shape().dim(2));
        float[] sums = b.sum(2).toArray();
        for (float s : sums) {
            assertEquals(1, s, 1e-5);
        }
        float[] outside = k.basis(Var.of(Tensor.of(Shape.of(1, 3), 2.5f, -3, 9))).value().toArray();
        assertArrayEquals(new float[outside.length], outside, 0);
    }

    @Test
    void gradientsReachInputsAndCoefficients() {
        KanLayer k = KanLayer.of(2, 3, 4, -1, 1, 3);
        Tensor x = Tensor.uniform(Shape.of(5, 2), -0.9f, 0.9f, new SplittableRandom(4));
        GradCheck.Result g = GradCheck.check(v -> k.forward(v.get(0)).square().sum(), List.of(x), 1e-3f, 2e-2f);
        assertTrue(g.passed(), g.toString());
        Var out = k.forward(Var.of(x)).sum();
        out.backward();
        for (Var p : k.parameters()) {
            assertTrue(p.grad().abs().sum().item() > 0);
        }
    }

    @Test
    void edgeFunctionsAgreeWithForward() {
        KanLayer k = KanLayer.of(1, 2, 6, -1, 1, 5);
        float[] xs = {-0.7f, 0.1f, 0.55f};
        float[] f = k.forward(Var.of(Tensor.of(Shape.of(3, 1), xs))).value().toArray();
        float[] e1 = k.edge(0, 1, xs);
        for (int n = 0; n < 3; n++) {
            assertEquals(f[n * 2 + 1], e1[n], 1e-5);
        }
    }

    @Test
    void gridExtensionKeepsTheFunction() {
        KanLayer k = KanLayer.of(2, 2, 3, -1, 1, 6);
        // make the splines non-trivial
        k.parameters().get(1).assign(Tensor.normal(Shape.of(2 * 6, 2), 0, 1, new SplittableRandom(7)));
        Tensor samples = Tensor.uniform(Shape.of(400, 2), -1, 1, new SplittableRandom(8));
        KanLayer fine = k.refine(12, -1, 1, samples);
        assertEquals(12, fine.intervals());
        Tensor test = Tensor.uniform(Shape.of(50, 2), -0.95f, 0.95f, new SplittableRandom(9));
        float diff = k.forward(Var.of(test)).value().sub(fine.forward(Var.of(test)).value()).abs().max().item();
        assertTrue(diff < 1e-3, "max difference " + diff);   // a cubic on 3 intervals is exactly a cubic on 12
    }
}
