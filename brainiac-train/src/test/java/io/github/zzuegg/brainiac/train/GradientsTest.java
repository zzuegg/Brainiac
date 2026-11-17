package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class GradientsTest {

    private static Var withGrad(float... g) {
        Var v = Var.param(Tensor.zeros(Shape.of(g.length)));
        v.setGrad(Tensor.of(g));
        return v;
    }

    @Test
    void clipsTheGlobalNormAndReturnsItBefore() {
        Var a = withGrad(3, 0);
        Var b = withGrad(0, 4);
        Var none = Var.param(Tensor.zeros(Shape.of(1)));
        Tensor before = Gradients.clipNorm(List.of(a, b, none), 1);
        assertEquals(5, before.item(), 1e-5);
        assertArrayEquals(new float[] {0.6f, 0}, a.grad().toArray(), 1e-6f);
        assertArrayEquals(new float[] {0, 0.8f}, b.grad().toArray(), 1e-6f);
        assertNull(none.grad());
        assertEquals(1, Gradients.norm(List.of(a, b)).item(), 1e-5);
        // below the limit: unchanged
        Gradients.clipNorm(List.of(a, b), 2);
        assertArrayEquals(new float[] {0.6f, 0}, a.grad().toArray(), 0);
        assertNull(Gradients.norm(List.of(none)));
    }

    @Test
    void clipsValues() {
        Var a = withGrad(-3, 0.5f, 2);
        Gradients.clipValue(List.of(a), 1);
        assertArrayEquals(new float[] {-1, 0.5f, 1}, a.grad().toArray(), 0);
    }

    @Test
    void muonOrthogonalizeIsPublicAndNearlyOrthogonal() {
        Tensor g = Tensor.uniform(Shape.of(4, 6), -1, 1, new SplittableRandom(1));
        Tensor o = Muon.orthogonalize(g, 5);
        float[] gram = o.matmul(o.transpose()).toArray();
        for (int i = 0; i < 4; i++) {
            // the quintic iteration leaves singular values near 1, roughly 0.7 to 1.2
            assertEquals(1, gram[i * 4 + i], 0.6, "diagonal " + i);
        }
    }
}
