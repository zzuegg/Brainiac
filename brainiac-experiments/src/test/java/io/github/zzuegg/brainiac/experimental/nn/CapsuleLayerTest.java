package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CapsuleLayerTest {
    @Test
    void squashKeepsDirectionAndBoundsLength() {
        float[] v = CapsuleLayer.squash(Var.of(Tensor.of(Shape.of(2, 2), 3, 4, 0.03f, 0.04f))).value().toArray();
        assertEquals(25f / 26 * 0.6f, v[0], 1e-5);
        assertEquals(25f / 26 * 0.8f, v[1], 1e-5);
        assertEquals(0.0025f / 1.0025f * 0.6f, v[2], 1e-6);
    }

    @Test
    void routingSharpensCouplings() {
        Tensor u = Tensor.normal(Shape.of(4, 10, 5), 0, 1, new SplittableRandom(1));
        double[] sharpness = new double[3];
        for (int r = 1; r <= 3; r++) {
            CapsuleLayer c = CapsuleLayer.of(10, 5, 3, 4, r, 2);
            Var out = c.forward(Var.of(u));
            assertEquals(Shape.of(4, 3, 4), out.shape());
            sharpness[r - 1] = c.lastCoupling().max(2, true).mean().item();
        }
        assertEquals(1.0 / 3, sharpness[0], 1e-6);   // first iteration: uniform
        assertTrue(sharpness[2] > sharpness[1] && sharpness[1] > sharpness[0]);
    }

    @Test
    void gradientsAndMarginLoss() {
        CapsuleLayer c = CapsuleLayer.of(3, 2, 2, 3, 3, 4);
        Tensor u = Tensor.normal(Shape.of(2, 3, 2), 0, 1, new SplittableRandom(5));
        GradCheck.Result g = GradCheck.check(v -> CapsuleLayer.lengths(c.forward(v.get(0))).sum(), List.of(u),
                1e-3f, 2e-2f);
        assertTrue(g.passed(), g.toString());
        // lengths 0.95 (present) and 0.05 (absent) cost nothing; 0.5 present costs 0.16
        Var lengths = Var.of(Tensor.of(Shape.of(2, 2), 0.95f, 0.05f, 0.5f, 0.05f));
        float loss = CapsuleLayer.MARGIN_LOSS.compute(lengths, Tensor.of(Shape.of(2), 0, 0)).value().item();
        assertEquals(0.16f / 2, loss, 1e-6);
    }
}
