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

class GlobalWorkspaceLayerTest {
    @Test
    void exactlyTopKSpecialistsWrite() {
        GlobalWorkspaceLayer g = GlobalWorkspaceLayer.of(8, 2, 3, 1);
        Var h = Var.of(Tensor.normal(Shape.of(4, 7, 8), 0, 1, new SplittableRandom(2)));
        assertEquals(Shape.of(4, 7, 8), g.forward(h).shape());
        for (boolean[] row : g.lastWriters()) {
            int n = 0;
            for (boolean w : row) {
                n += w ? 1 : 0;
            }
            assertEquals(3, n);
        }
    }

    @Test
    void gradientsAreCorrectAwayFromTies() {
        GlobalWorkspaceLayer g = GlobalWorkspaceLayer.of(4, 2, 2, 3);
        Tensor h = Tensor.normal(Shape.of(2, 4, 4), 0, 1, new SplittableRandom(4));
        GradCheck.Result r = GradCheck.check(v -> g.forward(v.get(0)).square().sum(), List.of(h), 1e-3f, 3e-2f);
        assertTrue(r.passed(), r.toString());
    }
}
