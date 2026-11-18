package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Sinkhorn plans keep their marginals, approach exact transport and carry gradients. */
class OptimalTransportTest {

    @Test
    void thePlanHasTheMarginalsAndFindsTheMatching() {
        // three points onto the same points shuffled: the exact plan is the permutation
        Var x = Var.of(Tensor.of(Shape.of(3, 1), 0f, 1f, 2f));
        Var y = Var.of(Tensor.of(Shape.of(3, 1), 2f, 0f, 1f));
        Tensor w = Tensor.of(1 / 3f, 1 / 3f, 1 / 3f);
        Var cost = OptimalTransport.squaredDistances(x, y);
        float[] p = OptimalTransport.sinkhorn(cost, w, w, 0.05f, 200).value().toArray();
        for (int i = 0; i < 3; i++) {
            assertEquals(1 / 3f, p[3 * i] + p[3 * i + 1] + p[3 * i + 2], 1e-4f);
            assertEquals(1 / 3f, p[i] + p[3 + i] + p[6 + i], 1e-4f);
        }
        // 0 -> column 1, 1 -> column 2, 2 -> column 0
        assertEquals(1 / 3f, p[1], 1e-3f);
        assertEquals(1 / 3f, p[5], 1e-3f);
        assertEquals(1 / 3f, p[6], 1e-3f);
        assertEquals(0f, OptimalTransport.sinkhornCost(cost, w, w, 0.05f, 200).value().item(), 1e-3f);
    }

    @Test
    void gradientsReachThePoints() {
        Tensor a = Tensor.of(0.5f, 0.5f);
        Tensor b = Tensor.of(0.25f, 0.25f, 0.5f);
        Tensor y = Tensor.of(Shape.of(3, 2), 0f, 0f, 1f, 0f, 0f, 1f);
        GradCheck.Result r = GradCheck.check(v -> OptimalTransport.sinkhornCost(
                OptimalTransport.squaredDistances(v.get(0), Var.of(y)), a, b, 0.5f, 30),
                List.of(Tensor.of(Shape.of(2, 2), 0.2f, 0.3f, 0.7f, 0.4f)));
        assertTrue(r.passed(), r.toString());
    }
}
