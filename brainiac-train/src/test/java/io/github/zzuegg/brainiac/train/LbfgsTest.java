package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import org.junit.jupiter.api.Test;

/** L-BFGS finds the Rosenbrock minimum; minimize runs a whole step. */
class LbfgsTest {

    @Test
    void solvesRosenbrock() {
        Var x = Var.param(Tensor.of(-1.2f));
        Var y = Var.param(Tensor.of(1f));
        // (1 - x)^2 + 100 (y - x^2)^2, minimum 0 at (1, 1)
        Lbfgs.Result r = Lbfgs.defaults().minimize(List.of(x, y),
                () -> x.neg().add(1).square().add(y.sub(x.square()).square().mul(100)).sum(), 200);
        assertEquals(1, x.value().item(), 1e-3);
        assertEquals(1, y.value().item(), 2e-3);
        assertTrue(r.loss() < 1e-6, "loss " + r.loss());
    }

    @Test
    void minimizeTakesAWholeStepWithoutStaleGradients() {
        Var w = Var.param(Tensor.of(3f));
        Optimizer sgd = Optimizer.sgd(0.1f);
        w.mul(100).sum().backward();   // a stale gradient that must not leak into the step
        Var l = sgd.minimize(List.of(w), () -> w.square().sum());
        assertEquals(9f, l.value().item(), 1e-6f);
        assertEquals(3 - 0.1f * 6, w.value().item(), 1e-6f);
        assertEquals(null, w.grad());
    }
}
