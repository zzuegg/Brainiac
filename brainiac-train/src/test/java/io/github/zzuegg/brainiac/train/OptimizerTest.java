package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

import org.junit.jupiter.api.Test;

class OptimizerTest {

    /** Sets up a parameter whose gradient is {@code grad}, via loss = sum(p * grad). */
    private static Var withGradient(float[] value, float[] grad) {
        Var p = Var.param(Tensor.of(value));
        p.mul(Tensor.of(grad)).sum().backward();
        return p;
    }

    @Test
    void sgdStepsAgainstTheGradientAndClearsIt() {
        Var p = withGradient(new float[] {1, 2}, new float[] {0.5f, -1});
        Optimizer.sgd(0.1f).step(List.of(p));
        assertArrayEquals(new float[] {0.95f, 2.1f}, p.value().toArray(), 1e-6f);
        assertNull(p.grad());
    }

    @Test
    void momentumAccumulatesVelocity() {
        Optimizer opt = Optimizer.momentum(0.1f, 0.5f);
        Var p = withGradient(new float[] {0}, new float[] {1});
        opt.step(List.of(p));                  // v = 1     -> p = -0.1
        p.mul(Tensor.of(1)).sum().backward();
        opt.step(List.of(p));                  // v = 1.5   -> p = -0.25
        assertArrayEquals(new float[] {-0.25f}, p.value().toArray(), 1e-6f);
    }

    @Test
    void adamFirstStepMovesByTheLearningRate() {
        Var p = withGradient(new float[] {1, 1}, new float[] {3, -0.001f});
        Optimizer.adam(0.01f).step(List.of(p));
        // bias-corrected m/sqrt(v) is sign(grad) on the first step
        assertArrayEquals(new float[] {0.99f, 1.01f}, p.value().toArray(), 1e-4f);
    }

    @Test
    void skipsParametersWithoutGradientAndValidatesSettings() {
        Var p = Var.param(Tensor.of(1));
        Optimizer.adam(0.1f).step(List.of(p));
        assertArrayEquals(new float[] {1}, p.value().toArray());
        assertThrows(IllegalArgumentException.class, () -> Optimizer.sgd(0));
        assertThrows(IllegalArgumentException.class, () -> Optimizer.momentum(0.1f, 1));
    }
}
