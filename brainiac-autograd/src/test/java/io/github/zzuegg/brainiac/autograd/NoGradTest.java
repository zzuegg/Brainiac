package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class NoGradTest {

    @Test
    void recordsNothingInsideAndEverythingAfter() {
        Var w = Var.param(Tensor.of(1f, 2f, 3f));
        assertTrue(Var.gradEnabled());
        Var y = Var.noGrad(() -> {
            assertFalse(Var.gradEnabled());
            Var inner = Var.noGrad(() -> w.mul(2f));   // nested
            assertFalse(Var.gradEnabled());
            return inner.square().sum();
        });
        assertTrue(Var.gradEnabled());
        assertFalse(y.requiresGrad());
        assertArrayEquals(new float[] {56f}, y.value().toArray(), 0f);

        Var z = w.square().sum();
        assertTrue(z.requiresGrad());
        z.backward();
        assertNotNull(w.grad());
        assertArrayEquals(new float[] {2f, 4f, 6f}, w.grad().toArray(), 0f);
    }

    @Test
    void restoresTheModeAfterAnException() {
        try {
            Var.noGrad(() -> {
                throw new IllegalStateException("boom");
            });
        } catch (IllegalStateException expected) {
            // the mode must be back on
        }
        assertTrue(Var.gradEnabled());
    }
}
