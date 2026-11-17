package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class NoGradTest {

    @Test
    void operationsInsideRecordNothingAndGiveTheSameValues() {
        Var w = Var.param(Tensor.of(Shape.of(2, 2), 1, 2, 3, 4));
        Var x = Var.of(Tensor.of(Shape.of(1, 2), 5, 6));
        Var with = x.matmul(w).tanh();
        assertTrue(with.requiresGrad());
        assertTrue(Var.gradEnabled());
        Var without = Var.noGrad(() -> {
            assertFalse(Var.gradEnabled());
            Var nested = Var.noGrad(() -> x.matmul(w));
            assertFalse(Var.gradEnabled(), "nested calls keep it off");
            return nested.tanh();
        });
        assertTrue(Var.gradEnabled());
        assertFalse(without.requiresGrad());
        assertArrayEquals(with.value().toArray(), without.value().toArray());
    }
}
