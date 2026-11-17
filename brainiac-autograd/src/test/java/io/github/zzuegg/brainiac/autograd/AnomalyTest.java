package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class AnomalyTest {

    @AfterEach
    void off() {
        Var.anomalyDetection(false);
    }

    @Test
    void forwardNaNNamesTheOperationAndCaller() {
        Var x = Var.param(Tensor.of(Shape.of(2), new float[] {1, -1}));
        Var.anomalyDetection(true);
        ArithmeticException e = assertThrows(ArithmeticException.class, () -> x.mul(2f).log());
        assertTrue(e.getMessage().startsWith("log at AnomalyTest."), e.getMessage());
    }

    @Test
    void backwardNaNNamesTheOperation() {
        Var x = Var.param(Tensor.of(Shape.of(2), new float[] {0, 4}));
        Var.anomalyDetection(true);
        Var loss = x.sqrt().mul(0f).sum();
        ArithmeticException e = assertThrows(ArithmeticException.class, loss::backward);
        assertTrue(e.getMessage().contains("gradient of sqrt at AnomalyTest."), e.getMessage());
    }

    @Test
    void offByDefaultAndInfinitiesPass() {
        Var x = Var.param(Tensor.of(Shape.of(2), new float[] {1, -1}));
        assertDoesNotThrow(() -> x.log());
        Var.anomalyDetection(true);
        Var y = Var.param(Tensor.of(Shape.of(2), new float[] {0, 2}));
        assertDoesNotThrow(() -> y.log().sum());
    }
}
