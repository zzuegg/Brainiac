package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class WhereTest {

    @Test
    void unusedOverflowStaysOutOfTheValue() {
        Var x = Var.param(Tensor.of(-1f, 200f));
        // exp(200) overflows to infinity but is never picked
        Var picked = Var.where(x.greater(0), x, x.exp());
        assertArrayEquals(new float[] {(float) Math.exp(-1), 200f}, picked.value().toArray(), 1e-6f);
        // with the unused branch kept finite, the gradient is clean too
        Var y = Var.where(x.greater(0), x, x.minimum(0).exp());
        y.sum().backward();
        assertArrayEquals(new float[] {(float) Math.exp(-1), 1f}, x.grad().toArray(), 1e-6f);
    }

    @Test
    void tiesSendTheGradientToTheOtherValue() {
        Var a = Var.param(Tensor.of(1f, 2f));
        Var b = Var.param(Tensor.of(1f, 1f));
        a.maximum(b).sum().backward();
        assertArrayEquals(new float[] {0f, 1f}, a.grad().toArray());
        assertArrayEquals(new float[] {1f, 0f}, b.grad().toArray());
    }
}
