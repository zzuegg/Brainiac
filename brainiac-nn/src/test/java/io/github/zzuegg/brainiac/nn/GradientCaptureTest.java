package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class GradientCaptureTest {

    private static void close(float[] want, float[] got) {
        assertEquals(want.length, got.length);
        for (int i = 0; i < want.length; i++) {
            assertEquals(want[i], got[i], 1e-5f, "[" + i + "]");
        }
    }

    @Test
    void hooksSeeAndReplaceGradients() {
        Var x = Var.param(Tensor.of(Shape.of(3), 1f, 2f, 3f));
        Var y = x.mul(x);
        List<Tensor> seen = new ArrayList<>();
        y.onGradient(g -> {
            seen.add(g);
            return g.mul(0.5f);
        });
        // y feeds the loss twice: the hook sees the summed gradient once
        y.add(y).sum().backward();
        assertEquals(1, seen.size());
        assertArrayEquals(new float[] {2, 2, 2}, seen.get(0).toArray());
        // d/dx of x^2 with the halved gradient: 0.5 * 2 * 2x
        assertArrayEquals(new float[] {2, 4, 6}, x.grad().toArray());
        assertThrows(IllegalStateException.class, () -> Var.of(Tensor.zeros(Shape.of(2))).onGradient(g -> g));
    }

    @Test
    void denseCapturesWhatKfacNeeds() {
        Dense d = new Dense(3, 2, Activation.TANH, Init.XAVIER, new SplittableRandom(1));
        List<Tensor[]> captured = new ArrayList<>();
        d.capture((in, grad) -> captured.add(new Tensor[] {in, grad}));
        Tensor x = Tensor.uniform(Shape.of(4, 3), -1, 1, new SplittableRandom(2));
        d.forward(Var.of(x)).square().sum().backward();
        assertEquals(1, captured.size());
        assertEquals(Shape.of(4, 2), captured.get(0)[1].shape());
        // the weight's gradient is the input times the pre-activation gradient
        close(x.transpose().matmul(captured.get(0)[1]).toArray(), d.weight().grad().toArray());
        d.capture(null);
        d.forward(Var.of(x)).sum().backward();
        assertEquals(1, captured.size());
    }
}
