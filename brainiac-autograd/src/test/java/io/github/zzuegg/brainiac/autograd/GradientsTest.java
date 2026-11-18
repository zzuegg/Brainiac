package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Global-norm and value clipping of gradients. */
class GradientsTest {

    @Test
    void clipsToTheGlobalNorm() {
        Var a = Var.param(Tensor.of(1, 2));
        Var b = Var.param(Tensor.of(3));
        a.mul(Var.of(Tensor.of(3, 0))).sum().add(b.mul(4).sum()).backward();   // gradients [3, 0] and [4]: norm 5
        assertEquals(5f, Gradients.globalNorm(List.of(a, b)).item(), 1e-6f);
        assertEquals(5f, Gradients.clipByGlobalNorm(List.of(a, b), 1).item(), 1e-6f);
        assertArrayEquals(new float[] {0.6f, 0}, a.grad().toArray(), 1e-5f);
        assertArrayEquals(new float[] {0.8f}, b.grad().toArray(), 1e-5f);
        Gradients.clipByGlobalNorm(List.of(a, b), 10);   // under the limit: unchanged
        assertArrayEquals(new float[] {0.8f}, b.grad().toArray(), 1e-5f);
        Gradients.clipByValue(List.of(a, b), 0.7f);
        assertArrayEquals(new float[] {0.7f}, b.grad().toArray(), 0);
    }
}
