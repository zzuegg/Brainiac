package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** The fused attention kernel gives the values and gradients of the composite one. */
class FusedAttentionTest {

    @ParameterizedTest
    @CsvSource({
        "2, 4, 4, 5, 5, 6, 6, true",
        "2, 4, 2, 5, 7, 6, 3, true",
        "1, 3, 1, 1, 9, 4, 5, true",
        "2, 2, 2, 6, 4, 8, 8, false",
        "1, 8, 2, 33, 33, 64, 64, true",
    })
    void matchesTheComposite(int batch, int heads, int kvHeads, int tq, int tk, int dk, int dv, boolean causal) {
        SplittableRandom r = new SplittableRandom(batch * 31L + tq);
        Tensor q = Tensor.uniform(Shape.of(batch, heads, tq, dk), -1, 1, r);
        Tensor k = Tensor.uniform(Shape.of(batch, kvHeads, tk, dk), -1, 1, r);
        Tensor v = Tensor.uniform(Shape.of(batch, kvHeads, tk, dv), -1, 1, r);
        Tensor seed = Tensor.uniform(Shape.of(batch, heads, tq, dv), -1, 1, r);
        float scale = (float) (1 / Math.sqrt(dk));

        Var[] fused = {Var.param(q), Var.param(k), Var.param(v)};
        Var a = Functional.attention(fused[0], fused[1], fused[2], causal, scale);
        a.backward(seed);
        Var[] plain = {Var.param(q), Var.param(k), Var.param(v)};
        Tensor bias = causal ? Functional.causalMask(tq, tk, q.device()) : null;
        Var b = Functional.attention(plain[0], plain[1], plain[2], bias, scale, 0, null);
        b.backward(seed);

        assertEquals(b.shape(), a.shape());
        assertArrayEquals(b.value().toArray(), a.value().toArray(), 1e-5f);
        for (int i = 0; i < 3; i++) {
            assertArrayEquals(plain[i].grad().toArray(), fused[i].grad().toArray(), 1e-5f, "gradient " + i);
        }
    }
}
