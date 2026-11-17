package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RingAttentionTest {

    @ParameterizedTest
    @CsvSource({"false,1", "false,3", "true,3", "true,4", "true,11"})
    void matchesFullAttentionWithGradients(boolean causal, int shards) {
        SplittableRandom r = new SplittableRandom(7);
        Shape s = Shape.of(2, 2, 11, 4);
        Tensor q = Tensor.normal(s, 0, 2, r);
        Tensor k = Tensor.normal(s, 0, 2, r);
        Tensor v = Tensor.normal(s, 0, 1, r);
        Tensor seed = Tensor.normal(s, 0, 1, r);
        float scale = 0.5f;

        Var q1 = Var.param(q);
        Var k1 = Var.param(k);
        Var v1 = Var.param(v);
        Var full = Functional.attention(q1, k1, v1, causal, scale);
        full.backward(seed);

        Var q2 = Var.param(q);
        Var k2 = Var.param(k);
        Var v2 = Var.param(v);
        RingAttention.Stats[] st = new RingAttention.Stats[1];
        Var ring = RingAttention.attention(q2, k2, v2, causal, scale, shards, st);
        ring.backward(seed);

        assertArrayEquals(full.value().toArray(), ring.value().toArray(), 1e-5f);
        assertArrayEquals(q1.grad().toArray(), q2.grad().toArray(), 1e-4f);
        assertArrayEquals(k1.grad().toArray(), k2.grad().toArray(), 1e-4f);
        assertArrayEquals(v1.grad().toArray(), v2.grad().toArray(), 1e-4f);

        int pairs = shards * shards;
        int skipped = causal ? shards * (shards - 1) / 2 : 0;
        assertEquals(new RingAttention.Stats(pairs - skipped, skipped, shards * (shards - 1),
                (int) Math.pow(Math.ceilDiv(11, shards), 2)), st[0]);
    }
}
