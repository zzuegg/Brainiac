package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tensor.TernaryMatrix;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BitLinearTest {

    @Test
    void packedProductMatchesTheFakeQuantizedForward() {
        BitLinear layer = BitLinear.of(100, 7, 1);   // not a multiple of 64 inputs
        Tensor x = Tensor.normal(Shape.of(3, 100), 0, 1, new SplittableRandom(2));
        float[] dense = layer.forward(Var.of(x)).value().toArray();
        TernaryMatrix m = layer.pack();
        float[] packed = m.matmul(x.toArray(), 3);
        for (int i = 0; i < dense.length; i++) {
            assertEquals(dense[i], packed[i], 1e-3 * (1 + Math.abs(dense[i])));
        }
        assertEquals(2L * 7 * 2 * 8, m.bytes());
        assertTrue(m.sparsity() > 0.2 && m.sparsity() < 0.6, "about a third of absmean ternary weights are 0");
    }

    @Test
    void weightsAreTernaryAndGradientsPassStraightThrough() {
        Tensor w = Tensor.normal(Shape.of(8, 4), 0, 1, new SplittableRandom(3));
        float scale = w.abs().mean().item();
        for (float v : BitLinear.quantizeWeights(w).toArray()) {
            float q = v / scale;
            assertTrue(Math.abs(q - Math.round(q)) < 1e-4 && Math.abs(q) <= 1.0001f, "ternary: " + q);
        }
        BitLinear layer = BitLinear.of(8, 4, 4);
        Tensor x = Tensor.normal(Shape.of(5, 8), 0, 1, new SplittableRandom(5));
        layer.forward(Var.of(x)).sum().backward();
        // d sum(x_q W_q) / dW = x_q^T 1, as if W_q were W
        Tensor expected = BitLinear.quantizeActivations(x).transpose().matmul(Tensor.ones(Shape.of(5, 4)));
        assertArrayEquals(expected.toArray(), layer.weight().grad().toArray(), 1e-4f);
    }
}
