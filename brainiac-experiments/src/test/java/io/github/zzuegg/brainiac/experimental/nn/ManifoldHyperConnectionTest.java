package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ManifoldHyperConnectionTest {

    @Test
    void sinkhornGivesDoublyStochasticMatricesClosedUnderProducts() {
        Var product = null;
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < 30; i++) {
            Var m = ManifoldHyperConnection.sinkhorn(Var.of(Tensor.uniform(Shape.of(4, 4), -3, 3, r)), 30);
            product = product == null ? m : product.matmul(m);
        }
        float[] p = product.value().toArray();
        for (int i = 0; i < 4; i++) {
            float row = 0;
            float col = 0;
            for (int j = 0; j < 4; j++) {
                row += p[i * 4 + j];
                col += p[j * 4 + i];
                assertTrue(p[i * 4 + j] >= 0);
            }
            assertEquals(1, row, 1e-3f);
            assertEquals(1, col, 1e-3f);
        }
    }

    @Test
    void residualMixingIsDoublyStochasticPerToken() {
        ManifoldHyperConnection mhc = new ManifoldHyperConnection(
                new GatedAttention(8, 2, true, 0f, new SplittableRandom(2)), 3, 8, new SplittableRandom(3));
        Var x = Var.of(Tensor.uniform(Shape.of(2, 4, 3, 8), -1, 1, new SplittableRandom(4)));
        float[] m = mhc.residualMixing(x).toArray();
        for (int k = 0; k < m.length / 9; k++) {
            for (int i = 0; i < 3; i++) {
                assertEquals(1, m[k * 9 + i * 3] + m[k * 9 + i * 3 + 1] + m[k * 9 + i * 3 + 2], 1e-3f);
            }
        }
        assertEquals(Shape.of(2, 4, 3, 8), mhc.forward(x).shape());
    }

    @Test
    void stackOfHyperConnectedAttentionLearnsRecall() {
        SplittableRandom r = new SplittableRandom(5);
        List<ManifoldHyperConnection> blocks = List.of(
                new ManifoldHyperConnection(new GatedAttention(16, 2, true, 0f, r), 2, 16, r),
                new ManifoldHyperConnection(new GatedAttention(16, 2, true, 0f, r), 2, 16, r));
        Layer stack = new Layer() {
            @Override
            public Var forward(Var input) {
                Var x = ManifoldHyperConnection.expand(input, 2);
                for (ManifoldHyperConnection b : blocks) {
                    x = b.forward(x);
                }
                return ManifoldHyperConnection.collapse(x);
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>();
                blocks.forEach(b -> p.addAll(b.parameters()));
                return p;
            }
        };
        double acc = MixerHarness.train(stack, MixerHarness.recall(6, 6, 3), 16, 250, 0.02f, 6);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }
}
