package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DifferentialAttentionTest {

    @Test
    void mapRowsSumToOneMinusLambdaAndRespectCausality() {
        DifferentialAttention layer = new DifferentialAttention(8, 2, true, 1, new SplittableRandom(1));
        Var x = Var.of(Tensor.uniform(Shape.of(2, 5, 8), -1, 1, new SplittableRandom(2)));
        float[] w = layer.weights(x).toArray();
        float lambda = layer.lambda();
        for (int row = 0; row < w.length / 5; row++) {
            float sum = 0;
            for (int j = 0; j < 5; j++) {
                sum += w[row * 5 + j];
                if (j > row % 5) {
                    assertEquals(0, w[row * 5 + j], 1e-6f, "future key leaked");
                }
            }
            assertEquals(1 - lambda, sum, 1e-4f);
        }
        assertEquals(Shape.of(2, 5, 8), layer.forward(x).shape());
    }

    @Test
    void learnsAssociativeRecall() {
        DifferentialAttention layer = new DifferentialAttention(16, 2, true, 1, new SplittableRandom(3));
        double acc = MixerHarness.train(layer, MixerHarness.recall(6, 6, 3), 16, 250, 0.02f, 4);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }
}
