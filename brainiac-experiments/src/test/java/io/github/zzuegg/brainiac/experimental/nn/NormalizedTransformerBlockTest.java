package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class NormalizedTransformerBlockTest {

    private static void assertUnitRows(float[] v, int width) {
        for (int r = 0; r < v.length / width; r++) {
            double n = 0;
            for (int i = 0; i < width; i++) {
                n += v[r * width + i] * v[r * width + i];
            }
            assertEquals(1, Math.sqrt(n), 1e-4);
        }
    }

    @Test
    void hiddenStatesStayOnTheSphereAndWeightsRenormalize() {
        NormalizedTransformerBlock block = new NormalizedTransformerBlock(8, 2, 16, true, 10000f, 0.1f,
                new SplittableRandom(1));
        Var x = Var.of(Tensor.uniform(Shape.of(2, 5, 8), -3, 3, new SplittableRandom(2)));
        assertUnitRows(block.forward(x).value().toArray(), 8);
        Var w = block.parameters().get(0);
        w.assign(w.value().mul(3f));
        block.renormalize();
        assertUnitRows(w.value().transpose().toArray(), 8);
    }

    @Test
    void learnsAssociativeRecall() {
        NormalizedTransformerBlock block = new NormalizedTransformerBlock(16, 2, 32, true, 0f, 0.5f,
                new SplittableRandom(3));
        double acc = MixerHarness.train(block, MixerHarness.recall(6, 6, 3), 16, 250, 0.02f, 4, null,
                block::renormalize);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }
}
