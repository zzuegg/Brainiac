package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class GatedAttentionTest {

    @Test
    void closedGatesSilenceTheLayerWithoutASink() {
        GatedAttention layer = new GatedAttention(8, 2, true, 10000f, new SplittableRandom(1));
        Var x = Var.of(Tensor.uniform(Shape.of(2, 6, 8), -1, 1, new SplittableRandom(2)));
        for (float g : layer.gates(x).toArray()) {
            assertTrue(g > 0 && g < 1);
        }
        layer.gateBias().assign(Tensor.full(Shape.of(8), -30f));
        for (float y : layer.forward(x).value().toArray()) {
            assertEquals(0, y, 1e-6f);
        }
    }

    @Test
    void learnsAssociativeRecall() {
        GatedAttention layer = new GatedAttention(16, 2, true, 0f, new SplittableRandom(3));
        double acc = MixerHarness.train(layer, MixerHarness.recall(6, 6, 3), 16, 250, 0.02f, 4);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }

    @Test
    void forgetBiasIsTheSumOfLogGatesBetweenKeyAndQuery() {
        ForgettingAttention fox = new ForgettingAttention(8, 2, false, new SplittableRandom(5));
        Var x = Var.of(Tensor.uniform(Shape.of(1, 4, 8), -1, 1, new SplittableRandom(6)));
        float[] f = fox.forgetGates(x).toArray();
        float[] d = fox.forgetBias(x).toArray();
        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j <= i; j++) {
                    double expected = 0;
                    for (int l = j + 1; l <= i; l++) {
                        expected += Math.log(f[h * 4 + l]);
                    }
                    assertEquals(expected, d[(h * 4 + i) * 4 + j], 1e-4);
                }
            }
        }
    }

    @Test
    void forgettingAttentionSolvesRecencyWithoutPositionsWhereSoftmaxCannot() {
        MixerHarness.Task task = MixerHarness.recentRecall(3, 6, 4);
        double fox = MixerHarness.train(new ForgettingAttention(16, 2, true, new SplittableRandom(7)),
                task, 16, 300, 0.02f, 8);
        double plain = MixerHarness.train(new GatedAttention(16, 2, true, 0f, new SplittableRandom(7)),
                task, 16, 300, 0.02f, 8);
        assertTrue(fox > 0.85, "fox " + fox);
        assertTrue(fox > plain + 0.1, "fox " + fox + " vs no-position attention " + plain);
    }
}
