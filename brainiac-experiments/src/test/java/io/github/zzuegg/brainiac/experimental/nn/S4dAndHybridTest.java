package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class S4dAndHybridTest {

    @Test
    void s4dConvolutionModeEqualsRecurrentMode() {
        S4dLayer s4 = new S4dLayer(4, 8, new SplittableRandom(1));
        LinearRecurrenceTest.assertModesAgree(s4.parameters(), s4::forward, s4::recurrent,
                LinearRecurrenceTest.input(2, 2, 12, 4), 1e-4f);
    }

    @Test
    void s4dKernelIsTheImpulseResponse() {
        S4dLayer s4 = new S4dLayer(2, 4, new SplittableRandom(3));
        Var k = s4.kernel(5);
        assertEquals(Shape.of(2, 5), k.shape());
        // an impulse in, the kernel out (through the same convolution)
        Tensor impulse = Tensor.of(Shape.of(1, 2, 5), 1, 0, 0, 0, 0, 1, 0, 0, 0, 0);
        float[] y = MixerOps.causalConvolution(Var.of(impulse), k).value().toArray();
        assertArrayEquals(k.value().toArray(), y, 1e-5f);
    }

    @Test
    void s4dForecastsSines() {
        double mse = SineForecast.train(new S4dLayer(16, 16, new SplittableRandom(5)), 16, 16, 300, 0.01f, 6);
        assertTrue(mse < 0.02, "mse " + mse);
    }

    @Test
    void hybridStackInterleavesThreeLinearLayersPerAttentionLayerAndLearnsRecall() {
        HybridLinearStack stack = new HybridLinearStack(16, 2, 4, 3, new SplittableRandom(7));
        assertArrayEquals(new boolean[] {false, false, false, true}, stack.fullAttentionLayers());
        assertEquals(0.25, stack.kvCacheFraction(), 1e-9);
        double acc = MixerHarness.train(stack, MixerHarness.recall(6, 6, 3), 16, 120, 0.01f, 8);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }
}
