package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReservoirLayersTest {
    @Test
    void layerForwardMatchesStepping() {
        EchoStateReservoir r = EchoStateReservoir.of(2, 40, 0.9, 0.5, 0.5, 0.2, 4);
        float[] in = new float[6 * 2];
        for (int i = 0; i < in.length; i++) {
            in[i] = (float) Math.sin(i);
        }
        float[] states = r.forward(Var.of(Tensor.of(Shape.of(1, 6, 2), in))).value().toArray();
        assertTrue(r.parameters().isEmpty());
        r.reset();
        for (int t = 0; t < 6; t++) {
            double[] h = r.step(new double[] {in[2 * t], in[2 * t + 1]});
            for (int i = 0; i < 40; i++) {
                assertEquals(h[i], states[t * 40 + i], 1e-5);
            }
        }
    }

    @Test
    void echoStatePropertyForgetsTheStart() {
        EchoStateReservoir a = EchoStateReservoir.of(1, 50, 0.8, 0.5, 1, 0.2, 9);
        EchoStateReservoir b = EchoStateReservoir.of(1, 50, 0.8, 0.5, 1, 0.2, 9);
        for (int i = 0; i < 50; i++) {
            b.step(new double[] {5});   // drive b somewhere else first
        }
        double[] ha = null;
        double[] hb = null;
        for (int t = 0; t < 200; t++) {
            double[] u = {Math.sin(t * 0.3)};
            ha = a.step(u);
            hb = b.step(u);
        }
        assertArrayEquals(ha, hb, 1e-6);
    }

    @Test
    void nextGenerationFeaturesAreConstantLinearAndQuadratic() {
        NextGenReservoirFeatures f = NextGenReservoirFeatures.of(2, 2, 3);
        assertEquals(3, f.warmup());
        assertEquals(1 + 4 + 10, f.size());
        double[][] s = {{1, 2}, {0, 0}, {0, 0}, {3, 4}};
        double[] x = f.features(s, 3);
        assertArrayEquals(new double[] {1, 3, 4, 1, 2, 9, 12, 3, 6, 16, 4, 8, 1, 2, 4}, x, 0);
    }
}
