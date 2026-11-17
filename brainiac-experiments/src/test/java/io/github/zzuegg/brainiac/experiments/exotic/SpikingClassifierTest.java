package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.SurrogateSpike;
import io.github.zzuegg.brainiac.experimental.data.SpikeEncoding;
import io.github.zzuegg.brainiac.experimental.nn.LifLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SpikingClassifierTest {

    @Test
    void surrogateGradientIsTheFastSigmoidDerivative() {
        Var v = Var.param(Tensor.of(Shape.of(3), -0.5f, 0.2f, 1f));
        Var s = SurrogateSpike.spike(v, 2f);
        assertArrayEquals(new float[] {0, 1, 1}, s.value().toArray());
        s.sum().backward();
        assertArrayEquals(new float[] {1 / 4f, 1 / 1.96f, 1 / 9f}, v.grad().toArray(), 1e-5f);
    }

    @Test
    void lifNeuronIntegratesLeaksAndResets() {
        LifLayer lif = LifLayer.of(1, 1, 0.5f, 1f, 1);
        lif.parameters().get(0).assign(Tensor.of(Shape.of(1, 1), 0.6f));
        // constant input 1 -> current 0.6: u = 0.6, 0.9, 1.05 (spike), 0.525 - ... soft reset
        Var out = lif.forward(Var.of(Tensor.ones(Shape.of(1, 4, 1))));
        assertArrayEquals(new float[] {0, 0, 1, 0}, out.value().toArray());
        assertEquals(0.5f * (1.05f) + 0.6f - 1f, lif.lastPotential().item(), 1e-5f);
    }

    @Test
    void encodingsProduceTheRightSpikes() {
        Tensor lat = SpikeEncoding.latency(new float[][] {{1f, 0.5f, 0f}}, 5);
        float[] l = lat.toArray();
        assertEquals(1, l[0]);              // value 1 at step 0
        assertEquals(1, l[2 * 3 + 1]);      // value 0.5 at step 2
        assertEquals(2, lat.sum().item());  // value 0 never
        float rate = SpikeEncoding.rate(new float[][] {{0.3f}}, 4000, new SplittableRandom(1)).mean().item();
        assertEquals(0.3, rate, 0.03);
    }

    @Test
    void spikingNetworkLearnsPatternsWithRateAndLatencyCoding() {
        for (boolean latency : new boolean[] {false, true}) {
            SpikingClassifier net = SpikingClassifier.of(latency, 3);
            for (int s = 0; s < 150; s++) {
                net.trainStep(32);
            }
            double acc = net.accuracy(200, 7);
            assertTrue(acc > 0.9, (latency ? "latency" : "rate") + " coding accuracy " + acc);
            assertTrue(net.hiddenSpikeRate() < 0.5, "sparse activity " + net.hiddenSpikeRate());
        }
    }
}
