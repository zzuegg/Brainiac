package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.Plasticity;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.SplittableRandom;

class PlasticityAndScaleTest {

    @Test
    void redoRecyclesDeadNeuronsWithoutChangingTheFunction() {
        Model m = Nets.mlp(4, 2, Activation.RELU, 1, 16, 16);
        // kill four neurons of the first layer by large negative biases
        float[] b = m.layers().get(0).parameters().get(1).value().toArray();
        for (int i = 0; i < 4; i++) {
            b[i] = -100;
        }
        m.layers().get(0).parameters().get(1).assign(Tensor.of(Shape.of(16), b));
        Tensor x = Tensor.normal(Shape.of(64, 4), 0, 1, new SplittableRandom(2));
        double before = Plasticity.dormantFraction(m, x, 0.0f);
        float[] out0 = m.predict(x).toArray();
        int recycled = Plasticity.redo(m, x, 0.0f, null, 3);
        float[] out1 = m.predict(x).toArray();
        double after = Plasticity.dormantFraction(m, x, 0.0f);
        double maxDiff = 0;
        for (int i = 0; i < out0.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(out0[i] - out1[i]));
        }
        System.out.printf("W10 4.10 ReDo: dormant share %.3f -> %.3f, %d neurons recycled, output change %.2e%n",
                before, after, recycled, maxDiff);
        assertTrue(before >= 4 / 32.0 && after < before, before + " " + after);
        assertEquals(0, maxDiff, 1e-4);
    }

    @Test
    void interventionsRunAndReportDiagnostics() {
        Map<String, PlasticityAndScale.Outcome> r = PlasticityAndScale.plasticity(() -> new CartPole(200), 500,
                new int[] {2}, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s return %.0f dormant %.2f rank %.0f, ", k, v.ret(), v.dormant(),
                v.featureRank())));
        System.out.printf("W10 4.10 cart-pole 500 steps: %s%n", b);
        assertEquals(PlasticityAndScale.Intervention.values().length, r.size());
        for (PlasticityAndScale.Outcome o : r.values()) {
            assertTrue(Double.isFinite(o.ret()) && o.dormant() >= 0 && o.dormant() <= 1, o.toString());
        }
    }
}
