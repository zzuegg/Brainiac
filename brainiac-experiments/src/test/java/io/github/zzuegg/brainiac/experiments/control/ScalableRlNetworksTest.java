package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.RunningObservationNorm;
import io.github.zzuegg.brainiac.experimental.rl.HlGaussValue;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Map;

class ScalableRlNetworksTest {

    @Test
    void hlGaussTargetsAreSmoothedOneHots() {
        HlGaussValue head = HlGaussValue.of(-1, 1, 20);
        float[] p = head.target(Tensor.of(Shape.of(1, 1), new float[] {0.33f})).toArray();
        double sum = 0;
        double mean = 0;
        for (int i = 0; i < p.length; i++) {
            sum += p[i];
            mean += p[i] * (-1 + (i + 0.5) * 0.1);
        }
        assertEquals(1, sum, 1e-5);
        assertEquals(0.33, mean, 0.01, "the target's mean is the scalar");
        // logits equal to the log target decode to the scalar
        float[] logits = new float[p.length];
        for (int i = 0; i < p.length; i++) {
            logits[i] = (float) Math.log(Math.max(1e-30, p[i]));
        }
        assertEquals(0.33, head.value(Var.of(Tensor.of(Shape.of(1, 20), logits))).value().item(), 0.01);
    }

    @Test
    void runningNormStandardizesInputs() {
        RunningObservationNorm norm = new RunningObservationNorm(2);
        float[] x = new float[200];
        for (int i = 0; i < 100; i++) {
            x[2 * i] = 10 + (i % 2 == 0 ? 3 : -3);
            x[2 * i + 1] = -5 + (i % 2 == 0 ? 0.5f : -0.5f);
        }
        Var out = norm.forward(Var.of(Tensor.of(Shape.of(100, 2), x)), true);
        float[] o = out.value().toArray();
        assertEquals(1, Math.abs(o[0]), 1e-3);
        assertEquals(1, Math.abs(o[1]), 1e-3);
        assertEquals(100, norm.count(), 0);
    }

    @Test
    void simbaWithValueClassificationOnThePendulum() {
        Map<String, ScalableRlNetworks.Outcome> r = ScalableRlNetworks.run(1500, 50, 32, false, 1);
        System.out.printf("pendulum SAC, 1500 steps: random %.1f, MLP+MSE %.1f (%d critic params), "
                + "SimBa+HL-Gauss %.1f (%d critic params)%n", r.get("random").meanReturn(),
                r.get("MLP + MSE").meanReturn(), r.get("MLP + MSE").criticParameters(),
                r.get("SimBa + HL-Gauss").meanReturn(), r.get("SimBa + HL-Gauss").criticParameters());
        assertTrue(r.get("SimBa + HL-Gauss").meanReturn() > r.get("MLP + MSE").meanReturn(), r.toString());
    }
}
