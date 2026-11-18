package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SecondOrderShootoutTest {

    @Test
    void kroneckerMethodsReachTheTargetNoSlowerThanAdamW() {
        Map<String, SecondOrderShootout.Row> r = SecondOrderShootout.run(150, 64, 0.3, 1, 2, 3);
        StringBuilder line = new StringBuilder("Spirals MLP, 150 steps, 3 seeds, loss/acc/steps-to-0.3/ms-per-step:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.2f/%.0f/%.1f%s;", k, v.loss(), v.accuracy(),
                v.stepsToTarget(), v.millisPerStep(), v.diverged() > 0 ? "/div" + v.diverged() : "")));
        System.out.println(line);
        SecondOrderShootout.Row adam = r.get("AdamW");
        assertTrue(adam.accuracy() > 0.8, line.toString());
        assertTrue(r.get("K-FAC").loss() < adam.loss() * 1.5, line.toString());
        assertTrue(r.get("EKFAC").accuracy() > 0.8, line.toString());
        assertTrue(r.get("AdaFisher").accuracy() > 0.7, line.toString());
        assertTrue(r.get("AdaHessian").accuracy() > 0.7, line.toString());
    }
}
