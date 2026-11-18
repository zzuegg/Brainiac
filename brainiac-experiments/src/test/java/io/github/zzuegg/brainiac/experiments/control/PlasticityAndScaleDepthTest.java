package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PlasticityAndScaleDepthTest {

    @Test
    void depthStudyReportsGradientSpread() {
        Map<String, PlasticityAndScale.DepthOutcome> r = PlasticityAndScale.depth(PlasticityAndScale.Algorithm.DQN,
                new int[] {8}, 150, 1, 16);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s return %.0f spread %.1f, ", k, v.ret(), v.gradientSpread())));
        System.out.printf("W10 4.11 cart-pole DQN 150 steps: %s%n", b);
        double plain = r.get("PLAIN/Adam @ depth 8").gradientSpread();
        double skip = r.get("MULTI_SKIP/Adam @ depth 8").gradientSpread();
        assertTrue(Double.isFinite(plain) && Double.isFinite(skip), b.toString());
    }
}
