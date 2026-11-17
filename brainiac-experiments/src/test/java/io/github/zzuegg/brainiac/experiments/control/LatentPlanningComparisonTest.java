package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class LatentPlanningComparisonTest {

    @Test
    void latentPlanningSwingsThePendulumUp() {
        Map<String, LatentPlanningComparison.Outcome> r = LatentPlanningComparison.run(20, 300, 10, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.0f (10-step error %.3f), ", k, v.ret(), v.error())));
        System.out.printf("W6 3.7 pendulum from 2000 random steps, return: %s%n", b);
        LatentPlanningComparison.Outcome planet = r.get("PlaNet (RSSM + overshooting)");
        assertTrue(planet.ret() > -600, b.toString());
        assertTrue(planet.error() < r.get("MLP dynamics + CEM (true reward)").error(), b.toString());
    }
}
