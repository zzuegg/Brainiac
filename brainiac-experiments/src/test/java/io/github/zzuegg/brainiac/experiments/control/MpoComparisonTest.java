package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class MpoComparisonTest {

    @Test
    void mpoImprovesThePendulumPolicy() {
        Map<String, MpoComparison.Outcome> r = MpoComparison.run(1500, 100, 2);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.0f +- %.0f, ", k, v.mean(), v.std())));
        System.out.printf("W6 3.11 pendulum 1500 steps, 2 seeds, greedy return: %s%n", b);
        assertTrue(r.get("MPO").mean() > -800, r.toString());
    }
}
