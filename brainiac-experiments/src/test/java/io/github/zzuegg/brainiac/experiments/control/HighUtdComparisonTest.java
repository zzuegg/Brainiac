package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class HighUtdComparisonTest {

    @Test
    void highUpdateToDataLearnsFasterPerEnvironmentStep() {
        Map<String, HighUtdComparison.Outcome> r = HighUtdComparison.run(1000, 100, 5, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.0f (bias %+.2f), ", k, v.ret(), v.bias())));
        System.out.printf("W6 3.10 pendulum 1000 steps return (Q bias vs MC): %s%n", b);
        double sac1 = r.get("SAC UTD 1").ret();
        assertTrue(r.get("REDQ UTD 5").ret() > sac1, r.toString());
        assertTrue(r.get("DroQ UTD 5").ret() > sac1, r.toString());
    }
}
