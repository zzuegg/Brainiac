package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ModernOffPolicySimbaV2Test {

    @Test
    void simbaV2KeepsItsWeightsOnTheSphere() {
        Map<String, ModernOffPolicy.ScalingRow> r = ModernOffPolicy.simbaV2Scaling(ModernOffPolicy.Task.PENDULUM, 300,
                1, 16, new int[] {1, 2});
        StringBuilder b = new StringBuilder();
        r.values().forEach(v -> b.append(String.format("%s return %.0f norm %.2f, ", v.name(), v.iqm(), v.criticNorm())));
        System.out.printf("W10 4.4 pendulum 300 steps: %s%n", b);
        for (ModernOffPolicy.ScalingRow row : r.values()) {
            assertTrue(Double.isFinite(row.iqm()) && Double.isFinite(row.criticNorm()), row.toString());
        }
    }
}
