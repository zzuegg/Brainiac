package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ModernOffPolicyMountainCarTest {

    @Test
    void mountainCarGoal() {
        Map<String, double[]> r = ModernOffPolicy.mountainCarGoal(1_000, 1, 32);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format(Locale.ROOT, "%s=%.2f/%.2f/%.1f ", k, v[0], v[1], v[2])));
        System.out.printf("11.6.17 continuous mountain car greedy-reach/train-reach/return at 1k steps: %s%n",
                sb.toString().trim());
        assertTrue(r.values().stream().allMatch(v -> v[0] >= 0 && v[0] <= 1));
    }
}
