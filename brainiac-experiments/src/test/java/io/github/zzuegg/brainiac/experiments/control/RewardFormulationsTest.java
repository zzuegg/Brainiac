package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RewardFormulationsTest {

    @Test
    void averageReward() {
        Map<String, Double> r = RewardFormulations.averageReward(50_000, 2);
        System.out.printf("11.6.12 access-control reward rate: %s%n", RewardFormulations.line(r));
        assertTrue(r.get("differential Q") > 0.8 * r.get("optimal (RVI)"));
    }

    @Test
    void preferences() {
        Map<String, double[]> r = RewardFormulations.preferences(1_000, 60, 1, 32);
        System.out.printf("11.6.11 point reach true return/success/queries at 1k steps: %s%n",
                RewardFormulations.fmt(r));
        assertTrue(r.get("PEBBLE")[2] <= 60);
    }
}
