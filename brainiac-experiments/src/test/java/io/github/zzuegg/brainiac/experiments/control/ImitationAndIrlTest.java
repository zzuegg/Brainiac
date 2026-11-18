package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ImitationAndIrlTest {

    @Test
    void interactiveImitation() {
        Map<String, double[]> r = ImitationAndIrl.imitation(40, 4, 3);
        System.out.printf("12.6.4 CartPole return/labels/interventions (40 labels): %s%n",
                TabularExplorationTest.table(r));
        assertTrue(r.get("expert")[0] > 150, "the scripted expert balances");
        assertTrue(r.get("DAGGER")[0] >= r.get("BEHAVIOR_CLONING")[0] * 0.8, "DAgger is not far below BC");
    }

    @Test
    void inverseRl() {
        Map<String, double[]> r = ImitationAndIrl.irl(8, 40, 3);
        System.out.printf("12.6.5 object world 8x8 EVD train/transfer: %s%n", TabularExplorationTest.table(r));
        assertTrue(r.values().stream().allMatch(v -> v[0] >= -1e-6));
        assertTrue(r.get("DEEP_MAXENT_IRL")[1] <= r.get("behaviour cloning")[1] + 1e-6,
                "a reward transfers at least as well as cloned actions");
    }
}
