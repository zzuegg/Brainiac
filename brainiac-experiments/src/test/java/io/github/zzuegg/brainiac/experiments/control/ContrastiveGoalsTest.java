package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ContrastiveGoalsTest {

    @Test
    void contrastiveRlReachesGoalsWithoutRewards() {
        Map<String, Double> r = ContrastiveGoals.run(6, 2000, new int[] {4, 16}, 1);
        System.out.printf("bit flipping (6 bits, 2000 steps) success: HER+DQN %.2f, contrastive RL depth 4 %.2f, "
                + "depth 16 %.2f%n", r.get("HER + DQN"), r.get("contrastive RL, depth 4"),
                r.get("contrastive RL, depth 16"));
        assertTrue(r.get("contrastive RL, depth 4") > 0.5, r.toString());
        assertTrue(r.get("contrastive RL, depth 16") > 0.5, "deep residual encoders train too: " + r);
    }
}
