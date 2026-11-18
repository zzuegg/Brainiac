package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class MetaRlBanditsTest {

    @Test
    void metaLearnedAgentsExploreNewBanditsInContext() {
        Map<String, Double> r = MetaRlBandits.run(20, 450, 100, 1);
        System.out.printf("W6 3.16 one-good-arm 5-armed bandits, mean reward per pull over 20 pulls of a new task:"
                + " RL2 %.2f, VariBAD %.2f vs MAML-PG %.2f, PPO from scratch %.2f, random %.2f%n", r.get("RL2"),
                r.get("VariBAD"), r.get("MAML policy gradient"), r.get("PPO from scratch (core)"), r.get("random"));
        double ppo = r.get("PPO from scratch (core)");
        assertTrue(r.get("RL2") > ppo + 0.2, r.toString());
        assertTrue(r.get("VariBAD") > ppo + 0.2, r.toString());
    }
}
