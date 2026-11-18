package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class LearnedCommunicationTest {

    @Test
    void learnedChannelsCarryTheGoal() {
        Map<String, double[]> r = GoalsImitationAndTeams.communication(5, 8, 120, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.2f/%.2f; ", k, v[0], v[1])));
        System.out.printf("hidden goal, 5 agents, 8 goals, 120 x 64 episodes (reward/success): %s%n", sb);
        assertTrue(r.get("CommNet")[0] > r.get("no communication")[0] + 0.3, "CommNet communicates");
        assertTrue(r.get("TarMAC")[0] > r.get("no communication")[0] + 0.3, "TarMAC communicates");
        assertTrue(r.get("TarMAC attention")[0] > 1.0 / 5, "TarMAC attends to the informed agent");
    }
}
