package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experiments.control.GoalsImitationAndTeams.TeamScore;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CooperativeMarlTest {

    private static String format(Map<String, TeamScore> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s %.1f (coverage %.2f, %.1f s); ", k, v.teamReturn(),
                v.coverage(), v.seconds())));
        return sb.toString();
    }

    @Test
    void centralisedCriticsAndPpoLearnCooperativeNavigation() {
        Map<String, TeamScore> r = GoalsImitationAndTeams.navigation(2, 12_000, 1, "random", "IPPO", "MAPPO", "COMA",
                "MADDPG");
        System.out.printf("cooperative navigation, 2 agents, 12k joint steps, team return: %s%n", format(r));
        double random = r.get("random").teamReturn();
        assertTrue(r.get("MAPPO").teamReturn() > random, "MAPPO learns");
        assertTrue(r.get("IPPO").teamReturn() > random, "IPPO learns");
        assertTrue(r.get("MADDPG").teamReturn() > random, "MADDPG learns");
    }

    @Test
    void valueDecompositionBaselinesOnNavigation() {
        Map<String, TeamScore> r = GoalsImitationAndTeams.navigation(2, 8_000, 1, "random", "independent DQN", "VDN",
                "QMIX");
        System.out.printf("cooperative navigation, 2 agents, 8k joint steps, value-based: %s%n", format(r));
        assertTrue(r.get("VDN").teamReturn() > r.get("random").teamReturn(), "VDN learns");
    }

    @Test
    void learnedChannelsCarryTheGoal() {
        Map<String, double[]> r = GoalsImitationAndTeams.communication(5, 8, 150, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.2f/%.2f; ", k, v[0], v[1])));
        System.out.printf("hidden goal, 5 agents, 8 goals, 150 x 64 episodes (reward/success): %s%n", sb);
        assertTrue(r.get("CommNet")[0] > r.get("no communication")[0] + 0.3, "CommNet communicates");
        assertTrue(r.get("TarMAC")[0] > r.get("no communication")[0] + 0.3, "TarMAC communicates");
        assertTrue(r.get("TarMAC attention")[0] > 1.0 / 5, "TarMAC attends to the informed agent");
    }
}
