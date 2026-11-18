package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experiments.control.GoalsImitationAndTeams.TeamScore;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ValueDecompositionNavigationTest {

    private static String format(Map<String, TeamScore> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s %.1f (coverage %.2f, %.1f s); ", k, v.teamReturn(),
                v.coverage(), v.seconds())));
        return sb.toString();
    }

    @Test
    void valueDecompositionBaselinesOnNavigation() {
        Map<String, TeamScore> r = GoalsImitationAndTeams.navigation(2, 6_000, 1, "random", "independent DQN", "VDN",
                "QMIX");
        System.out.printf("cooperative navigation, 2 agents, 6k joint steps, value-based: %s%n", format(r));
        assertTrue(r.get("VDN").teamReturn() > r.get("random").teamReturn(), "VDN learns");
    }
}
