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
}
