package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.NInARow;
import org.junit.jupiter.api.Test;

import java.util.Map;

class ControlExperimentsTest {

    @Test
    void continuousControlComparisonRunsEveryMethod() {
        Map<String, Double> r = ContinuousControlComparison.run(400, 50, 1);
        System.out.println("continuous control (tiny budget): " + r);
        assertEquals(6, r.size());
        r.values().forEach(v -> assertTrue(Double.isFinite(v)));
    }

    @Test
    void rainbowAblationLearnsTheMaze() {
        Map<String, RainbowAblation.Outcome> r = RainbowAblation.run(2500, 2);
        System.out.println("rainbow ablation: " + r);
        assertTrue(r.get("Rainbow").greedyReturn() > 0.5, r.toString());
        assertTrue(r.get("C51").greedyReturn() > 0.5, r.toString());
    }

    @Test
    void alphaZeroSearchBeatsRandomOnConnectThree() {
        Map<String, Double> r = SelfPlayArena.run(NInARow.connectThree(), 15, 20, 24, 20, 1);
        System.out.println("self-play arena: " + r);
        assertTrue(r.get("AlphaZero+MCTS vs random") >= 0.75, r.toString());
    }

    @Test
    void offlineRlImprovesOnMixedData() {
        Map<String, Double> r = OfflineRlComparison.run(100, 25, 1000, 1);
        System.out.println("offline RL: " + r);
        assertTrue(r.get("IQL") > r.get("data") && r.get("CQL") > r.get("data"), r.toString());
    }

    @Test
    void teamsGoalsAndImitation() {
        Map<String, Double> teams = GoalsImitationAndTeams.teams(2400, 3);
        System.out.println("two-step game: " + teams);
        assertEquals(8, teams.get("QMIX"), 1e-6);
        Map<String, Double> her = GoalsImitationAndTeams.hindsight(new int[] {5}, 2500, 1);
        System.out.println("bit flipping: " + her);
        assertTrue(her.get("5 bits, HER") >= her.get("5 bits, plain"));
        Map<String, Double> imitation = GoalsImitationAndTeams.imitation(5, 6000, 1);
        System.out.println("imitation from 5 demos: " + imitation);
        assertTrue(imitation.get("MaxEnt IRL") > imitation.get("random"));
    }
}
