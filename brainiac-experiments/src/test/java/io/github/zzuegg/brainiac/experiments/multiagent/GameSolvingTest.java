package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.experimental.games.CommitmentAndCorrelation;
import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GameSolvingTest {

    private static String table(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append('=');
            for (int i = 0; i < v.length; i++) {
                sb.append(String.format(Locale.ROOT, i == 0 ? "%.4g" : "/%.4g", v[i]));
            }
            sb.append(' ');
        });
        return sb.toString().trim();
    }

    @Test
    void searchAndLearnedEvaluation() {
        Map<String, double[]> nodes = GameSolving.searchNodes(5, 5, 4, 5);
        Map<String, double[]> mcts = GameSolving.alphaBetaVsMcts(5, 5, 4, 3, 200, 6);
        Map<String, double[]> td = GameSolving.tdLeaf(5, 5, 4, 30, 2, 6, 1);
        Map<String, Double> dice = GameSolving.expectimax(30, 2, 200, 1);
        System.out.printf("12.6.12 5x5 connect-4 nodes/value at depth 5: %s | W/D/L: %s | TD-Leaf vs hand-written W/D/L: %s"
                + " | dice race expectimax win rate: %s%n", table(nodes), table(mcts), table(td), dice);
        assertEquals(nodes.get("NEGAMAX")[1], nodes.get("ALPHA_BETA_PLAIN")[1], 1e-9, "alpha-beta keeps the value");
        assertTrue(nodes.get("ALPHA_BETA_PLAIN")[0] < nodes.get("NEGAMAX")[0], "pruning visits fewer nodes");
        assertTrue(mcts.get("alpha-beta vs random")[0] >= 5, "alpha-beta beats random play");
        assertTrue(dice.get("random") > 0.5, "expectimax beats random play in the dice race");
    }

    @Test
    void commitmentAndCorrelation() {
        Map<String, double[]> c = GameSolving.commitment(4, 1);
        Map<String, double[]> dyn = GameSolving.dynamics(CommitmentAndCorrelation.shapley(), 5_000);
        System.out.printf("12.6.13 Stackelberg/best-Nash leader value, CE/best-Nash welfare: %s | Shapley game CE/CCE gap"
                + " after 5k rounds: %s%n", table(c), table(dyn));
        for (String g : new String[] {"chicken", "battle of the sexes", "shapley"}) {
            double[] v = c.entrySet().stream().filter(e -> e.getKey().equalsIgnoreCase(g)).findFirst()
                    .map(Map.Entry::getValue).orElse(c.values().iterator().next());
            assertTrue(v[0] >= v[1] - 1e-6, "commitment never hurts the leader");
            assertTrue(v[2] >= v[3] - 1e-6, "the best CE has at least the best Nash welfare");
        }
        double[] sec = c.get("security game");
        assertTrue(sec[0] >= sec[1] - 1e-6, "SSE coverage beats uniform coverage");
        assertTrue(dyn.get("INTERNAL_REGRET_MATCHING")[0] < 0.1, "internal regret matching nears a CE");
    }
}
