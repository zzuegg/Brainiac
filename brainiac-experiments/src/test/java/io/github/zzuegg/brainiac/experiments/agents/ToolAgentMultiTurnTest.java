package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 8, 8.4.18: ArCHer's turn-level critic against trajectory-level multi-turn training. */
class ToolAgentMultiTurnTest {

    @Test
    void multiTurnTrainersAgainstInteractions() {
        Map<String, ToolAgentBench.TurnRow[]> r = ToolAgentBench.multiTurn(6000, 3, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, rows) -> {
            sb.append(' ').append(k).append(" [");
            for (ToolAgentBench.TurnRow row : rows) {
                sb.append(String.format(" %d:%.2f/%.2f/%.1ft", row.interactions(), row.success(), row.sampled(),
                        row.turns()));
            }
            sb.append("];");
        });
        System.out.printf("8.4.18 greedy/sampled success and turns by interactions (random player 0.68):%s%n", sb);
        ToolAgentBench.TurnRow[] grpo = r.get("trajectory GRPO");
        ToolAgentBench.TurnRow[] archer = r.get("ArCHer");
        assertTrue(grpo[grpo.length - 1].sampled() > 0.9, "trajectory GRPO learns to stop repeating guesses");
        assertTrue(archer[archer.length - 1].sampled() > 0.4, "ArCHer keeps a working policy");
    }
}
