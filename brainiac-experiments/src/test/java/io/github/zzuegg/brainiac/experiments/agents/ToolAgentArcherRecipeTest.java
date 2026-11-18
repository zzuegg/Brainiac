package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.17: the baselines and the critic warm-up and double-Q changes of ArCHer's recipe. */
class ToolAgentArcherRecipeTest {

    static String describe(Map<String, ToolAgentBench.ArcherRow[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, rows) -> {
            sb.append(' ').append(k).append(" [");
            for (ToolAgentBench.ArcherRow row : rows) {
                sb.append(String.format(" %d:%.2f/%.2f", row.row().interactions(), row.row().success(),
                        row.row().sampled()));
            }
            ToolAgentBench.ArcherRow last = rows[rows.length - 1];
            sb.append(String.format(" ] overest %.3f td %.3f;", last.overestimation(), last.tdError()));
        });
        return sb.toString();
    }

    @Test
    void archerRecipeBaselinesAndCriticChanges() {
        Map<String, ToolAgentBench.ArcherRow[]> r = ToolAgentBench.archerRecipe(6000, 2, 1, "trajectory GRPO",
                "ArCHer (wave 8)", "ArCHer + critic warm-up", "ArCHer + double Q");
        System.out.printf("9.2.17 greedy/sampled success by interactions (wave-8 ArCHer is the baseline):%s%n",
                describe(r));
        assertEquals(4, r.size());
    }
}
