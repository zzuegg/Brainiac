package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.17: ArCHer's critic warm-up, double Q, slow actor and entropy, ablated. */
class ToolAgentArcherRecipeTest {

    @Test
    void archerRecipeAblation() {
        Map<String, ToolAgentBench.ArcherRow[]> r = ToolAgentBench.archerRecipe(6000, 3, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, rows) -> {
            ToolAgentBench.ArcherRow last = rows[rows.length - 1];
            sb.append(String.format(" %s %.2f/%.2f overest %.3f td %.3f;", k, last.row().success(),
                    last.row().sampled(), last.overestimation(), last.tdError()));
        });
        System.out.printf("9.2.17 greedy/sampled success at 6000 interactions (wave-8 ArCHer row is the baseline):%s%n",
                sb);
        assertEquals(6, r.size());
    }
}
