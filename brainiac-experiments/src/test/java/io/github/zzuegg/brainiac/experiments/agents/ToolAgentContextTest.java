package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 8, 8.4.15: Sculptor's context tools on needles whose key is overwritten later. */
class ToolAgentContextTest {

    @Test
    void sculptorHidesSupersededValues() {
        int[] lengths = {16, 64};
        Map<String, ToolAgentBench.ContextCell[]> r = ToolAgentBench.context(lengths, 3, 60, 450, 200, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, cells) -> {
            sb.append(' ').append(k).append(" [");
            for (int i = 0; i < cells.length; i++) {
                sb.append(String.format("%s%d:%.2f/%.0ftok", i == 0 ? "" : " ", lengths[i], cells[i].accuracy(),
                        cells[i].tokens()));
            }
            sb.append("];");
        });
        System.out.printf("8.4.15 overwritten-key accuracy/tokens by length:%s%n", sb);
        int last = lengths.length - 1;
        double sculptor = r.get("Sculptor, hide superseded")[last].accuracy();
        assertTrue(sculptor > r.get("MemGPT pager retrieval")[last].accuracy() + 0.1, "beats retrieval");
        assertTrue(sculptor > r.get("truncation")[last].accuracy() + 0.2, "beats truncation");
        assertTrue(r.get("Sculptor, hide superseded")[last].tokens() < r.get("RLM")[last].tokens() / 4,
                "reads far fewer tokens than RLM");
    }
}
