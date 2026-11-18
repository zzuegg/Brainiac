package io.github.zzuegg.brainiac.experiments.context;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.15: MemAgent and recursive LMs on needles 4 to 32 times the window. */
class LongContextNeedleTest {

    @Test
    void memoryAgentsAndRecursionReadPastTheWindow() {
        int[] lengths = {16, 32, 64, 128};
        Map<String, LongContextNeedle.Cell[]> r = LongContextNeedle.run(lengths, 60, 100, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, cells) -> {
            sb.append(' ').append(k).append(" [");
            for (int i = 0; i < cells.length; i++) {
                sb.append(String.format("%s%dx:%.2f", i == 0 ? "" : " ", lengths[i] / LongContextNeedle.WINDOW,
                        cells[i].accuracy()));
            }
            sb.append(String.format("] %.0f calls at 32x;", cells[cells.length - 1].calls()));
        });
        System.out.printf("needle accuracy by context/window:%s%n", sb);
        int last = lengths.length - 1;
        assertTrue(r.get("RLM")[last].accuracy() > 0.8, "RLM keeps finding the needle at 32x");
        assertTrue(r.get("MemAgent RL")[last].accuracy() > r.get("MemAgent SFT only")[last].accuracy() + 0.2,
                "RL teaches the memory to keep the needle");
        assertTrue(r.get("MemAgent RL")[last].accuracy() > r.get("truncation")[last].accuracy() + 0.3,
                "MemAgent beats truncation");
    }
}
