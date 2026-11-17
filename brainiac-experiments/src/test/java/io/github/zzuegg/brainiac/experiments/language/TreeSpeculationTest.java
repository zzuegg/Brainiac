package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TreeSpeculationTest {

    @Test
    void eagleTreeAcceptsMostTokensPerPassAndStaysLossless() {
        int maxNew = 24;
        Map<String, TreeSpeculation.Outcome> r = TreeSpeculation.run(32, 100, 6, maxNew, 1);
        StringBuilder line = new StringBuilder("Tree speculation, tokens per target pass:");
        int total = r.get("greedy").tokens().length;
        r.forEach((k, v) -> line.append(String.format(" %s %.2f;", k, total / (double) v.passes())));
        System.out.println(line);
        for (Map.Entry<String, TreeSpeculation.Outcome> e : r.entrySet()) {
            assertArrayEquals(r.get("greedy").tokens(), e.getValue().tokens(), e.getKey());
        }
        int tree = r.get("eagle tree").passes();
        assertTrue(tree < r.get("eagle chain").passes(), "the tree must beat the chain");
        assertTrue(tree < r.get("core speculative (distilled draft model)").passes());
        assertTrue(tree < r.get("lookahead (Jacobi + n-grams)").passes());
    }
}
