package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ParallelHybridTest {

    @Test
    void parallelHybridMatchesTheSequentialHybrid() {
        List<HybridAndCanonShootout.HybridRow> rows = HybridAndCanonShootout.hybrids(250, 0, 0, 1);
        HybridAndCanonShootout.HybridRow hymba = rows.get(0);
        HybridAndCanonShootout.HybridRow kimi = rows.get(1);
        System.out.printf("associative recall: Hymba x2 (%d params) %.3f vs Kimi 3:1 x4 (%d params) %.3f%n",
                hymba.parameters(), hymba.recall(), kimi.parameters(), kimi.recall());
        assertTrue(hymba.recall() > 0.9);
        assertTrue(hymba.recall() > kimi.recall() - 0.05);
    }
}
