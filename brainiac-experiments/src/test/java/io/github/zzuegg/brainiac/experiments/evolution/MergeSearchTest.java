package io.github.zzuegg.brainiac.experiments.evolution;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MergeSearchTest {

    @Test
    void evolvedMergeRecipeBeatsHandSetMerges() {
        Map<String, Double> r = MergeSearch.run(150, 150, 3);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.3f, ", k, v)));
        System.out.printf("mixed addition + multiplication accuracy, 3 seeds, 150 fitness evaluations: %s%n", sb);
        double handSet = Math.max(Math.max(r.get("task arithmetic (1.0)"), r.get("task arithmetic (0.5)")),
                Math.max(Math.max(r.get("TIES"), r.get("DARE")), r.get("SLERP")));
        assertTrue(r.get("evolved, per tensor") > handSet, "the searched recipe beats every hand-set merge");
        assertTrue(r.get("multi-task") >= r.get("evolved, per tensor"), "multi-task training bounds merging");
    }
}
