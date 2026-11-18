package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableModels.TreeRow;

import java.util.List;

import org.junit.jupiter.api.Test;

class OptimalTreesTest {

    private static void print(String table, TreeRow r) {
        System.out.printf("%s %-15s accuracy %.3f truth agreement %.3f objective %.4f leaves %2d binary %3d "
                + "subproblems %7d time %7.1f ms%n", table, r.method(), r.accuracy(), r.truth(), r.objective(),
                r.leaves(), r.binary(), r.subproblems(), r.millis());
    }

    @Test
    void dynamicProgrammingFindsTheXorTreeGreedyMisses() {
        List<TreeRow> rows = InterpretableModels.optimalTrees(1000, 1);
        rows.forEach(r -> print("Trees", r));
        assertTrue(rows.get(0).objective() <= rows.get(1).objective() + 1e-12, "the optimum is no worse than greedy");
        assertTrue(rows.get(0).truth() > rows.get(1).truth() + 0.1, "optimal recovers the XOR root greedy misses");
    }

    @Test
    void guessedThresholdsMatchFineQuantilesWithFewerFeatures() {
        List<TreeRow> rows = InterpretableModels.thresholds(1000, 1);
        rows.forEach(r -> print("Thresholds", r));
        TreeRow coarse = rows.get(0);
        TreeRow fine = rows.get(1);
        TreeRow guessed = rows.get(2);
        assertTrue(guessed.truth() >= coarse.truth(), "guessed thresholds are at least as good as coarse quantiles");
        assertTrue(guessed.binary() < fine.binary(), "guessing searches fewer binary features");
    }
}
