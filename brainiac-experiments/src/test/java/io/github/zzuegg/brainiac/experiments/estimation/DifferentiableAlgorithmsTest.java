package io.github.zzuegg.brainiac.experiments.estimation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class DifferentiableAlgorithmsTest {

    @Test
    void relaxedSortingLearnsRankingsFromOrderAlone() {
        Map<String, DifferentiableAlgorithms.SortRow> r = DifferentiableAlgorithms.sorting(5, 300, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.2f/%.2f %.1fms; ", k, v.exact(), v.elementwise(),
                v.msPerStep())));
        System.out.printf("sort n=5 exact/element acc: %s%n", sb);
        for (String m : new String[] {"NEURAL_SORT", "SOFT_SORT", "FAST_SOFT_RANK", "MONOTONIC_ODD_EVEN"}) {
            assertTrue(r.get(m).exact() > 0.5, sb.toString());
        }
        assertTrue(r.get("REGRESSION_ORACLE").exact() > 0.5, sb.toString());
    }

    @Test
    void solverInTheLoopAndDecisionFocusedLosses() {
        Map<String, DifferentiableAlgorithms.PathRow> w = DifferentiableAlgorithms.warcraftPaths(8, 150, 2);
        StringBuilder sb = new StringBuilder();
        w.forEach((k, v) -> sb.append(String.format("%s %.2f (regret %.3f); ", k, v.optimalRate(), v.regret())));
        System.out.printf("Warcraft-lite 8x8 optimal-path rate: %s%n", sb);
        assertTrue(w.get("PERTURBED_FY").optimalRate() > w.get("DIRECT_PATH").optimalRate() + 0.2, sb.toString());
        assertTrue(w.get("BLACKBOX").optimalRate() > w.get("DIRECT_PATH").optimalRate(), sb.toString());

        Map<String, Double> d = DifferentiableAlgorithms.decisions(new int[] {1, 6}, 200, 120, 3);
        StringBuilder sd = new StringBuilder();
        d.forEach((k, v) -> sd.append(String.format("%s %.3f; ", k, v)));
        System.out.printf("normalised regret: %s%n", sd);
        assertTrue(d.get("path d=6 / SPO_PLUS") < d.get("path d=6 / TWO_STAGE_MSE"), sd.toString());
        assertTrue(d.get("path d=6 / PG_CENTRAL") < d.get("path d=6 / TWO_STAGE_MSE"), sd.toString());
    }

    @Test
    void optNetLearnsAPolytopeAndSudokuConstraints() {
        Map<String, double[]> p = DifferentiableAlgorithms.polytopeProjection(150, 4);
        Map<String, double[]> s = DifferentiableAlgorithms.sudoku(200, 60, 6, 5);
        System.out.printf("polytope projection MSE/outside: OptNet %.4f/%.2f, MLP %.4f/%.2f; 4x4 sudoku cell/solved/s:"
                + " OptNet %.2f/%.2f/%.1f, SATNet %.2f/%.2f/%.1f, MLP %.2f/%.2f/%.1f%n",
                p.get("OptNet")[0], p.get("OptNet")[1], p.get("MLP")[0], p.get("MLP")[1],
                s.get("OptNet")[0], s.get("OptNet")[1], s.get("OptNet")[2], s.get("SATNet")[0], s.get("SATNet")[1],
                s.get("SATNet")[2], s.get("MLP")[0], s.get("MLP")[1], s.get("MLP")[2]);
        assertTrue(p.get("OptNet")[0] < p.get("MLP")[0], "OptNet projects better than the MLP");
        assertTrue(s.get("OptNet")[0] > 0.25, "OptNet runs and beats chance on blank cells");
    }
}
