package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SearchPlanningTest {

    @Test
    void gridSearch() {
        List<SearchPlanning.GridRow> rows = SearchPlanning.grids(64, 4, 1211);
        StringBuilder line = new StringBuilder("12.3.11 expanded/cost-ratio:");
        for (SearchPlanning.GridRow r : rows) {
            line.append(String.format(" %s %s %.0f/%.3f;", r.map(), r.method(), r.expanded(), r.costRatio()));
        }
        System.out.println(line);
        for (int f = 0; f < 3; f++) {
            SearchPlanning.GridRow dijkstra = rows.get(6 * f + 1);
            SearchPlanning.GridRow astar = rows.get(6 * f + 2);
            SearchPlanning.GridRow jps = rows.get(6 * f + 5);
            assertEquals(1.0, astar.costRatio(), 1e-9, line.toString());
            assertEquals(1.0, jps.costRatio(), 1e-9, line.toString());
            assertTrue(astar.expanded() <= dijkstra.expanded(), line.toString());
        }
    }

    @Test
    void puzzles() {
        List<SearchPlanning.PuzzleRow> rows = SearchPlanning.puzzles(12, 300, 1212);
        StringBuilder line = new StringBuilder("12.3.12 optimal/expanded/inadmissible:");
        for (SearchPlanning.PuzzleRow r : rows) {
            line.append(String.format(" %s %s %.2f/%.0f/%.3f;", r.puzzle(), r.method(), r.optimalRate(),
                r.expanded(), r.inadmissible()));
        }
        System.out.println(line);
        assertEquals(1.0, rows.get(0).optimalRate(), 1e-9, line.toString());
        assertTrue(rows.get(4).optimalRate() >= 0.5, line.toString());
    }
}
