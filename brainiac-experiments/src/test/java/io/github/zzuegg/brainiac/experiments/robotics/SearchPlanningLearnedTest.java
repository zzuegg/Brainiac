package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SearchPlanningLearnedTest {

    private static String line(String head, List<SearchPlanning.LearnedRow> rows) {
        StringBuilder line = new StringBuilder(head);
        for (SearchPlanning.LearnedRow r : rows) {
            line.append(String.format(" %s %.2f/%.3f/%.1f;", r.method(), r.success(), r.costRatio(), r.expanded()));
        }
        return line.toString();
    }

    @Test
    void valueIterationNetwork() {
        List<SearchPlanning.LearnedRow> rows = SearchPlanning.learnedPlanners(8, 12, 60, 20, 200, 1213);
        String line = line("12.3.13 VIN success/cost-ratio:", rows);
        System.out.println(line);
        assertEquals(1.0, rows.get(0).success(), 1e-9, line);
        assertTrue(rows.get(1).success() > 0.2, line);
    }

    @Test
    void neuralAStar() {
        List<SearchPlanning.LearnedRow> rows = SearchPlanning.neuralAStar(12, 40, 20, 40, 1214);
        String line = line("12.3.13 Neural A* success/cost-ratio/expanded:", rows);
        System.out.println(line);
        assertTrue(rows.get(2).success() > 0.9, line);
        assertTrue(rows.get(2).expanded() < rows.get(0).expanded(), line);
    }
}
