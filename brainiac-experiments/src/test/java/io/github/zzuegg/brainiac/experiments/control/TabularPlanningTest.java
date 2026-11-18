package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TabularPlanningTest {

    @Test
    void dynaMazes() {
        Map<String, double[]> r = TabularExploration.dynaMaze(10, 3_000, 3);
        System.out.printf("12.6.2 Dyna maze 3k steps episodes/last-5 length/simulated updates: %s%n",
                TabularExplorationTest.table(r));
        assertTrue(r.get("DYNA_Q")[0] > r.get("Q_LEARNING")[0], "planning completes more episodes");
        Map<String, double[]> b = TabularExploration.changingMaze(false, 1_000, 3_000, 10, 3);
        System.out.printf("12.6.2 blocking maze episodes before/after change: %s%n", TabularExplorationTest.table(b));
        Map<String, double[]> s = TabularExploration.changingMaze(true, 3_000, 6_000, 10, 3);
        System.out.printf("12.6.2 shortcut maze episodes before/after change: %s%n", TabularExplorationTest.table(s));
        assertTrue(b.get("DYNA_Q_PLUS")[1] > b.get("DYNA_Q")[1], "Dyna-Q+ finds the new path after the block");
    }

    @Test
    void options() {
        Map<String, double[]> r = TabularExploration.options(2, 5, 20, 1);
        System.out.printf("12.6.3 four rooms cover/hitting time/options (and learned Laplacian cosines): %s%n",
                TabularExplorationTest.table(r));
        assertTrue(Math.min(r.get("eigenoptions (4 vectors)")[0], r.get("covering options (4 pairs)")[0])
                < r.get("primitive")[0], "options shorten the cover time");
    }
}
