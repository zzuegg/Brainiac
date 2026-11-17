package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DeceptiveMazeTest {

    @Test
    void wallsStopTheRobot() {
        double[] up = new double[2 * DeceptiveMaze.STEPS];
        for (int s = 0; s < DeceptiveMaze.STEPS; s++) {
            up[2 * s + 1] = 1;
        }
        // straight up hits the roof of the cup at y = 5
        assertArrayEquals(new double[] {0, 4}, DeceptiveMaze.simulate(up), 1e-9);
    }

    @Test
    void noveltySearchEscapesTheTrapWhereObjectiveSearchDoesNot() {
        int solvedNovelty = 0;
        int solvedObjective = 0;
        for (long seed = 1; seed <= 3; seed++) {
            DeceptiveMaze.Outcome novelty = DeceptiveMaze.search(1, 200, seed);
            DeceptiveMaze.Outcome objective = DeceptiveMaze.search(0, 200, seed);
            solvedNovelty += novelty.solved() ? 1 : 0;
            solvedObjective += objective.solved() ? 1 : 0;
        }
        assertTrue(solvedNovelty >= 2 && solvedNovelty > solvedObjective,
                "novelty solved " + solvedNovelty + "/3, objective " + solvedObjective + "/3");
    }
}
