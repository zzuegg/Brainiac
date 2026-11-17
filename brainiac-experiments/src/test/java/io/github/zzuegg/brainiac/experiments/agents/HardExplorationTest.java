package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class HardExplorationTest {

    @Test
    void goExploreCrossesTheDeadlyMazeWhereRandomWalksDie() {
        HardExploration.Result r = HardExploration.maze(1000, 1);
        System.out.println("go-explore maze " + r);
        assertTrue(r.bestReturn() > 0, r.toString());
        assertEquals(0, r.randomSolved(), r.toString());
        assertTrue(r.goExploreCells() > r.randomCells(), r.toString());
    }

    @Test
    void goExploreSolvesMountainCar() {
        HardExploration.Result r = HardExploration.mountainCar(200, 1);
        System.out.println("go-explore mountain car " + r);
        assertTrue(r.goExploreSolved(), r.toString());
        assertTrue(r.bestReturn() > -200, r.toString());
    }
}
