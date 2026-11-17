package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class CuriosityExplorationTest {

    @Test
    void curiosityExploresTheSparseMazeBetter() {
        CuriosityExploration.Result none = CuriosityExploration.run(CuriosityExploration.Bonus.NONE, 6000, 1);
        CuriosityExploration.Result rnd = CuriosityExploration.run(CuriosityExploration.Bonus.RND, 6000, 1);
        CuriosityExploration.Result icm = CuriosityExploration.run(CuriosityExploration.Bonus.ICM, 6000, 1);
        System.out.println("none " + none + " RND " + rnd + " ICM " + icm);
        assertTrue(rnd.distinctCells() > none.distinctCells() + 10, "RND " + rnd + " vs " + none);
        assertTrue(icm.distinctCells() >= none.distinctCells(), "ICM " + icm + " vs " + none);
        assertTrue(rnd.goalsReached() > none.goalsReached(), "RND " + rnd + " vs " + none);
    }
}
