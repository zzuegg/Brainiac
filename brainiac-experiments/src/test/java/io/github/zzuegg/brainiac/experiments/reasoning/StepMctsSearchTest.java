package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class StepMctsSearchTest {

    @Test
    void mctsOverStepsMatchesOrBeatsTreeSearchAtSimilarPrmCalls() {
        StepMctsSearch.Result r = StepMctsSearch.run(new ChainTask(4), 40, 10, 20, 1);
        System.out.printf("step search accuracy: MCTS %.3f (%.1f PRM calls) vs ToT depth-first %.3f (<= %d calls), "
                + "PRM best-of-%d %.3f, one sample %.3f%n", r.mcts(), r.mctsCalls(), r.depthFirst(),
                r.depthFirstCalls(), r.bestOfNCalls(), r.bestOfN(), r.single());
        assertTrue(r.mcts() > r.single() + 0.1, r.toString());
        assertTrue(r.mcts() > r.depthFirst(), r.toString());
        assertTrue(r.mcts() >= r.bestOfN() - 0.05, r.toString());
        assertTrue(r.mctsCalls() <= r.depthFirstCalls() + 2, r.toString());
    }
}
