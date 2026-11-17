package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class AutoResearcherTest {

    @Test
    void researcherLoopFindsMoreWinningConfigsThanRandomSearch() {
        AutoResearcher.Result r = AutoResearcher.run(24, 1);
        System.out.printf("auto-researcher: %.2f of its runs beat the baseline (best %.3f, %d reproduced findings) vs "
                + "random config search %.2f (best %.3f); baseline accuracy %.3f%n", r.researcherBeat(),
                r.researcherBest(), r.accepted(), r.randomBeat(), r.randomBest(), r.baselineAccuracy());
        assertTrue(r.researcherBeat() > r.randomBeat() + 0.1, r.toString());
        assertTrue(r.accepted() >= 2, r.toString());
        assertTrue(r.writeUps().getFirst().startsWith("Hypothesis: "), r.writeUps().toString());
    }
}
