package io.github.zzuegg.brainiac.experiments.memory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EpisodicAgentComparisonTest {

    @Test
    void rememberingAgentOutperformsMemorylessOnes() {
        EpisodicAgentComparison.Result r = EpisodicAgentComparison.run(8000, 1);
        assertTrue(r.episodicGreedy() > 0.5, "greedy episodic return " + r.episodicGreedy());
        assertTrue(r.episodicLate() > r.memorylessLate() + 0.5, r.toString());
        assertTrue(r.episodicGreedy() > r.randomGreedy() + 0.5, r.toString());
        assertTrue(r.episodicEpisodes() > 1.5 * r.memorylessEpisodes(), r.toString());
    }
}
