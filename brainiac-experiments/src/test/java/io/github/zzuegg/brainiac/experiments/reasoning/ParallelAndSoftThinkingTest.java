package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.11: soft thinking and ParaThinker against CoT, self-consistency and DeepConf. */
class ParallelAndSoftThinkingTest {

    @Test
    void softThinkingMatchesGreedyAndParallelPathsMerge() {
        Map<String, ParallelAndSoftThinking.Arm> arms = ParallelAndSoftThinking.run(new ChainTask(5), 120, 4, 1);
        StringBuilder sb = new StringBuilder();
        arms.forEach((k, v) -> sb.append(String.format(" %s %.2f (%.1f tok);", k, v.accuracy(), v.tokens())));
        System.out.printf("chain(5) held-out accuracy:%s%n", sb);
        double greedy = arms.get("CoT greedy").accuracy();
        double sampled = arms.get("CoT sampled").accuracy();
        assertTrue(arms.get("soft thinking").accuracy() > sampled, "soft thinking beats a sampled chain");
        assertTrue(arms.get("soft thinking").accuracy() >= greedy - 0.05, "soft thinking keeps up with greedy");
        assertTrue(arms.get("self-consistency x4").accuracy() > sampled, "voting over paths beats one path");
        assertTrue(arms.get("ParaThinker x4").accuracy() >= sampled - 0.02, "the learned merger keeps one path's level");
    }
}
