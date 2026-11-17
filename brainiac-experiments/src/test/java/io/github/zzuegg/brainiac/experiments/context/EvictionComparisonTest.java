package io.github.zzuegg.brainiac.experiments.context;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EvictionComparisonTest {

    @Test
    void learnedEvictionKeepsTheFactsQuestionsNeed() {
        Map<String, Double> e = EvictionComparison.run(16, 300, 7);
        System.out.printf("eviction error at question steps: %s%n", e);
        assertTrue(e.get("learned") < 0.5 * e.get("h2o"), e.toString());
        assertTrue(e.get("learned") < 0.5 * e.get("sliding"), e.toString());
    }

    @Test
    void snapKvKeepsWhatThePromptsQuestionVotesFor() {
        Map<String, Double> e = EvictionComparison.snapKv(12, 3);
        System.out.printf("prompt compression error: %s%n", e);
        assertTrue(e.get("snapkv") < 0.2 * e.get("recent"), e.toString());
    }
}
