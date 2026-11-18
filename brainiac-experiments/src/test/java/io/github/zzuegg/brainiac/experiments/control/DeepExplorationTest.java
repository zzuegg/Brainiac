package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class DeepExplorationTest {

    @Test
    void randomisedValueFunctionsExploreTheDeepSea() {
        Map<String, double[]> t = DeepExploration.deepSea(10, 2_500, 2, false);
        Map<String, Double> river = DeepExploration.riverSwim(2_000, 2);
        System.out.printf("11.6.6 deep sea N=10, 2500 steps {episodes to first reward, solved share}: %s| "
                + "river swim greedy return: %s%n", DeepExploration.fmt(t), river);
        assertTrue(t.get("RLSVI tabular")[0] < t.get("DQN eps-greedy")[0], DeepExploration.fmt(t));
        assertTrue(t.get("RLSVI tabular")[1] >= t.get("DQN eps-greedy")[1], DeepExploration.fmt(t));
    }
}
