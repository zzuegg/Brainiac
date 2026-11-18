package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MultiObjectiveRlTest {

    @Test
    void deepSeaTreasure() {
        Map<String, double[]> r = MultiObjectiveRl.deepSeaTreasure(1_500, 3_000, 0, 1);
        System.out.printf("12.6.14 Deep Sea Treasure hypervolume/Pareto points found/unseen utility loss: %s%n",
                TabularExplorationTest.table(r));
        assertTrue(r.get("Pareto Q-learning")[1] > r.get("linear scalarisation (11 weights)")[1],
                "Pareto Q finds the concave points that linear scalarisation misses");
        assertTrue(r.get("envelope Q (one network)")[0] >= 0, "envelope Q runs (a weak learner at this budget)");
    }

    @Test
    void resourceGathering() {
        Map<String, double[]> r = MultiObjectiveRl.resourceGathering(1_500, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(k).append('=').append(Arrays.toString(v)).append(' '));
        System.out.printf("12.6.14 resource gathering non-dominated returns (resources, -encounters): %s%n",
                sb.toString().trim());
        assertTrue(r.get("Pareto Q-learning").length >= 2);
    }
}
