package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class DeepExplorationSkillsTest {

    @Test
    void skillsCoverThePointMaze() {
        Map<String, double[]> t = DeepExploration.skills(800, 1);
        System.out.printf("11.6.8 skills after 800 reward-free steps {cells of 100, held-out goals reached}: %s%n",
                DeepExploration.fmt(t));
        for (double[] v : t.values()) {
            assertTrue(v[0] >= 1, DeepExploration.fmt(t));
        }
    }
}
