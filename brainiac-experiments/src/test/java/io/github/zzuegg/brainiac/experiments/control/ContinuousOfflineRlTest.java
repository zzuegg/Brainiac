package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ContinuousOfflineRlTest {

    @Test
    void flowQLearningImprovesOnMixedContinuousData() {
        Map<String, Double> r = ContinuousOfflineRl.run(20, 80, 1000, 1);
        System.out.printf("offline WindyReach, normalized return: data %.2f, BC %.2f, flow BC %.2f, FQL %.2f, "
                + "FQL+Q-chunking %.2f, IQL %.2f, CQL %.2f%n", r.get("data"), r.get("BC"), r.get("flow BC"),
                r.get("FQL"), r.get("FQL + Q-chunking (h=4)"), r.get("IQL (5 levels)"), r.get("CQL (5 levels)"));
        assertTrue(r.get("FQL") > r.get("BC") + 0.2 && r.get("FQL") > r.get("flow BC") + 0.2, r.toString());
        assertTrue(r.get("FQL + Q-chunking (h=4)") > r.get("BC"), r.toString());
    }
}
