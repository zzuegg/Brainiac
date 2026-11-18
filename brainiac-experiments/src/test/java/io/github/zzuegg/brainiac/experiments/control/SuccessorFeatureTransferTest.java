package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SuccessorFeatureTransferTest {

    @Test
    void gpiTransfersZeroShotToNewRewards() {
        double gpi = 0;
        double inferred = 0;
        double small = 0;
        double large = 0;
        int seeds = 3;
        for (long seed = 1; seed <= seeds; seed++) {
            Map<String, Double> r = SuccessorFeatureTransfer.run(4000, 300, 4000, seed);
            gpi += r.get("SF + GPI, zero-shot") / seeds;
            inferred += r.get("SF + GPI, inferred w") / seeds;
            small += r.get("Q-learning, 300 steps") / seeds;
            large += r.get("Q-learning, 4000 steps") / seeds;
        }
        System.out.printf("W6 3.13 new-task return: SF+GPI zero-shot %.3f (inferred w %.3f) vs Q-learning from scratch"
                + " %.3f at 300 steps, %.3f at 4000 steps%n", gpi, inferred, small, large);
        assertTrue(gpi > small, "GPI " + gpi + " vs small " + small);
        assertTrue(gpi > large - 0.3, "GPI " + gpi + " vs large " + large);
    }
}
