package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class RaftFlowTest {

    @Test
    void recurrentRefinementFollowsLargeMotions() {
        double[][] raft = DensePrediction.raft(true, 120, 3, new int[] {1, 2, 4, 8}, 1);
        double[] zero = DensePrediction.classicalFlow(DensePrediction.FlowMethod.ZERO, true, 0.02f, 1);
        System.out.printf("RAFT-lite moving-pixel EPE, 6-10 px, after 1/2/4/8 iterations: %.3f/%.3f/%.3f/%.3f"
                + " (zero flow %.3f)%n", raft[0][1], raft[1][1], raft[2][1], raft[3][1], zero[1]);
        assertTrue(raft[2][1] < zero[1], "RAFT-lite follows large motions");
    }
}
