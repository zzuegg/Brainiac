package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ActionTokenPoliciesTest {

    @Test
    void fastTokensMatchBinningWithAThirdOfTheTokens() {
        Map<String, double[]> r = ChunkedRobotPolicies.tokenized(60, 600, 20, 1);
        double[] fast = r.get("FAST");
        double[] bins = r.get("binning");
        System.out.printf("AR token policy on multimodal reach: FAST success %.2f with %.1f tokens per 8x2 chunk, "
                + "binning success %.2f with %.0f tokens%n", fast[0], fast[1], bins[0], bins[1]);
        assertTrue(fast[1] < bins[1] / 2, "FAST needs far fewer tokens per chunk");
        assertTrue(fast[0] > 0.35, "and the policy still reaches the goal: " + fast[0]);
    }
}
