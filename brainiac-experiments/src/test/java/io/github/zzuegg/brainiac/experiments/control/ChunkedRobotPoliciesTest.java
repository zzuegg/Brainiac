package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.FlowActionPolicy;
import org.junit.jupiter.api.Test;

import java.util.Map;

class ChunkedRobotPoliciesTest {

    @Test
    void diffusionPolicyHandlesMultimodalDemonstrations() {
        Map<String, Double> r = ChunkedRobotPolicies.imitation(60, 1500, 40, 1);
        System.out.printf("multimodal reach success: demonstrator %.2f, MSE BC %.2f, Gaussian BC %.2f, "
                + "Diffusion Policy %.2f, flow action expert %.2f%n", r.get("demonstrator"), r.get("MSE BC"),
                r.get("Gaussian BC"), r.get("Diffusion Policy"), r.get("flow action expert"));
        assertTrue(r.get("Diffusion Policy") > r.get("MSE BC") + 0.3, r.toString());
        assertTrue(r.get("flow action expert") > r.get("MSE BC") + 0.3, r.toString());
    }

    @Test
    void realTimeChunkingSmoothsChunkBoundaries() {
        FlowActionPolicy policy = ChunkedRobotPolicies.flowPolicy(ChunkedRobotPolicies.demonstrations(60, 2), 1500, 2);
        Map<String, double[]> r = ChunkedRobotPolicies.realTime(policy, 2, 4, 40, 2);
        double[] naive = r.get("NAIVE");
        double[] ens = r.get("TEMPORAL_ENSEMBLE");
        double[] rtc = r.get("INPAINTING");
        System.out.printf("delay 2: success / jump at switch: naive %.2f / %.3f, temporal ensemble %.2f / %.3f, "
                + "RTC %.2f / %.3f%n", naive[0], naive[1], ens[0], ens[1], rtc[0], rtc[1]);
        assertTrue(rtc[1] < naive[1], "RTC joins chunks more smoothly");
        assertTrue(rtc[0] >= naive[0] - 0.1, "and succeeds as often");
    }
}
