package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TdMpcComparisonTest {

    @Test
    void tdMpcIsSampleEfficientOnThePendulum() {
        Map<String, Double> r = TdMpcComparison.run(1000, 1);
        System.out.printf("W6 3.8 pendulum 1000 steps, greedy return: TD-MPC (MPPI) %.0f, its policy prior alone %.0f vs"
                + " SAC %.0f, MLP dynamics + CEM (true reward) %.0f%n", r.get("TD-MPC (MPPI)"),
                r.get("TD-MPC policy prior only"), r.get("SAC"), r.get("MLP dynamics + CEM (true reward)"));
        assertTrue(r.get("TD-MPC (MPPI)") > r.get("SAC"), r.toString());
    }
}
