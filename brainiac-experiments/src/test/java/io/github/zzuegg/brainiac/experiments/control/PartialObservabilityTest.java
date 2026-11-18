package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PartialObservabilityTest {

    @Test
    void recurrentReplayRemembersTheCue() {
        Map<String, Double> r = PartialObservability.run(4, 4000, 3);
        System.out.printf("W6 3.4 T-maze (length 4) success: R2D2 %.2f vs zero-state replay %.2f, frame-stack DQN %.2f%n",
                r.get("R2D2 (stored state, burn-in)"), r.get("zero state, no burn-in"),
                r.get("DQN, 2 stacked frames (core)"));
        assertTrue(r.get("R2D2 (stored state, burn-in)") > 0.8, r.toString());
        assertTrue(r.get("R2D2 (stored state, burn-in)") > r.get("DQN, 2 stacked frames (core)"), r.toString());
    }
}
