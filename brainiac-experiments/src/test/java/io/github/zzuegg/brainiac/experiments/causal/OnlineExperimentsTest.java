package io.github.zzuegg.brainiac.experiments.causal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class OnlineExperimentsTest {

    @Test
    void peekingInflatesFalsePositivesButMsprtStaysValid() {
        Map<String, Double> r = OnlineExperiments.peeking(200, 4000, 20, 1);
        String line = HeterogeneousEffectsTest.line("A/A false-positive rate:", r);
        System.out.println(line);
        assertTrue(r.get("t-test peeked daily") > r.get("fixed horizon t-test"), line);
        assertTrue(r.get("mSPRT (always valid)") <= 0.08, line);
    }

    @Test
    void cupedCutsTheVarianceByRhoSquared() {
        Map<String, Double> r = OnlineExperiments.power(100, 20000, 0.7, 0.02, 1);
        String line = HeterogeneousEffectsTest.line("2% lift, rho 0.7:", r);
        System.out.println(line);
        // theory: 1 - rho^2 = 0.51
        assertTrue(r.get("relative n CUPED") < 0.65, line);
        assertTrue(r.get("power CUPED") >= r.get("power t-test"), line);
    }
}
