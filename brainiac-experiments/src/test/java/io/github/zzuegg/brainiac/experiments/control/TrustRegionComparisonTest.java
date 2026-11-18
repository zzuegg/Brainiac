package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TrustRegionComparisonTest {

    @Test
    void trpoLearnsCartPoleWithinItsTrustRegion() {
        Map<String, TrustRegionComparison.Outcome> r = TrustRegionComparison.run(15_000, 3);
        TrustRegionComparison.Outcome trpo = r.get("TRPO");
        System.out.printf("W6 3.1 TRPO CartPole return %.1f (min %.1f, KL mean %.4f max %.4f) vs PPO %.1f, A2C %.1f%n",
                trpo.meanReturn(), trpo.minReturn(), trpo.meanKl(), trpo.maxKl(), r.get("PPO (core)").meanReturn(),
                r.get("A2C (core)").meanReturn());
        assertTrue(trpo.meanReturn() > 60, r.toString());
        assertTrue(trpo.maxKl() <= 0.03 + 1e-6, r.toString());
    }
}
