package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ConstrainedRlComparisonTest {

    @Test
    void constrainedMethodsKeepTheCostNearTheLimit() {
        Map<String, ConstrainedRlComparison.Outcome> r = ConstrainedRlComparison.run(8_000, 0.2f, 3);
        ConstrainedRlComparison.Outcome cpo = r.get("CPO");
        ConstrainedRlComparison.Outcome lag = r.get("PPO-Lagrangian");
        ConstrainedRlComparison.Outcome k0 = r.get("PPO, shaping k=0");
        ConstrainedRlComparison.Outcome k01 = r.get("PPO, shaping k=0.1");
        System.out.printf("W6 3.14 hazard grid (limit 0.2) return/cost: CPO %.2f/%.2f, PPO-Lagrangian %.2f/%.2f"
                + " vs PPO k=0 %.2f/%.2f, PPO k=0.1 %.2f/%.2f%n", cpo.meanReturn(), cpo.meanCost(), lag.meanReturn(),
                lag.meanCost(), k0.meanReturn(), k0.meanCost(), k01.meanReturn(), k01.meanCost());
        assertTrue(lag.meanCost() < k0.meanCost(), r.toString());
        assertTrue(cpo.meanCost() < k0.meanCost(), r.toString());
        assertTrue(lag.meanReturn() > 0.5 && cpo.meanReturn() > 0.5, r.toString());
        assertTrue(lag.meanCost() < 0.4 && cpo.meanCost() < 0.5, r.toString());
    }
}
