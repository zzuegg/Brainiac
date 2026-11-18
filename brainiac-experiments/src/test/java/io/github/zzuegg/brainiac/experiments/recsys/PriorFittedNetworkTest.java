package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PriorFittedNetworkTest {

    @Test
    void pfnClassifiesSmallTablesInContext() {
        Map<String, Map<String, Double>> r = TabularShootout.priorFitted(150, 6, 1);
        String line = TabularStressTest.table("PFN accuracy (100 context rows):", r);
        System.out.println(line);
        Map<String, Double> prior = r.get("SCM/MLP prior");
        // 150 pre-training steps fit the CPU budget; 1500 steps reach 0.78 / 0.71 (see report)
        assertTrue(prior.get("PFN (in context)") > 0.65, line);
        assertTrue(prior.get("PFN (in context)") > prior.get("logistic") - 0.12, line);
        assertTrue(r.get("trees (held out)").get("PFN (in context)") > 0.55, line);
    }
}
