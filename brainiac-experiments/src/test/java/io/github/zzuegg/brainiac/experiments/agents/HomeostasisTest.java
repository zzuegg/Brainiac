package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class HomeostasisTest {

    @Test
    void driveReductionTeachesBalancedForaging() {
        Homeostasis.Result r = Homeostasis.run(5000, 1);
        System.out.println("homeostasis " + r);
        assertTrue(r.survivalTrained() > 2 * r.survivalRandom(), r.toString());
        assertTrue(r.foodShare() > 0.3 && r.foodShare() < 0.7, r.toString());
        assertTrue(r.meanDeviation() < 0.5, r.toString());
    }
}
