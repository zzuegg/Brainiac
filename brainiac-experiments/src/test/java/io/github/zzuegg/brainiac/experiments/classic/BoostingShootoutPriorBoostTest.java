package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class BoostingShootoutPriorBoostTest {

    @Test
    void pfnBoostAgainstItsParts() {
        Map<String, double[]> r = BoostingShootout.priorBoost(new int[] {30, 300}, 150, 2, 128, 1);
        String line = BoostingShootoutLightGbmTest.line("PFN-Boost acc PFN/GBT/PFN-Boost/stacking/scale", r,
                " %.3f");
        System.out.println(line);
        for (double[] v : r.values()) {
            assertTrue(v[2] > 0.55, line);
            assertTrue(v[2] >= Math.min(v[0], v[1]) - 0.05, line);
        }
    }
}
