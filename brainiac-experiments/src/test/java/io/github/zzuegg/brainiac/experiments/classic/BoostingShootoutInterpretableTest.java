package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class BoostingShootoutInterpretableTest {

    @Test
    void ebmRecoversShapesAndPlantedPairs() {
        Map<String, double[]> r = BoostingShootout.interpretable(2000, 300, 1);
        String line = BoostingShootoutLightGbmTest.line("EBM acc/AUC/shape-corr/ms (pairs: ranks)", r, " %.3f");
        System.out.println(line);
        double[] ebm = r.get("EBM (GA2M, 2 pairs)");
        assertTrue(ebm[0] > r.get("logistic regression")[0], line);
        assertTrue(ebm[2] > 0.8, line);
        double[] ranks = r.get("EBM planted pairs");
        assertTrue(ranks[0] <= 3 && ranks[1] <= 3, line);
        assertTrue(ebm[0] > r.get("core GBT (black box)")[0] - 0.05, line);
    }
}
