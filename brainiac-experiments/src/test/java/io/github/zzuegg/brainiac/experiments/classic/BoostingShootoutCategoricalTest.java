package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class BoostingShootoutCategoricalTest {

    @Test
    void orderedTargetStatisticsAvoidTheLeakyEncodingsOverfit() {
        Map<String, double[]> r = BoostingShootout.categorical(1500, 300, 100, 1);
        String line = BoostingShootoutLightGbmTest.line("CatBoost test-logloss/AUC/train-logloss/gap", r, " %.3f");
        System.out.println(line);
        double[] ordered = r.get("CatBoost-style ordered boosting");
        double[] leaky = r.get("GBT + leaky target encoding");
        assertTrue(ordered[3] < leaky[3], line);
        assertTrue(ordered[0] < leaky[0], line);
        assertTrue(ordered[1] > 0.6, line);
    }
}
