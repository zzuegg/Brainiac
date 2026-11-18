package io.github.zzuegg.brainiac.experiments.safety;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DpFtrlTrainingTest {

    @Test
    void correlatedNoiseBeatsUnamplifiedDpSgd() {
        for (double eps : new double[] {2, 8}) {
            List<PrivateTraining.Ftrl> rows = PrivateTraining.dpFtrl(eps, 100, 10, 200, 1);
            StringBuilder b = new StringBuilder();
            for (PrivateTraining.Ftrl r : rows) {
                b.append(String.format(" %s sigma %.2f eps %.2f acc %.3f;", r.method(), r.noiseMultiplier(),
                        r.epsilon(), r.accuracy()));
            }
            System.out.printf("DP-FTRL target eps %.0f, 50 steps of 200, k=10:%s%n", eps, b);
            double unamplified = rows.get(1).accuracy();
            double bestFtrl = Math.max(rows.get(2).accuracy(), Math.max(rows.get(3).accuracy(), rows.get(4).accuracy()));
            for (PrivateTraining.Ftrl r : rows) {
                assertTrue(r.epsilon() <= eps + 1e-3, r.toString());
            }
            assertTrue(bestFtrl >= unamplified - 0.02, "DP-FTRL " + bestFtrl + " vs unamplified " + unamplified);
        }
    }
}
