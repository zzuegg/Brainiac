package io.github.zzuegg.brainiac.experiments.bandits;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.ClickLogs;

import java.util.Map;

import org.junit.jupiter.api.Test;

class RankingBanditsTest {

    private static String fmt(Map<String, double[]> t) {
        StringBuilder b = new StringBuilder();
        t.forEach((k, v) -> b.append(String.format("%s %.1f/%.3f; ", k, v[0], v[1])));
        return b.toString();
    }

    @Test
    void cascadeAwarePoliciesUnderBothClickModels() {
        Map<String, double[]> cascade = RankingBandits.table(ClickLogs.ClickModel.CASCADE, 60, 5, 3_000, 3);
        Map<String, double[]> pbm = RankingBandits.table(ClickLogs.ClickModel.POSITION_BASED, 60, 5, 3_000, 3);
        System.out.printf("11.6.5 regret@3000 regret/CTR cascade: %s| PBM: %s%n", fmt(cascade), fmt(pbm));
        double random = cascade.get("RANDOM")[0];
        assertTrue(cascade.get("CASCADE_KL_UCB")[0] < random && cascade.get("CASCADE_LIN_UCB")[0] < random,
                fmt(cascade));
        assertTrue(pbm.get("PBM_UCB")[0] < pbm.get("RANDOM")[0], fmt(pbm));
    }
}
