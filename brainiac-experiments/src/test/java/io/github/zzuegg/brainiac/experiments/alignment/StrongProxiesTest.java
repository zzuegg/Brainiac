package io.github.zzuegg.brainiac.experiments.alignment;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/** W10 4.15: reward-model ensembles and WARM with stronger proxies (a small corner of the Run). */
class StrongProxiesTest {

    @Test
    void strongerProxiesReportAccuracyAndCurves() {
        List<RewardOveroptimization.StrongProxyRow> rows = RewardOveroptimization.strongProxies(new int[] {1}, 200, 40,
                2, 1, 1);
        RewardOveroptimization.StrongProxyRow row = rows.get(0);
        StringBuilder b = new StringBuilder();
        row.accuracy().forEach((k, v) -> b.append(String.format("%s %.3f, ", k, v)));
        row.curves().forEach((k, c) -> b.append(String.format("%s BoN64 gold %.3f RL final %.3f; ", k, c.bonFinal(),
                c.rlFinal())));
        System.out.printf("W10 4.15 strong proxies, %d pairs: accuracy %s%n", row.pairs(), b);
        for (double v : row.accuracy().values()) {
            assertTrue(v >= 0 && v <= 1, b.toString());
        }
        assertTrue(row.curves().containsKey("gold (oracle)") && row.curves().containsKey("WARM"), b.toString());
    }
}
