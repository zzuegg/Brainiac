package io.github.zzuegg.brainiac.experiments.alignment;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.2: BOND and J-BOND distil best-of-N into the policy. */
class BondTest {

    @Test
    void bondDistilsBestOfN() {
        Map<String, RewardOveroptimization.BondRow> r = RewardOveroptimization.bond(16, 600, 40, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format(" %s %.3f (KL %.2f, %d/query);", k, v.gold(), v.kl(),
                v.samplesPerQuery())));
        System.out.printf("9.2.2 gold reward, N=16 (best-of-N at inference is the target):%s%n", sb);
        assertTrue(r.size() > 5);
    }
}
