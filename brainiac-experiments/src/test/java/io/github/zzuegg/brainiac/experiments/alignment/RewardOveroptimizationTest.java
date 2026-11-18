package io.github.zzuegg.brainiac.experiments.alignment;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.1 and 9.2.2: reward over-optimization, ensembles, WARM and BOND. */
class RewardOveroptimizationTest {

    @Test
    void ensemblesAndWarmAgainstOveroptimization() {
        RewardOveroptimization.Result r = RewardOveroptimization.run(600, 30, 3, 1);
        StringBuilder sb = new StringBuilder();
        r.curves().forEach((k, c) -> {
            sb.append(String.format("%n  %-32s", k));
            if (!c.bestOfN().isEmpty()) {
                sb.append(" BoN gold");
                c.bestOfN().forEach(p -> sb.append(String.format(" %.3f", p.gold())));
            }
            sb.append(" | RL gold@KL");
            c.rl().forEach(p -> sb.append(String.format(" %.3f@%.2f", p.gold(), p.kl())));
        });
        System.out.printf("9.2.1 proxy accuracy %s (WARM members %.3f); gold by best-of-1/4/16/64 and GRPO:%s%n",
                r.proxyAccuracy(), r.warmMemberAccuracy(), sb);
        assertTrue(r.curves().size() > 5);
    }

    @Test
    void bondDistilsBestOfN() {
        Map<String, RewardOveroptimization.BondRow> r = RewardOveroptimization.bond(16, 600, 40, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format(" %s %.3f (KL %.2f, %d/query);", k, v.gold(), v.kl(),
                v.samplesPerQuery())));
        System.out.printf("9.2.2 gold reward, N=16:%s%n", sb);
        assertTrue(r.size() > 5);
    }
}
