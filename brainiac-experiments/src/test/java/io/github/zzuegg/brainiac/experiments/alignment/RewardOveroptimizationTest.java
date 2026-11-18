package io.github.zzuegg.brainiac.experiments.alignment;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.1: reward over-optimization against a gold reward; ensembles and WARM. */
class RewardOveroptimizationTest {

    @Test
    void ensemblesAndWarmAgainstOveroptimization() {
        RewardOveroptimization.Result r = RewardOveroptimization.run(600, 16, 2, 1);
        StringBuilder sb = new StringBuilder();
        r.curves().forEach((k, c) -> {
            sb.append(String.format("%n  %-18s BoN gold", k));
            c.bestOfN().forEach(p -> sb.append(String.format(" %.3f", p.gold())));
            sb.append(" | GRPO gold@KL");
            c.rl().forEach(p -> sb.append(String.format(" %.3f@%.2f", p.gold(), p.kl())));
        });
        System.out.printf("9.2.1 proxy accuracy %s (WARM members %.3f); gold by best-of-1/4/16/64 and GRPO"
                + " (single RM is the baseline):%s%n", r.proxyAccuracy(), r.warmMemberAccuracy(), sb);
        List<RewardOveroptimization.Point> oracle = r.curves().get("gold (oracle)").bestOfN();
        for (int i = 1; i < oracle.size(); i++) {
            assertTrue(oracle.get(i).gold() >= oracle.get(i - 1).gold(), "best-of-n under the gold reward never hurts");
        }
    }
}
