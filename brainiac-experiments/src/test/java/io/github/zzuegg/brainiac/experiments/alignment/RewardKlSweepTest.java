package io.github.zzuegg.brainiac.experiments.alignment;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.1 baselines: a single proxy under a KL-penalty sweep and PPO. */
class RewardKlSweepTest {

    @Test
    void klPenaltySweep() {
        Map<String, List<RewardOveroptimization.Point>> r = RewardOveroptimization.klSweep(600, 20, 2, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, c) -> {
            sb.append(' ').append(k).append(" gold@KL");
            c.forEach(p -> sb.append(String.format(" %.3f@%.2f", p.gold(), p.kl())));
            sb.append(';');
        });
        System.out.printf("9.2.1 single proxy RM by KL penalty:%s%n", sb);
        assertEquals(4, r.size());
    }
}
