package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SpuriousCorrelationTest {

    @Test
    void lastLayerRetrainingRaisesWorstGroupAccuracy() {
        for (DomainGeneralization.Spurious s : DomainGeneralization.Spurious.values()) {
            Map<String, DomainGeneralization.GroupScores> r = DomainGeneralization.spurious(s, 400, 1);
            StringBuilder sb = new StringBuilder();
            r.forEach((k, v) -> sb.append(String.format("  %s %.3f/%.3f", k, v.average(), v.worstGroup())));
            System.out.printf("Spurious %-14s (average/worst group)%s%n", s, sb);
            assertTrue(r.get("DFR").worstGroup() > r.get("ERM").worstGroup(), s + ": DFR beats ERM on worst group");
        }
    }
}
