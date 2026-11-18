package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.interp.ExplainabilityBenchmark.AttentionMethod;
import io.github.zzuegg.brainiac.experiments.interp.ExplainabilityBenchmark.AttentionStudy;

import org.junit.jupiter.api.Test;

class AttentionRolloutBenchmarkTest {

    @Test
    void attributionsFindTheCopiedSource() {
        AttentionStudy s = ExplainabilityBenchmark.attention(600, 1);
        System.out.printf("Copy task accuracy %.3f%n", s.copyAccuracy());
        s.scores().forEach((m, v) -> System.out.printf("Attention %-22s source agreement %.3f deletion AUC %.3f"
                + " (%.3f ms/sequence)%n", m, v.agreement(), v.deletionAuc(), v.millis()));
        assertTrue(s.copyAccuracy() > 0.6, "the copy model learns");
        double best = Math.max(s.scores().get(AttentionMethod.ROLLOUT).agreement(),
                s.scores().get(AttentionMethod.INTEGRATED_GRADIENTS).agreement());
        assertTrue(best > 0.3, "some attribution points at the source (chance is about 1/7)");
    }
}
