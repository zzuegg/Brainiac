package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.DisaggregatedScheduler;

import org.junit.jupiter.api.Test;

class SharedPrefixServingTest {

    @Test
    void hydragenReadsThePrefixOnceAndDecodesFaster() {
        SharedPrefixServing.HydragenResult r = SharedPrefixServing.hydragen(1024, 16, 32, 15, 1);
        System.out.printf("Hydragen, 32 samples of a 1024-token prompt: %.2f ms vs %.2f ms per-sequence (%.1fx), "
                + "keys read %.0f%%, max diff %.1e%n", r.hydragenMillis(), r.perSequenceMillis(), r.speedup(),
                100 * r.keyReadsShare(), r.maxDifference());
        assertTrue(r.maxDifference() < 1e-4, "the decomposition is exact");
        assertTrue(r.keyReadsShare() < 0.1);
        assertTrue(r.speedup() > 1.2, "one batched prefix product beats 32 matrix-vector passes");
    }

    @Test
    void disaggregationRaisesGoodputUnderATightTpotGoal() {
        SharedPrefixServing.DisaggregationResult r = SharedPrefixServing.disaggregation(40, 10, 2);
        DisaggregatedScheduler.Slo slo = new DisaggregatedScheduler.Slo(60, 3.0);
        System.out.printf("Disaggregated vs colocated chunked (2 workers): goodput %.1f vs %.1f req/s, SLO attainment "
                + "%.2f vs %.2f, mean TTFT %.1f vs %.1f ms, mean TPOT %.2f vs %.2f ms%n",
                r.disaggregated().goodput(slo), r.colocated().goodput(slo), r.disaggregated().attainment(slo),
                r.colocated().attainment(slo), r.disaggregated().meanTtft(), r.colocated().meanTtft(),
                r.disaggregated().meanTpot(), r.colocated().meanTpot());
        assertTrue(r.sameTokens(), "the layout must not change the tokens");
        assertTrue(r.disaggregated().meanTpot() < r.colocated().meanTpot());
        assertTrue(r.disaggregated().attainment(slo) > r.colocated().attainment(slo));
    }
}
