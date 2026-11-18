package io.github.zzuegg.brainiac.experiments.serving;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class ThroughputShootoutTest {

    @Test
    void sgmvKeepsThroughputFlatInTheAdapterCount() {
        Map<Integer, Map<String, ThroughputShootout.LoraRow>> r = ThroughputShootout.multiLora(new int[] {4, 16, 64},
                300, 1.6, 1);
        r.forEach((count, rows) -> rows.forEach((k, v) -> System.out.printf(
                "8.4.7 %2d adapters %-30s %.0f tok/s, %.1f tok/iteration, p95 %.0f iterations%n", count, k,
                v.tokensPerSecond(), v.tokensPerIteration(), v.p95Latency())));
        ThroughputShootout.LoraRow sgmv = r.get(64).get("SGMV (S-LoRA/Punica)");
        ThroughputShootout.LoraRow merged = r.get(64).get("merged, per-adapter batches");
        assertTrue(sgmv.p95Latency() < merged.p95Latency(), r.toString());
        assertTrue(sgmv.tokensPerIteration() > 2 * merged.tokensPerIteration(), r.toString());
    }

    @Test
    void vtcBoundsTheServiceGap() {
        Map<String, ThroughputShootout.FairRow> r = ThroughputShootout.fairness(400, new long[] {1, 2, 3});
        r.forEach((k, v) -> System.out.printf(
                "8.4.8 %-18s service gap %.0f, light p95 %.0f, heavy p95 %.0f iterations, %.2f tokens/iteration%n", k,
                v.serviceGap(), v.lightP95(), v.heavyP95(), v.throughput()));
        assertTrue(r.get("VTC").lightP95() < r.get("FCFS").lightP95(), r.toString());
        assertTrue(r.get("VTC").throughput() > 0.95 * r.get("FCFS").throughput(), r.toString());
        assertTrue(r.get("request-rate limit").throughput() < r.get("VTC").throughput(), r.toString());
    }
}
