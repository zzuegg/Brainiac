package io.github.zzuegg.brainiac.experiments.serving;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class KvMemoryShootoutTest {

    @Test
    void miniCacheHalvesMergedLayersForASmallDivergence() {
        Map<String, KvMemoryShootout.MemoryRow> r = KvMemoryShootout.crossLayerMerging(96, 24, new long[] {1, 2, 3});
        r.forEach((k, v) -> System.out.printf("8.4.3 %-36s %.0f B/token, KL %.2e, top-1 %.2f, retained %.2f%n", k,
                v.bytesPerToken(), v.kl(), v.top1(), v.retention()));
        KvMemoryShootout.MemoryRow full = r.get("full cache");
        KvMemoryShootout.MemoryRow mini = r.get("MiniCache (2,3),(4,5)");
        KvMemoryShootout.MemoryRow naive = r.get("naive layer sharing (untrained CLA)");
        assertTrue(mini.bytesPerToken() < 0.85 * full.bytesPerToken(), r.toString());
        assertTrue(mini.kl() < naive.kl(), r.toString());
    }

    @Test
    void lshSamplingIsUnbiasedOnDenseAttention() {
        Map<String, Map<String, KvMemoryShootout.AttentionRow>> r = KvMemoryShootout.sampledAttention(1024, 20,
                new long[] {1, 2, 3}, 6, 100, 3);
        r.forEach((task, rows) -> rows.forEach((k, v) -> System.out.printf(
                "8.4.4 %-11s %-28s rel. error %.3f, accuracy %.2f, keys touched %.0f%n", task, k, v.relativeError(),
                v.accuracy(), v.keysTouched())));
        Map<String, KvMemoryShootout.AttentionRow> agg = r.get("aggregation");
        assertTrue(agg.get("MagicPIG (LSH sampling)").relativeError() < agg.get("top-k (same keys)").relativeError(),
                agg.toString());
        assertTrue(r.get("needle").get("top-k (same keys)").accuracy() > 0.9, r.toString());
    }
}
