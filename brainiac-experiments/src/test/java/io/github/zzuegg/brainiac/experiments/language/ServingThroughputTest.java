package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.ChunkedPrefillScheduler;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ServingThroughputTest {

    @Test
    void prefixCacheCutsPromptWorkAndTimeToFirstToken() {
        ServingThroughput.PrefixResult r = ServingThroughput.prefixReuse(160, 8, 8, 1);
        System.out.printf("Prefix cache TTFT ms: %s; prompt tokens read: %s; max logit diff %.1e%n",
                r.ttftMillis(), r.promptTokensRead(), r.maxLogitDiff());
        assertTrue(r.maxLogitDiff() < 1e-3, "reuse must not change the logits");
        assertTrue(r.promptTokensRead().get("paged + prefix cache") * 10 < r.promptTokensRead().get("no reuse"));
        assertTrue(r.ttftMillis().get("paged + prefix cache") < 0.7 * r.ttftMillis().get("no reuse"),
                r.ttftMillis().toString());
    }

    @Test
    void pagedCacheHoldsLessAndFitsMoreSequences() {
        ServingThroughput.MemoryResult r = ServingThroughput.pagedMemory(16, 16, 256, 2);
        System.out.printf("Paged KV peak %d KB vs core contiguous %d KB (one padded batch %d KB); "
                + "sequences in 256 blocks: paged %d vs core %d%n", r.pagedPeakBytes() / 1024,
                r.contiguousBytes() / 1024, r.batchedBytes() / 1024, r.pagedFit(), r.contiguousFit());
        assertTrue(r.identical(), "paged serving must give the core greedy tokens");
        assertTrue(r.pagedPeakBytes() < 0.8 * r.contiguousBytes());
        assertTrue(r.pagedFit() > 1.2 * r.contiguousFit());
    }

    @Test
    void continuousBatchingCutsTailLatency() {
        Map<String, ServingThroughput.Latency> r = ServingThroughput.continuousBatching(32, 4, 8, 5);
        StringBuilder line = new StringBuilder("Batching under Poisson arrivals (tok/pass, p50/p95 passes, tok/s):");
        r.forEach((k, v) -> line.append(String.format(" %s %.2f, %.0f/%.0f, %.0f;", k, v.tokensPerPass(),
                v.p50Passes(), v.p95Passes(), v.tokensPerSecond())));
        System.out.println(line);
        ServingThroughput.Latency fixed = r.get("static batching");
        for (String k : new String[] {"continuous (core Llama)", "continuous (PagedLlama)"}) {
            assertTrue(r.get(k).p95Passes() < 0.8 * fixed.p95Passes(), line.toString());
            assertTrue(r.get(k).tokensPerPass() > fixed.tokensPerPass(), line.toString());
        }
    }

    @Test
    void chunkedPrefillSmoothsDecodeLatency() {
        ServingThroughput.ChunkResult r = ServingThroughput.chunkedPrefill(256, 32, 3);
        long[] mono = r.monolithic().decodeGapsInWork();
        long[] chunk = r.chunked().decodeGapsInWork();
        double[] monoMs = r.monolithic().decodeGapsInMillis();
        double[] chunkMs = r.chunked().decodeGapsInMillis();
        System.out.printf("Chunked prefill decode gap p99/max: %d/%d tokens, %.2f/%.2f ms (%.0f tok/s) vs monolithic "
                + "%d/%d tokens, %.2f/%.2f ms (%.0f tok/s)%n",
                ChunkedPrefillScheduler.Report.quantile(chunk, 0.99), chunk[chunk.length - 1],
                ChunkedPrefillScheduler.Report.quantile(chunkMs, 0.99), chunkMs[chunkMs.length - 1],
                r.chunked().tokensPerSecond(), ChunkedPrefillScheduler.Report.quantile(mono, 0.99),
                mono[mono.length - 1], ChunkedPrefillScheduler.Report.quantile(monoMs, 0.99),
                monoMs[monoMs.length - 1], r.monolithic().tokensPerSecond());
        assertTrue(chunk[chunk.length - 1] * 3 < mono[mono.length - 1], "the worst stall must shrink");
        for (int i = 0; i < r.chunked().completions().size(); i++) {
            assertTrue(java.util.Arrays.equals(r.chunked().completions().get(i).tokens(),
                    r.monolithic().completions().get(i).tokens()), "scheduling must not change the tokens");
        }
    }

    @Test
    void offloadKeepsFewBytesHotAndBeatsWindowsAtTheSameBudget() {
        ServingThroughput.OffloadResult r = ServingThroughput.offload(192, 24, 8, 6, 4);
        System.out.printf("KV offload: hot %d KB of %d KB (windows keep %d tokens), %.0f B fetched/token, "
                + "%.0f vs %.0f tok/s (full); logit MAE %s; top-1 agreement %s%n", r.hotBytes() / 1024,
                r.fullBytes() / 1024, r.windowTokens(), r.transferPerToken(), r.tokensPerSecond()[0],
                r.tokensPerSecond()[1], r.meanAbsError(), r.top1Agreement());
        assertTrue(r.hotBytes() * 2.5 < r.fullBytes());
        assertTrue(r.meanAbsError().get("paged full KV") < 1e-4);
        double off = r.meanAbsError().get("offload, key sketch");
        assertTrue(off < 0.7 * r.meanAbsError().get("core slidingWindow"), r.meanAbsError().toString());
        assertTrue(off < 0.7 * r.meanAbsError().get("core StreamingLLM state"), r.meanAbsError().toString());
    }
}
