package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class PagedServingTest {

    static Llama llama(long seed) {
        return Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), seed, Device.cpu(), DType.FLOAT32);
    }

    static int[] tokens(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = r.nextInt(40);
        }
        return t;
    }

    private static float maxDiff(float[] a, float[] b) {
        float d = 0;
        for (int i = 0; i < a.length; i++) {
            d = Math.max(d, Math.abs(a[i] - b[i]));
        }
        return d;
    }

    @Test
    void pagedLogitsEqualTheCoreCacheAcrossBlockBoundaries() {
        Llama core = llama(1);
        PagedLlama paged = PagedLlama.of(core, 4, 32);
        int[] prompt = tokens(11, 2);
        LanguageModel.State a = core.newState(1);
        LanguageModel.State b = paged.newState(1);
        float diff = maxDiff(core.next(a, new int[][] {prompt}, null).toArray(),
                paged.next(b, new int[][] {prompt}, null).toArray());
        for (int t = 0; t < 6; t++) {
            int[][] one = {{t + 3}};
            diff = Math.max(diff, maxDiff(core.next(a, one, null).toArray(), paged.next(b, one, null).toArray()));
        }
        System.out.printf("PagedLlama vs core Llama: max logit difference %.2e over 17 tokens%n", diff);
        assertTrue(diff < 1e-4, "diff " + diff);
        assertEquals(5, ((PagedKvCache) paged.store()).usedBlocks());   // ceil(17 / 4)
    }

    @Test
    void generatorRunsUnchangedAndForksShareBlocks() {
        Llama core = llama(3);
        PagedLlama paged = PagedLlama.of(core, 4, 64);
        int[] prompt = tokens(9, 4);
        int[] want = new Generator(core, null).generate(new int[][] {prompt}, Generator.Options.greedy(8), null)
                .getFirst().tokens();
        int[] got = new Generator(paged, null).generate(new int[][] {prompt}, Generator.Options.greedy(8), null)
                .getFirst().tokens();
        assertArrayEquals(want, got);
        PagedKvCache cache = (PagedKvCache) paged.store();
        int before = cache.usedBlocks();
        LanguageModel.State s = paged.newState(1);
        paged.next(s, new int[][] {prompt}, null);
        int own = cache.usedBlocks() - before;
        LanguageModel.State copy = paged.fork(s);
        assertEquals(own, cache.usedBlocks() - before, "a fork copies nothing");
        paged.next(copy, new int[][] {{1}}, null);
        assertEquals(own + 1, cache.usedBlocks() - before, "writing the shared partial block copies only it");
        paged.release(s);
        paged.release(copy);
        assertEquals(before, cache.usedBlocks());
    }

    @Test
    void coreBatcherDrivesThePagedModelThroughJoin() throws Exception {
        Llama core = llama(8);
        PagedLlama paged = PagedLlama.of(core, 4, 256);
        Generator plain = new Generator(core, null);
        int[][] prompts = {tokens(5, 1), tokens(9, 2), tokens(3, 3), tokens(12, 4), tokens(6, 5)};
        int[] lengths = {7, 3, 11, 5, 9};
        try (Generator.Batcher batcher = new Generator(paged, null).batcher(3)) {
            List<java.util.concurrent.CompletableFuture<Generator.Output>> futures = new java.util.ArrayList<>();
            for (int i = 0; i < prompts.length; i++) {
                futures.add(batcher.submit(prompts[i], Generator.Options.greedy(lengths[i])
                        .withStopTokens(java.util.Set.of()), null));
            }
            for (int i = 0; i < prompts.length; i++) {
                int[] want = plain.generate(new int[][] {prompts[i]}, Generator.Options.greedy(lengths[i])
                        .withStopTokens(java.util.Set.of()), null).getFirst().tokens();
                assertArrayEquals(want, futures.get(i).get().tokens(), "request " + i);
            }
            assertTrue(batcher.largestBatch() > 1);
        }
    }

    @Test
    void raggedBatchAndTreePassesMatchSeparateRuns() {
        Llama core = llama(5);
        PagedLlama paged = PagedLlama.of(core, 4, 64);
        KvBlockStore store = paged.store();
        int a = store.newSequence();
        int b = store.newSequence();
        int[] pa = tokens(7, 6);
        int[] pb = tokens(3, 7);
        List<PagedLlama.Output> both = paged.forward(List.of(PagedLlama.Pass.chain(a, pa),
                PagedLlama.Pass.chain(b, pb)), false);
        LanguageModel.State sa = core.newState(1);
        LanguageModel.State sb = core.newState(1);
        assertTrue(maxDiff(core.next(sa, new int[][] {pa}, null).toArray(), both.get(0).logits()) < 1e-4);
        assertTrue(maxDiff(core.next(sb, new int[][] {pb}, null).toArray(), both.get(1).logits()) < 1e-4);
        // a tree: root 5 with children 6 and 7, and 8 under 7; each branch must equal its own chain
        PagedLlama.Output tree = paged.forward(List.of(new PagedLlama.Pass(a, new int[] {5, 6, 7, 8},
                new int[] {-1, 0, 0, 2})), true).getFirst();
        LanguageModel.State branch = sa.select(new int[] {0});
        float[] chain = core.nextAll(branch, new int[][] {{5, 7, 8}}).toArray();
        int v = core.vocabSize();
        assertTrue(maxDiff(java.util.Arrays.copyOfRange(chain, 2 * v, 3 * v), tree.row(3, v)) < 1e-4);
        float[] other = core.nextAll(sa, new int[][] {{5, 6}}).toArray();
        assertTrue(maxDiff(java.util.Arrays.copyOfRange(other, v, 2 * v), tree.row(1, v)) < 1e-4);
    }
}
