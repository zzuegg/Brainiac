package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.experimental.nn.AttentionCacheViewer;
import io.github.zzuegg.brainiac.experimental.nn.EvictingKvCache;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Experiment X10: renders the eviction comparison's stream as an attention-and-cache page.
 * Tokens are labelled {@code F} (fact), {@code Q} (question) or {@code .} (filler); the page
 * shows each question reading the facts the cache still holds, and the facts that were
 * dropped too early.
 */
public final class AttentionViewerDemo {

    private AttentionViewerDemo() {
    }

    /**
     * Runs a stream through a small cache and writes the page.
     *
     * @param file where to write the HTML
     * @param tokens the stream length
     * @param budget the cache budget
     * @param heavyHitters use H2O eviction instead of sinks
     * @return the number of recorded steps
     * @throws IOException if writing fails
     */
    public static int write(Path file, int tokens, int budget, boolean heavyHitters) throws IOException {
        MultiHeadAttention attention = EvictionComparison.layer();
        int[] kinds = new int[tokens];
        List<float[]> stream = EvictionComparison.stream(tokens, 11, kinds);
        EvictingKvCache cache = EvictingKvCache.of(attention).record(true)
                .budget(budget, heavyHitters ? EvictingKvCache.heavyHitters(budget / 4) : EvictingKvCache.sinks(2));
        for (int t = 0; t < tokens; t++) {
            cache.append(stream.get(t), (kinds[t] == 1 ? "F" : kinds[t] == 2 ? "Q" : ".") + t);
        }
        AttentionCacheViewer.write(file, "KV cache: " + (heavyHitters ? "heavy hitters" : "attention sinks")
                + ", budget " + budget, cache.steps());
        return cache.steps().size();
    }
}
