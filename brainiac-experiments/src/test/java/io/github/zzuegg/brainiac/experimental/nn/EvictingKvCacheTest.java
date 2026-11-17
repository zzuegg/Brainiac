package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EvictingKvCacheTest {

    private static MultiHeadAttention mha(MultiHeadAttention.Spec spec) {
        return (MultiHeadAttention) spec.create(Shape.of(10, 8), new SplittableRandom(3));
    }

    @Test
    void keepingEverythingMatchesTheLayersForwardPass() {
        MultiHeadAttention.Spec[] specs = {
            MultiHeadAttention.of(2).causal(true),
            MultiHeadAttention.of(4).kvHeads(2).causal(true).rotary(10000),
            MultiHeadAttention.of(2).headSize(6).causal(true).bias(false).rotary(500),
        };
        Tensor x = Tensor.normal(Shape.of(1, 10, 8), 0, 1, new SplittableRandom(4));
        for (MultiHeadAttention.Spec spec : specs) {
            MultiHeadAttention a = mha(spec);
            float[] full = a.forward(Var.of(x)).value().toArray();
            EvictingKvCache cache = EvictingKvCache.of(a);
            float[] xs = x.toArray();
            for (int t = 0; t < 10; t++) {
                float[] y = cache.append(java.util.Arrays.copyOfRange(xs, t * 8, t * 8 + 8));
                assertArrayEquals(java.util.Arrays.copyOfRange(full, t * 8, t * 8 + 8), y, 1e-4f, spec + " step " + t);
            }
        }
    }

    @Test
    void policiesDropTheRightTokens() {
        MultiHeadAttention a = mha(MultiHeadAttention.of(2).causal(true));
        SplittableRandom r = new SplittableRandom(5);
        EvictingKvCache sliding = EvictingKvCache.of(a).budget(4, EvictingKvCache.sliding());
        EvictingKvCache sinks = EvictingKvCache.of(a).budget(4, EvictingKvCache.sinks(2));
        EvictingKvCache h2o = EvictingKvCache.of(a).budget(4, EvictingKvCache.heavyHitters(2)).record(true);
        for (int t = 0; t < 9; t++) {
            float[] x = new float[8];
            for (int i = 0; i < 8; i++) {
                x[i] = (float) r.nextGaussian();
            }
            sliding.append(x);
            sinks.append(x);
            h2o.append(x);
        }
        assertArrayEquals(new int[] {5, 6, 7, 8}, sliding.positions());
        assertArrayEquals(new int[] {0, 1, 7, 8}, sinks.positions());
        assertEquals(4, h2o.size());
        int[] kept = h2o.positions();
        assertEquals(7, kept[2]);
        assertEquals(8, kept[3]);
        int dropped = 0;
        for (EvictingKvCache.Step s : h2o.steps()) {
            dropped += s.evicted().length;
            float sum = 0;
            for (float w : s.weights()) {
                sum += w;
            }
            assertEquals(1, sum, 1e-4f);
        }
        assertEquals(5, dropped);
    }

    @Test
    void snapKvKeepsTheWindowAndWhatItVotedFor() {
        float[][] votes = {
            {0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.2f, 0.2f},
            {0.0f, 0.3f, 0.0f, 0.0f, 0.0f, 0.4f, 0.0f, 0.2f, 0.1f},
        };
        boolean[] kept = SnapKv.of(2, 1).select(votes, 4);
        assertArrayEquals(new boolean[] {false, true, false, false, false, true, false, true, true}, kept);
        // pooling keeps the neighbours of a strongly voted token
        boolean[] pooled = SnapKv.of(2, 3).select(votes, 5);
        assertTrue(pooled[0] || pooled[2]);
    }

    @Test
    void learnedScorerFitsFutureAttention() {
        MultiHeadAttention a = mha(MultiHeadAttention.of(2).causal(true));
        SplittableRandom r = new SplittableRandom(6);
        java.util.List<float[]> stream = new java.util.ArrayList<>();
        for (int t = 0; t < 60; t++) {
            float[] x = new float[8];
            for (int i = 0; i < 8; i++) {
                x[i] = (float) r.nextGaussian();
            }
            stream.add(x);
        }
        LearnedEvictionScorer.Examples ex = LearnedEvictionScorer.harvest(a, stream, 8, 4);
        assertEquals(ex.features().length, ex.targets().length);
        LearnedEvictionScorer s = new LearnedEvictionScorer(8, 1);
        float before = s.fit(ex, 1);
        float after = s.fit(ex, 200);
        assertTrue(after < before, before + " -> " + after);
    }
}
