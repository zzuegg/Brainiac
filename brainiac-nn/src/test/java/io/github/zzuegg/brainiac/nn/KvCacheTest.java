package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Generating with cached keys and values gives what a full pass over the window gives. */
class KvCacheTest {

    private static final int V = 11;

    private static Model lm(int length, int layers, TransformerBlock.Spec block) {
        Model.Builder b = Model.sequential().input(length).add(Embedding.of(V, 16));
        for (int i = 0; i < layers; i++) {
            b.add(block);
        }
        return b.layer(LayerNorm.of(16)).dense(V, Activation.SOFTMAX).seed(3).build();
    }

    private static float[] full(Model model, int[] ids) {
        int length = model.inputShape().dim(0);
        float[] x = new float[length];
        for (int i = 0; i < ids.length; i++) {
            x[i] = ids[i];
        }
        return model.predict(Tensor.of(Shape.of(1, length), x)).toArray();
    }

    private static float[] row(float[] out, int position) {
        float[] r = new float[V];
        System.arraycopy(out, position * V, r, 0, V);
        return r;
    }

    @Test
    void cachedStepsMatchTheFullPass() {
        TransformerBlock.Spec[] blocks = {
            TransformerBlock.of(4).causal(true).rotary(10000),
            TransformerBlock.of(4).causal(true).kvHeads(2).gated(true).rmsNorm(true),
            TransformerBlock.of(2).causal(true).alibi(true).preNorm(false),
        };
        int[] ids = {1, 5, 2, 9, 9, 0, 3, 7, 4, 10};
        for (TransformerBlock.Spec spec : blocks) {
            Model model = lm(12, 2, spec);
            float[] expected = full(model, ids);
            KvCache cache = KvCache.of(model);
            // a prompt of three, then one token at a time
            assertArrayEquals(row(expected, 2), cache.append(ids[0], ids[1], ids[2]).toArray(), 1e-5f);
            for (int t = 3; t < ids.length; t++) {
                assertArrayEquals(row(expected, t), cache.append(ids[t]).toArray(), 1e-5f, "step " + t);
            }
            assertEquals(ids.length, cache.length());
            cache.reset();
            assertArrayEquals(row(expected, 0), cache.append(ids[0]).toArray(), 1e-5f);
        }
    }

    @Test
    void streamsRunTogetherAsTheyWouldAlone() {
        int[][] ids = {{1, 5, 2, 9, 9, 0, 3, 7, 4, 10, 6, 2}, {4, 4, 8, 1, 0, 2, 2, 5, 9, 3, 1, 7},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0}};
        for (KvCache.Strategy strategy : new KvCache.Strategy[] {KvCache.Strategy.restart(),
            KvCache.Strategy.sliding(), KvCache.Strategy.sinks(2)}) {
            Model model = lm(6, 2, TransformerBlock.of(4).causal(true).kvHeads(2).rotary(10000));
            KvCache together = KvCache.of(model, strategy);
            KvCache[] alone = new KvCache[ids.length];
            for (int b = 0; b < ids.length; b++) {
                alone[b] = KvCache.of(model, strategy);
            }
            for (int t = 0; t < ids[0].length; t += t == 0 ? 3 : 1) {
                int to = t == 0 ? 3 : t + 1;
                int[][] step = new int[ids.length][];
                for (int b = 0; b < ids.length; b++) {
                    step[b] = java.util.Arrays.copyOfRange(ids[b], t, to);
                }
                float[] out = together.append(step).toArray();
                for (int b = 0; b < ids.length; b++) {
                    assertArrayEquals(alone[b].append(step[b]).toArray(), row(out, b), 1e-5f,
                            strategy + ", stream " + b + ", step " + t);
                }
            }
        }
    }

    @Test
    void aFullWindowStartsOverFromItsLastHalf() {
        Model model = lm(4, 2, TransformerBlock.of(2).causal(true).rotary(10000));
        KvCache cache = KvCache.of(model);
        float[] last = null;
        // 3 1 4 1 | 5 restarts from "4 1 5" | 9 fills it | 2 restarts from "5 9 2" | 6 fills it
        for (int id : new int[] {3, 1, 4, 1, 5, 9, 2, 6}) {
            last = cache.append(id).toArray();
        }
        assertArrayEquals(row(full(model, new int[] {5, 9, 2, 6}), 3), last, 1e-5f);
        assertEquals(8, cache.length());
        // a long prompt keeps its last window
        cache.reset();
        assertArrayEquals(row(full(model, new int[] {9, 2, 6, 5}), 3),
                cache.append(3, 1, 4, 1, 5, 9, 2, 6, 5).toArray(), 1e-5f);
    }

    @Test
    void slidingAndSinksKeepWhatTheyPromise() {
        // with one block, the keys of old tokens depend on nothing further back, so the cache
        // equals a fresh pass over exactly the tokens it kept
        Model model = lm(4, 1, TransformerBlock.of(2).causal(true).rotary(10000));
        int[] ids = {3, 1, 4, 1, 5, 9, 2, 6};
        KvCache sliding = KvCache.of(model, KvCache.Strategy.sliding());
        KvCache sinks = KvCache.of(model, KvCache.Strategy.sinks(1));
        float[] a = null;
        float[] b = null;
        for (int id : ids) {
            a = sliding.append(id).toArray();
            b = sinks.append(id).toArray();
        }
        assertEquals(4, sliding.size());
        assertArrayEquals(row(full(model, new int[] {5, 9, 2, 6}), 3), a, 1e-5f);
        assertArrayEquals(row(full(model, new int[] {3, 9, 2, 6}), 3), b, 1e-5f);
        // a long prompt in one call gives the same as token by token
        KvCache once = KvCache.of(model, KvCache.Strategy.sinks(1));
        assertArrayEquals(b, once.append(ids).toArray(), 1e-5f);
    }

    /** A policy of one's own: here the sinks strategy again, so the two must agree. */
    record FirstAndLast(int first) implements KvCache.Strategy.Policy {
        @Override
        public int[] keep(int total, int fresh, int window, float[] received, Tensor keys) {
            assertEquals(total, keys.shape().dim(2));
            assertEquals(total, received.length);
            int[] kept = new int[window];
            for (int i = 0; i < window; i++) {
                kept[i] = i < first ? i : total - window + i;
            }
            return kept;
        }

        @Override
        public boolean tracksAttention() {
            return true;
        }
    }

    @Test
    void policiesOfOnesOwnPlugIn() {
        Model model = lm(4, 1, TransformerBlock.of(2).causal(true).rotary(10000));
        KvCache sinks = KvCache.of(model, KvCache.Strategy.sinks(1));
        KvCache own = KvCache.of(model, new FirstAndLast(1));
        for (int id : new int[] {3, 1, 4, 1, 5, 9, 2, 6}) {
            assertArrayEquals(sinks.append(id).toArray(), own.append(id).toArray(), 1e-6f);
        }
        assertEquals(4, own.size());
    }

    @Test
    void heavyHittersKeepTheWindowFullAndTheRecentTokens() {
        Model model = lm(6, 2, TransformerBlock.of(2).causal(true).rotary(10000));
        KvCache cache = KvCache.of(model, KvCache.Strategy.heavyHitters(3));
        KvCache bf16 = KvCache.of(model, 6, KvCache.Strategy.restart(), io.github.zzuegg.brainiac.tensor.DType.BFLOAT16);
        KvCache f32 = KvCache.of(model);
        for (int i = 0; i < 20; i++) {
            float[] p = cache.append(i % V).toArray();
            assertEquals(1, sum(p), 1e-4f);
            assertTrue(cache.size() <= 6);
            assertArrayEquals(f32.append(i % V).toArray(), bf16.append(i % V).toArray(), 2e-2f);
        }
        assertEquals(6, cache.size());
        assertEquals(KvCache.Strategy.heavyHitters(3), KvCache.Strategy.parse("h2o:3"));
        assertEquals(KvCache.Strategy.sinks(4), KvCache.Strategy.parse("sinks"));
    }

    private static float sum(float[] p) {
        float s = 0;
        for (float x : p) {
            s += x;
        }
        return s;
    }

    @Test
    void onlyCausalChainsCanBeCached() {
        assertThrows(IllegalArgumentException.class, () -> KvCache.of(lm(4, 1, TransformerBlock.of(2))));
        Model pooled = Model.sequential().input(4).add(Embedding.of(V, 8))
                .add(TransformerBlock.of(2).causal(true)).flatten().dense(V, Activation.SOFTMAX).build();
        assertThrows(IllegalArgumentException.class, () -> KvCache.of(pooled));
    }
}
