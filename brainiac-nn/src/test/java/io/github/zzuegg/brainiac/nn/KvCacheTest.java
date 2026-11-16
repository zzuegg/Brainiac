package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
    void onlyCausalChainsCanBeCached() {
        assertThrows(IllegalArgumentException.class, () -> KvCache.of(lm(4, 1, TransformerBlock.of(2))));
        Model pooled = Model.sequential().input(4).add(Embedding.of(V, 8))
                .add(TransformerBlock.of(2).causal(true)).flatten().dense(V, Activation.SOFTMAX).build();
        assertThrows(IllegalArgumentException.class, () -> KvCache.of(pooled));
    }
}
