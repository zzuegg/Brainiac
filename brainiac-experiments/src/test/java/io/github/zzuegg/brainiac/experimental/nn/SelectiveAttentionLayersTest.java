package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SelectiveAttentionLayersTest {

    private static Var input(int b, int t, int d, long seed) {
        return Var.of(Tensor.normal(Shape.of(b, t, d), 0, 1, new SplittableRandom(seed)));
    }

    /** Changing the last step must not change any earlier output. */
    private static void assertCausal(Layer layer, int t, int d) {
        Var x = input(2, t, d, 42);
        float[] y = layer.forward(x).value().toArray();
        float[] changed = x.value().toArray();
        for (int n = 0; n < 2; n++) {
            for (int c = 0; c < d; c++) {
                changed[(n * t + t - 1) * d + c] += 3;
            }
        }
        float[] y2 = layer.forward(Var.of(Tensor.of(Shape.of(2, t, d), changed))).value().toArray();
        for (int n = 0; n < 2; n++) {
            for (int i = 0; i < (t - 1) * d; i++) {
                assertEquals(y[n * t * d + i], y2[n * t * d + i], 1e-4f, layer + " leaks the future");
            }
        }
    }

    @Test
    void everyNewMixerIsCausal() {
        SplittableRandom r = new SplittableRandom(1);
        for (Layer layer : List.of(new IndexedSparseAttention(8, 2, 3, 0, 1f, 10000f, r),
                new BlockRoutedAttention(8, 2, 2, 2, 10000f, r),
                new CompressedBlockAttention(8, 2, 4, CompressedBlockAttention.Mode.CSA, 2, 2, 2, r),
                new CompressedBlockAttention(8, 2, 4, CompressedBlockAttention.Mode.HCA, 4, 0, 2, r),
                new ContextualPositionAttention(8, 2, 8, true, 10000f, r),
                new MultiTokenAttention(8, 2, 2, 5, true, 10000f, r),
                new PathPositionAttention(8, 2, false, r),
                new KimiDeltaAttention(8, 2, 2, 3, r),
                new DeltaProductLayer(8, 2, 2, true, 2, r),
                new LogLinearAttention(8, 2, 16, true, r))) {
            assertCausal(layer, 9, 8);
        }
    }

    @Test
    void indexedSparseAttentionWarmsUpDenseThenSelectsTopK() {
        IndexedSparseAttention dsa = new IndexedSparseAttention(8, 2, 3, 2, 1f, 10000f, new SplittableRandom(2));
        Var x = input(2, 10, 8, 3);
        dsa.forward(x, true);
        assertEquals(5.5, dsa.lastAttended(), 1e-9);                // dense causal: (1 + ... + 10) / 10
        assertNotNull(dsa.auxiliaryLoss());
        assertTrue(dsa.auxiliaryLoss().value().item() >= -1e-6, "KL is non-negative");
        dsa.forward(x, true);
        dsa.forward(x, true);                                         // past warm-up: sparse
        assertEquals((1 + 2 + 3 * 8) / 10.0, dsa.lastAttended(), 1e-9);
        dsa.auxiliaryLoss().backward();
        for (Var p : dsa.indexer().parameters()) {
            assertTrue(p.grad().abs().sum().item() > 0, "the distillation loss trains the indexer");
        }
        dsa.forward(x, false);
        assertNull(dsa.auxiliaryLoss());
        assertEquals(3, dsa.trainingPasses());
    }

    @Test
    void indexerDistillationLearnsToRankLikeAttention() {
        SplittableRandom r = new SplittableRandom(4);
        LightningIndexer indexer = new LightningIndexer(8, 2, 4, false, r);
        Var x = input(1, 6, 8, 5);
        // target: attend to the previous token
        float[] target = new float[36];
        for (int i = 0; i < 6; i++) {
            target[i * 6 + Math.max(0, i - 1)] = 1;
        }
        Tensor attention = Tensor.of(Shape.of(1, 1, 6, 6), target);
        Tensor keep = Tensor.of(Shape.of(6, 6), SparseSelectOps.causal(6));
        io.github.zzuegg.brainiac.train.Optimizer opt = io.github.zzuegg.brainiac.train.Optimizer.adam(0.05f);
        double first = 0;
        double last = 0;
        for (int s = 0; s < 150; s++) {
            Var loss = LightningIndexer.distillationLoss(indexer.scores(x, x), attention, keep);
            if (s == 0) {
                first = loss.value().item();
            }
            last = loss.value().item();
            loss.backward();
            opt.step(indexer.parameters());
        }
        assertTrue(last < first / 4, "KL " + first + " -> " + last);
    }

    @Test
    void blockRoutingAttendsToAFixedBudget() {
        BlockRoutedAttention moba = new BlockRoutedAttention(8, 2, 4, 2, 10000f, new SplittableRandom(6));
        moba.forward(input(1, 32, 8, 7));
        assertTrue(moba.lastAttended() <= 8);
        BlockRoutedAttention full = new BlockRoutedAttention(8, 2, 4, 8, 10000f, new SplittableRandom(6));
        full.forward(input(1, 32, 8, 7));
        assertEquals(16.5, full.lastAttended(), 1e-9);               // every block selected = dense causal
    }

    @Test
    void compressedAttentionCachesAFractionPerToken() {
        CompressedBlockAttention csa = new CompressedBlockAttention(16, 2, 8, CompressedBlockAttention.Mode.CSA, 4, 2,
                4, new SplittableRandom(8));
        assertEquals(2.0, csa.cachedPerToken(), 1e-9);
        csa.forward(input(2, 24, 16, 9), true);
        assertNotNull(csa.auxiliaryLoss());
        assertTrue(csa.lastAttended() <= 2 + 4);
        CompressedBlockAttention hca = new CompressedBlockAttention(16, 2, 8, CompressedBlockAttention.Mode.HCA, 8, 0,
                4, new SplittableRandom(8));
        assertEquals(1.0, hca.cachedPerToken(), 1e-9);
        hca.forward(input(2, 24, 16, 9), true);
        assertNull(hca.auxiliaryLoss());
    }

    @Test
    void contextualPositionsCountGates() {
        ContextualPositionAttention cope = new ContextualPositionAttention(8, 2, 8, false, 0f, new SplittableRandom(10));
        Tensor p = cope.contextualPositions(input(1, 5, 8, 11));
        float[] v = p.toArray();
        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 1; j <= i; j++) {
                    int base = (h * 5 + i) * 5;
                    assertTrue(v[base + j - 1] >= v[base + j], "positions grow towards the past");
                }
                assertTrue(v[(h * 5 + i) * 5 + i] > 0 && v[(h * 5 + i) * 5 + i] < 1);
            }
        }
    }

    @Test
    void multiTokenConvolutionStartsNearPlainAttention() {
        MultiTokenAttention mta = new MultiTokenAttention(8, 2, 2, 3, false, 0f, new SplittableRandom(12));
        Var logits = Var.of(Tensor.normal(Shape.of(1, 2, 5, 5), 0, 1, new SplittableRandom(13)));
        float[] a = logits.value().toArray();
        float[] c = mta.convolve(logits).value().toArray();
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], c[i], 0.1f);
        }
    }
}
