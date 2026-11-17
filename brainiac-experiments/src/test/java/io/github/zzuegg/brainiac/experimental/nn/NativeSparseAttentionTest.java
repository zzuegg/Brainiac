package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class NativeSparseAttentionTest {

    @Test
    void isCausalAndSelectsAtMostTopBlocksIncludingTheCurrentOne() {
        NativeSparseAttention nsa = new NativeSparseAttention(8, 2, 2, 3, 2, new SplittableRandom(1));
        float[] data = Tensor.uniform(Shape.of(1, 9, 8), -1, 1, new SplittableRandom(2)).toArray();
        Var x = Var.of(Tensor.of(Shape.of(1, 9, 8), data));
        float[] y = nsa.forward(x).value().toArray();
        float[] changed = data.clone();
        for (int i = 6 * 8; i < data.length; i++) {
            changed[i] += 1;
        }
        float[] y2 = nsa.forward(Var.of(Tensor.of(Shape.of(1, 9, 8), changed))).value().toArray();
        float[] early = java.util.Arrays.copyOf(y, 6 * 8);
        assertArrayEquals(early, java.util.Arrays.copyOf(y2, 6 * 8), 1e-5f);
        int[][][] sel = nsa.selectedBlocks(x);
        for (int t = 0; t < 9; t++) {
            assertTrue(sel[0][t].length <= 3);
            assertEquals(t / 2, sel[0][t][sel[0][t].length - 1]);
        }
    }

    @Test
    void learnsRecallThroughBlockSelection() {
        NativeSparseAttention nsa = new NativeSparseAttention(16, 2, 2, 3, 2, new SplittableRandom(3));
        double acc = MixerHarness.train(nsa, MixerHarness.recall(8, 6, 5), 16, 300, 0.02f, 4);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }
}
