package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MultiTokenPredictorTest {

    /** Sequences that count up by a per-sequence stride of 1 or 2, modulo 12. */
    private static int[][] batch(SplittableRandom r, int size, int length) {
        int[][] b = new int[size][length];
        for (int i = 0; i < size; i++) {
            int stride = 1 + r.nextInt(2);
            b[i][0] = r.nextInt(12);
            for (int t = 1; t < length; t++) {
                b[i][t] = (b[i][t - 1] + stride) % 12;
            }
        }
        return b;
    }

    @Test
    void speculativeDecodingMatchesGreedyWithFewerPasses() {
        SplittableRandom r = new SplittableRandom(1);
        MultiTokenPredictor model = new MultiTokenPredictor(12, 16, 3,
                new GatedAttention(16, 2, true, 10000f, r), r);
        Optimizer opt = Optimizer.adam(0.02f);
        Var first = null;
        Var last = null;
        for (int step = 0; step < 150; step++) {
            Var loss = model.loss(batch(r, 16, 10));
            first = first == null ? loss : first;
            last = loss;
            loss.backward();
            opt.step(model.parameters());
        }
        assertTrue(last.value().item() < first.value().item() * 0.3, "loss " + first + " -> " + last);
        int[] prompt = {3, 5, 7};
        int[] greedy = model.generateGreedy(prompt, 9);
        MultiTokenPredictor.Decoded fast = model.generateSpeculative(prompt, 9);
        assertArrayEquals(greedy, fast.tokens());
        assertArrayEquals(new int[] {9, 11, 1, 3, 5, 7, 9, 11, 1}, greedy);
        assertTrue(fast.tokensPerPass() > 1.5, "tokens per pass " + fast.tokensPerPass());
    }
}
