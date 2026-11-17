package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MaskedDiffusionTest {

    static Model model() {
        return Model.sequential().input(4).embedding(5, 16).positionalEncoding()
                .add(TransformerBlock.of(2)).dense(5).seed(1).build();
    }

    @Test
    void infillsAroundAPromptAndUnmasksOnSchedule() {
        MaskedDiffusion md = MaskedDiffusion.of(model(), 5, 4);
        List<int[][]> trace = new ArrayList<>();
        int[][] out = md.sample(3, 4, 2, new int[] {1, -1, -1, 2}, Sampling.GREEDY, new SplittableRandom(1), trace);
        assertEquals(2, trace.size());
        for (int[] row : out) {
            assertEquals(1, row[0]);
            assertEquals(2, row[3]);
            for (int t : row) {
                assertTrue(t != 4, "nothing left masked");
            }
        }
        // after the first of two steps, one of the two free positions is revealed
        long masked = java.util.Arrays.stream(trace.get(0)[0]).filter(t -> t == 4).count();
        assertEquals(1, masked);
    }

    @Test
    void learnsAFixedPatternFromMaskedTokens() {
        MaskedDiffusion md = MaskedDiffusion.of(model(), 5, 4);
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(2);
        int[][] data = {{0, 1, 2, 3}, {3, 2, 1, 0}};
        for (int s = 0; s < 400; s++) {
            Var loss = md.loss(new int[][] {data[s % 2], data[(s + 1) % 2]}, r);
            loss.backward();
            opt.step(md.model().parameters());
        }
        // given the first token the rest is determined
        assertArrayEquals(data[1], md.sample(1, 4, 3, new int[] {3, -1, -1, -1}, Sampling.GREEDY, r, null)[0]);
        assertArrayEquals(data[0], md.sample(1, 4, 3, new int[] {0, -1, -1, -1}, Sampling.GREEDY, r, null)[0]);
    }
}
