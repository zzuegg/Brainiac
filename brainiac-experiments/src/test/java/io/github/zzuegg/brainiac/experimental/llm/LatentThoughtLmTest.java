package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

class LatentThoughtLmTest {

    @Test
    void learnsToCopyThroughThoughts() {
        LatentThoughtLm lm = LatentThoughtLm.of(5, 16, 1);
        int[][] prefix = {{1, 2}, {3, 0}, {2, 2}};
        int[][] targets = {{2, 1, 4}, {0, 3, 4}, {2, 2, 4}};
        Optimizer opt = Optimizer.adam(1e-2f);
        float first = 0;
        float last = 0;
        for (int s = 0; s < 200; s++) {
            Var loss = lm.loss(prefix, 2, targets);
            if (s == 0) {
                first = loss.value().item();
            }
            last = loss.value().item();
            loss.backward();
            opt.step(lm.parameters());
        }
        System.out.println("latent copy loss " + first + " -> " + last);
        assertTrue(last < 0.05 * first);
        assertArrayEquals(new int[] {0, 3, 4}, lm.generate(new int[] {3, 0}, 2, 5, 4));
    }
}
