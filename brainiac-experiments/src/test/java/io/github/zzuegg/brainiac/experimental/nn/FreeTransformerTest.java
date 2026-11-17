package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.HashSet;
import java.util.Set;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class FreeTransformerTest {

    /** BOS (0), then a run over 1..8 going up or down by one, wrapping. */
    private static int[][] batch(SplittableRandom r, int size) {
        int[][] b = new int[size][8];
        for (int i = 0; i < size; i++) {
            int dir = r.nextBoolean() ? 1 : -1;
            int v = r.nextInt(8);
            for (int t = 1; t < 8; t++) {
                b[i][t] = 1 + v;
                v = Math.floorMod(v + dir, 8);
            }
        }
        return b;
    }

    /** @return +1 or -1 if the run is consistent, else 0 */
    private static int direction(int[] run) {
        int d = Math.floorMod(run[1] - run[0], 8) == 1 ? 1 : Math.floorMod(run[0] - run[1], 8) == 1 ? -1 : 0;
        for (int i = 1; i < run.length; i++) {
            if (run[i] < 1 || Math.floorMod(run[i] - run[i - 1], 8) != Math.floorMod(d, 8)) {
                return 0;
            }
        }
        return d;
    }

    @Test
    void latentCodesChooseTheGlobalDirectionBeforeWriting() {
        SplittableRandom r = new SplittableRandom(1);
        FreeTransformer model = new FreeTransformer(9, 16, 2, 2, 0.3f, r);
        Optimizer opt = Optimizer.adam(0.01f);
        FreeTransformer.Losses losses = null;
        for (int step = 0; step < 300; step++) {
            losses = model.loss(batch(r, 16), r);
            losses.total().backward();
            opt.step(model.parameters());
        }
        assertTrue(losses.kl().value().item() >= 0);
        assertTrue(losses.kl().value().item() > 0.05, "latent unused: kl " + losses.kl());
        Set<Integer> directions = new HashSet<>();
        int valid = 0;
        for (int code = 0; code < model.codes(); code++) {
            int d = direction(model.generate(new int[] {0}, 6, code));
            valid += d != 0 ? 1 : 0;
            directions.add(d);
        }
        assertTrue(valid >= 3, "only " + valid + " codes produce valid runs");
        assertTrue(directions.contains(1) && directions.contains(-1), "codes do not control direction: " + directions);
    }
}
