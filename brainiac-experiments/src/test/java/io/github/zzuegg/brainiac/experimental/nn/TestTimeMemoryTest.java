package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Titans memory, the Titans layer and the self-referential weight matrix: memories that learn while running. */
class TestTimeMemoryTest {

    private static Tensor unitRows(int n, int d, SplittableRandom r) {
        Tensor x = Tensor.normal(Shape.of(n, d), 0, 1, r);
        return x.div(x.square().sum(1, true).sqrt());
    }

    private static float error(TitansMemory m, Tensor k, Tensor v) {
        return m.read(k).sub(v).square().mean().item();
    }

    @Test
    void titansMemoryMemorizesAStreamAtTestTime() {
        SplittableRandom r = new SplittableRandom(1);
        Tensor keys = unitRows(6, 32, r);
        Tensor values = Tensor.normal(Shape.of(6, 8), 0, 1, r);
        for (int hidden : new int[] {0, 32}) {
            TitansMemory m = new TitansMemory(32, hidden, 8, new SplittableRandom(2))
                    .gates(hidden == 0 ? 0.4f : 0.1f, 0.3f, 0);
            float before = error(m, keys, values);
            for (int pass = 0; pass < 25; pass++) {
                for (int i = 0; i < 6; i++) {
                    m.write(keys.slice(0, i, i + 1), values.slice(0, i, i + 1));
                }
            }
            float after = error(m, keys, values);
            assertTrue(after < 0.1f * before, hidden + " hidden: " + before + " -> " + after);
        }
    }

    @Test
    void surprisingPairsChangeTheMemoryMoreThanKnownOnes() {
        SplittableRandom r = new SplittableRandom(3);
        Tensor keys = unitRows(2, 32, r);
        Tensor values = Tensor.normal(Shape.of(2, 8), 0, 1, r);
        TitansMemory m = new TitansMemory(32, 0, 8, new SplittableRandom(2)).gates(0.6f, 0, 0);
        for (int i = 0; i < 10; i++) {
            m.write(keys.slice(0, 0, 1), values.slice(0, 0, 1));
        }
        m.write(keys.slice(0, 0, 1), values.slice(0, 0, 1));
        float known = m.lastSurprise();
        m.write(keys.slice(0, 1, 2), values.slice(0, 1, 2));
        float novel = m.lastSurprise();
        assertTrue(novel > 10 * known, "novel " + novel + " known " + known);
    }

    @Test
    void forgettingFadesOldMemories() {
        SplittableRandom r = new SplittableRandom(4);
        Tensor keys = unitRows(11, 128, r);
        Tensor values = Tensor.normal(Shape.of(11, 8), 0, 1, r);
        float[] errs = new float[2];
        float[] alphas = {0, 0.2f};
        for (int a = 0; a < 2; a++) {
            TitansMemory m = new TitansMemory(128, 0, 8, new SplittableRandom(2)).gates(0.4f, 0, alphas[a]);
            for (int pass = 0; pass < 3; pass++) {
                m.write(keys.slice(0, 0, 1), values.slice(0, 0, 1));
            }
            for (int i = 1; i < 11; i++) {
                m.write(keys.slice(0, i, i + 1), values.slice(0, i, i + 1));
            }
            errs[a] = error(m, keys.slice(0, 0, 1), values.slice(0, 0, 1));
        }
        assertTrue(errs[1] > 2 * errs[0], "forgetting " + errs[1] + " vs keeping " + errs[0]);
    }

    @Test
    void titansLayerLearnsInContextRecall() {
        double acc = FastWeightsTest.recallAccuracy(new TitansLayer(8, 1, 8, new SplittableRandom(1)), 8, 400, 0.02f);
        assertTrue(acc > 0.85, "titans layer " + acc);
    }

    @Test
    void selfReferentialMatrixLearnsInContextRecallAndRewritesItself() {
        SelfReferentialWeightMatrix srwm = new SelfReferentialWeightMatrix(8, 8, 1, new SplittableRandom(1));
        double acc = FastWeightsTest.recallAccuracy(srwm, 8, 600, 0.02f);
        assertTrue(acc > 0.6, "srwm " + acc);
        Tensor w0 = srwm.parameters().get(0).value();
        float change = srwm.lastWeights().slice(0, 0, 1).sub(w0).abs().sum().item();
        assertTrue(change > 1e-2, "the matrix never changed");
    }
}
