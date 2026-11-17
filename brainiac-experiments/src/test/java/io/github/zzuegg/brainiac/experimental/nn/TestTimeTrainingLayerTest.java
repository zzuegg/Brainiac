package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TestTimeTrainingLayerTest {

    @Test
    void innerModelsLearnDuringTheForwardPass() {
        for (TestTimeTrainingLayer.Variant variant : TestTimeTrainingLayer.Variant.values()) {
            TestTimeTrainingLayer ttt = new TestTimeTrainingLayer(8, 1, variant, 4,
                    variant == TestTimeTrainingLayer.Variant.LINEAR ? 0.3f : 0.1f, new SplittableRandom(1));
            // the same four tokens again and again: the inner model memorizes them as it reads
            Tensor block = Tensor.normal(Shape.of(2, 4, 8), 0, 1, new SplittableRandom(2));
            Tensor[] copies = new Tensor[12];
            java.util.Arrays.fill(copies, block);
            ttt.forward(Var.of(Tensor.concat(1, copies)));
            float[] l = ttt.innerLosses();
            assertTrue(l[l.length - 1] < 0.5 * l[0], variant + ": " + java.util.Arrays.toString(l));
        }
    }

    @Test
    void dualFormEqualsOnlineUpdatesForOneTokenMiniBatches() {
        // with mini-batch 1, the dual form must reduce to plain per-token gradient steps; check
        // causality and that outer gradients through the inner updates are correct
        for (TestTimeTrainingLayer.Variant variant : TestTimeTrainingLayer.Variant.values()) {
            TestTimeTrainingLayer online = new TestTimeTrainingLayer(4, 1, variant, 1, 0.5f, new SplittableRandom(3));
            TestTimeTrainingLayer chunked = new TestTimeTrainingLayer(4, 1, variant, 3, 0.5f, new SplittableRandom(3));
            Tensor x = Tensor.uniform(Shape.of(1, 6, 4), -1, 1, new SplittableRandom(4));
            float[] a = online.forward(Var.of(x)).value().toArray();
            float[] b = chunked.forward(Var.of(x)).value().toArray();
            // the first token sees identical weights in both
            assertArrayEquals(java.util.Arrays.copyOf(a, 4), java.util.Arrays.copyOf(b, 4), 1e-5f);
            Tensor w = Tensor.uniform(Shape.of(1, 6, 4), 0.5f, 1.5f, new SplittableRandom(5));
            GradCheck.Result r = GradCheck.check(v -> chunked.forward(v.get(0)).mul(w).sum(), List.of(x));
            assertTrue(r.passed(), variant + ": " + r);
        }
    }

    @Test
    void tttLinearLearnsRecall() {
        double acc = MixerHarness.train(new TestTimeTrainingLayer(16, 2, TestTimeTrainingLayer.Variant.LINEAR, 4,
                0.5f, new SplittableRandom(6)), MixerHarness.recall(6, 6, 3), 16, 300, 0.01f, 7);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }

    @Test
    void tttMlpLearnsRecall() {
        double acc = MixerHarness.train(new TestTimeTrainingLayer(16, 2, TestTimeTrainingLayer.Variant.MLP, 4,
                0.5f, new SplittableRandom(8)), MixerHarness.recall(6, 6, 3), 16, 300, 0.01f, 9);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }
}
