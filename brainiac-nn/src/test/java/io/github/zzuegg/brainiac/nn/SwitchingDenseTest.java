package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Each unit's inputs choose its activation, softly or (in hard mode) outright. */
class SwitchingDenseTest {

    @TempDir
    Path dir;

    /** One input, one unit: z = x, control c = x, offsets -1 (tanh) and +1 (relu). */
    private static SwitchingDense oneUnit(boolean hard) {
        return new SwitchingDense(1, 1, new Activation[] {Activation.TANH, Activation.RELU}, hard,
                Tensor.of(Shape.of(1, 1), 1f), Tensor.zeros(Shape.of(1)),
                Tensor.of(Shape.of(1, 1), 1f), Tensor.zeros(Shape.of(1)),
                Tensor.of(Shape.of(2, 1), -1f, 1f), Tensor.zeros(Shape.of(2, 1)));
    }

    @Test
    void theControlValueBlendsTheCandidates() {
        SwitchingDense layer = oneUnit(false);
        float[] xs = {-3f, -0.5f, 0f, 0.5f, 3f};
        float[] y = layer.forward(Var.of(Tensor.of(Shape.of(5, 1), xs))).value().toArray();
        for (int i = 0; i < xs.length; i++) {
            double x = xs[i];
            // softmax(-x, x): the share of relu is sigmoid(2x)
            double relu = 1 / (1 + Math.exp(-2 * x));
            double expected = (1 - relu) * Math.tanh(x) + relu * Math.max(0, x);
            assertEquals(expected, y[i], 1e-5, "x = " + x);
        }
        Tensor shares = layer.shares(Tensor.of(Shape.of(5, 1), xs));
        assertEquals(Shape.of(2, 5, 1), shares.shape());
        float[] s = shares.toArray();
        for (int i = 0; i < xs.length; i++) {
            assertEquals(1f, s[i] + s[5 + i], 1e-6);
        }
        assertTrue(s[0] > 0.99f && s[9] > 0.99f, "low control picks tanh, high control relu");
    }

    @Test
    void hardModeUsesOnlyTheLikeliestCandidate() {
        SwitchingDense layer = oneUnit(true);
        float[] xs = {-3f, -0.5f, 0.5f, 3f};
        float[] y = layer.forward(Var.of(Tensor.of(Shape.of(4, 1), xs))).value().toArray();
        assertArrayEquals(new float[] {(float) Math.tanh(-3), (float) Math.tanh(-0.5), 0.5f, 3f}, y, 1e-6f);
    }

    @Test
    void gradientsMatchFiniteDifferences() {
        SwitchingDense layer = (SwitchingDense) SwitchingDense.of(5, Activation.TANH, Activation.SIGMOID,
                Activation.SOFTPLUS).create(Shape.of(4), new SplittableRandom(3));
        Tensor x = Tensor.uniform(Shape.of(3, 4), -1.5f, 1.5f, new SplittableRandom(4));
        Tensor w = Tensor.uniform(Shape.of(3, 5), -1, 1, new SplittableRandom(5));
        GradCheck.Result r = GradCheck.check(v -> layer.forward(v.get(0)).mul(w).sum(), List.of(x), 1e-3f, 2e-2f);
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void savesAndLoads() throws IOException {
        Model m = Model.sequential().input(3)
                .add(SwitchingDense.of(6, Activation.RELU, Activation.TANH, Activation.SILU))
                .add(SwitchingDense.of(2, Activation.IDENTITY, Activation.SIGMOID).hard(true))
                .seed(7).build();
        Path file = dir.resolve("m.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        Tensor in = Tensor.uniform(Shape.of(4, 3), -2, 2, new SplittableRandom(8));
        assertArrayEquals(m.predict(in).toArray(), loaded.predict(in).toArray());
        assertTrue(((SwitchingDense) loaded.layers().get(1)).hard());
    }

    @Test
    void needsTwoCandidates() {
        assertThrows(IllegalArgumentException.class, () -> SwitchingDense.of(3, Activation.RELU));
    }
}
