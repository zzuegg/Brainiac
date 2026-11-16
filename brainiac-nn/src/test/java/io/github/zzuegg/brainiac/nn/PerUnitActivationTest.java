package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
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

class PerUnitActivationTest {

    private static final Tensor X = Tensor.of(Shape.of(2, 4), -2.5f, -0.7f, 0.4f, 1.9f, 1.1f, -1.3f, -0.2f, 2.6f);

    @TempDir
    Path dir;

    @Test
    void everyUnitGetsItsOwnActivation() {
        Activation a = Activation.perUnit(Activation.TANH, Activation.RELU, Activation.TANH, Activation.SIGMOID);
        float[] y = a.forward(Var.of(X)).value().toArray();
        float[] x = X.toArray();
        for (int i = 0; i < x.length; i++) {
            double expected = switch (i % 4) {
                case 0, 2 -> Math.tanh(x[i]);
                case 1 -> Math.max(0, x[i]);
                default -> 1 / (1 + Math.exp(-x[i]));
            };
            assertEquals(expected, y[i], 1e-6, "element " + i);
        }
        assertEquals("perUnit(1 relu, 1 sigmoid, 2 tanh)", a.toString());   // a fixed order, not first use
        assertEquals("perUnit(1 relu, 1 sigmoid, 2 tanh)",
                Activation.perUnit(Activation.SIGMOID, Activation.TANH, Activation.RELU, Activation.TANH).toString());
        assertThrows(IllegalArgumentException.class, () -> a.forward(Var.of(Tensor.zeros(Shape.of(2, 3)))));
    }

    @Test
    void oneSharedActivationIsJustThatActivation() {
        assertSame(Activation.RELU, Activation.perUnit(Activation.RELU, Activation.RELU));
        assertThrows(IllegalArgumentException.class, () -> Activation.perUnit(Activation.prelu()));
    }

    @Test
    void gradientsAreRight() {
        Activation a = Activation.perUnit(Activation.SOFTSIGN, Activation.TANH, Activation.SILU, Activation.SIGMOID);
        GradCheck.Result r = GradCheck.check(v -> a.forward(v.get(0)).square().sum(), List.of(X));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void aPaletteGivesEveryMixTheSameComputation() {
        List<Activation> palette = List.of(Activation.RELU, Activation.TANH, Activation.SIGMOID, Activation.SILU);
        Activation a = Activation.perUnit(palette, Activation.TANH, Activation.RELU, Activation.TANH, Activation.RELU);
        Activation b = Activation.perUnit(palette, Activation.SILU, Activation.SIGMOID, Activation.SILU,
                Activation.SILU);
        // the same activations in the same order: only the masks differ
        assertEquals(a.config().getInt("count"), b.config().getInt("count"));
        for (int i = 0; i < 4; i++) {
            assertEquals(a.config().getActivation("a" + i), b.config().getActivation("a" + i));
        }
        assertEquals("perUnit(2 relu, 2 tanh)", a.toString());
        Activation plain = Activation.perUnit(Activation.TANH, Activation.RELU, Activation.TANH, Activation.RELU);
        assertArrayEquals(plain.forward(Var.of(X)).value().toArray(), a.forward(Var.of(X)).value().toArray(), 1e-6f);
        assertSame(Activation.TANH, Activation.perUnit(palette, Activation.TANH, Activation.TANH));
    }

    @Test
    void savesAndLoads() throws IOException {
        Model m = Model.sequential().input(4)
                .dense(5, Activation.perUnit(List.of(Activation.SIGMOID), Activation.RELU, Activation.TANH,
                        Activation.GELU, Activation.TANH, Activation.IDENTITY))
                .dense(2).seed(3).build();
        SplittableRandom r = new SplittableRandom(4);
        for (Var p : m.parameters()) {
            p.assign(Tensor.uniform(p.shape(), -0.9f, 0.9f, r));
        }
        Path file = dir.resolve("per-unit.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray(), 1e-6f);
    }

    @Test
    void brainTakesExactWiring() {
        Brain.Spec spec = Brain.builder()
                .region("a", 3).activation(Activation.perUnit(Activation.RELU, Activation.TANH, Activation.RELU))
                .connect("input", "a", 0.5)
                .build();
        Model m = Model.sequential().input(2).add(spec).dense(1).seed(1).build();
        Brain brain = (Brain) m.layers().get(0);
        boolean[] mask = {true, false, false, false, true, true};
        brain.setConnections(0, mask, new SplittableRandom(1));
        assertArrayEquals(mask, brain.connectionMask(0));
        float[] w = brain.weightValues(0);
        for (int k = 0; k < mask.length; k++) {
            assertEquals(mask[k], w[k] != 0, "weight " + k);
        }
        brain.setWeightValues(0, new float[] {1, 2, 3, 4, 5, 6});
        assertArrayEquals(new float[] {1, 0, 0, 0, 5, 6}, brain.weightValues(0));
        brain.setBiasValues("a", new float[] {0.5f, 0, 0});
        assertArrayEquals(new float[] {0.5f, 0, 0}, brain.biasValues("a"));
        assertThrows(IllegalArgumentException.class, () -> brain.setConnections(0, new boolean[5],
                new SplittableRandom(1)));
    }
}
