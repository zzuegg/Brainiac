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

class CompositeActivationTest {

    private static final Tensor X = Tensor.of(Shape.of(2, 4), -2.5f, -0.7f, 0.4f, 1.9f, 1.1f, -1.3f, -0.2f, 2.6f);

    @TempDir
    Path dir;

    private static float[] apply(Activation a, Tensor x) {
        return a.forward(Var.of(x)).value().toArray();
    }

    @Test
    void splitAppliesEachActivationToItsGroup() {
        Activation split = Activation.split(Activation.RELU, Activation.TANH).bind(4);
        float[] relu = apply(Activation.RELU, X);
        float[] tanh = apply(Activation.TANH, X);
        float[] y = apply(split, X);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 4; c++) {
                int i = r * 4 + c;
                assertEquals(c < 2 ? relu[i] : tanh[i], y[i], 1e-7f, "element " + i);
            }
        }
        assertEquals("split(relu x2, tanh x2)", split.toString());
    }

    @Test
    void splitWithSizesAndPerUnitChoice() {
        Activation perUnit = Activation.split(new int[] {1, 1, 1, 1},
                Activation.IDENTITY, Activation.RELU, Activation.SIGMOID, Activation.HARD_TANH);
        float[] y = apply(perUnit.bind(4), X);
        assertEquals(-2.5f, y[0], 0f);
        assertEquals(0f, y[1], 0f);
        assertEquals(1 / (1 + Math.exp(-0.4)), y[2], 1e-6);
        assertEquals(1f, y[3], 0f);

        assertThrows(IllegalArgumentException.class,
                () -> Activation.split(new int[] {1, 2}, Activation.RELU, Activation.TANH).bind(4));
        assertThrows(IllegalArgumentException.class,
                () -> Activation.split(Activation.RELU, Activation.TANH, Activation.SILU).bind(2));
    }

    @Test
    void unevenSplitGivesTheFirstGroupsMoreUnits() {
        Activation split = Activation.split(Activation.prelu(), Activation.prelu()).bind(5);
        assertEquals(List.of(Shape.of(3), Shape.of(2)),
                split.parameters().stream().map(Var::shape).toList());
    }

    @Test
    void splitGroupsGetTheirOwnLearnableParameters() {
        Model m = Model.sequential().input(4).dense(6, Activation.split(Activation.prelu(), Activation.TANH))
                .dense(2).seed(1).build();
        assertEquals(4 * 6 + 6 + 3 + 6 * 2 + 2, m.parameterCount());
        assertTrue(m.summary().contains("split(prelu x3, tanh x3)"), m.summary());
        GradCheck.Result r = GradCheck.check(v -> m.forward(v.get(0)).square().sum(), List.of(X));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void mixStartsAsTheAverageAndLearns() {
        Activation mix = Activation.mix(Activation.RELU, Activation.TANH).bind(4);
        float[] relu = apply(Activation.RELU, X);
        float[] tanh = apply(Activation.TANH, X);
        float[] y = apply(mix, X);
        for (int i = 0; i < y.length; i++) {
            assertEquals((relu[i] + tanh[i]) / 2, y[i], 1e-6f);
        }
        Var logits = mix.parameters().get(0);
        assertEquals(Shape.of(2, 4), logits.shape());
        mix.forward(Var.of(X)).sum().backward();
        assertTrue(logits.grad().abs().sum().item() > 0);

        logits.assign(Tensor.of(Shape.of(2, 4), 3, -1, 0, 2, 0, 1, 0, -2));
        float[] w = ((MixedActivation) mix).weights().sum(0).toArray();
        assertArrayEquals(new float[] {1, 1, 1, 1}, w, 1e-6f);
        GradCheck.Result r = GradCheck.check(v -> mix.forward(v.get(0)).square().sum(), List.of(X));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void mixParameterGradientsMatchFiniteDifferences() {
        Activation mix = Activation.mix(Activation.SIGMOID, Activation.TANH, Activation.SILU).bind(4);
        Var logits = mix.parameters().get(0);
        Tensor start = Tensor.uniform(Shape.of(3, 4), -1, 1, new SplittableRandom(3));
        logits.assign(start);
        logits.zeroGrad();
        mix.forward(Var.of(X)).square().sum().backward();
        float[] analytic = logits.grad().toArray();
        float[] values = start.toArray();
        for (int i = 0; i < values.length; i++) {
            float h = 1e-2f;
            float[] p = values.clone();
            p[i] += h;
            logits.assign(Tensor.of(Shape.of(3, 4), p));
            float up = mix.forward(Var.of(X)).square().sum().value().item();
            p[i] -= 2 * h;
            logits.assign(Tensor.of(Shape.of(3, 4), p));
            float down = mix.forward(Var.of(X)).square().sum().value().item();
            assertEquals((up - down) / (2 * h), analytic[i], 2e-2, "logit " + i);
        }
    }

    @Test
    void nestedCompositesSaveAndLoad() throws IOException {
        Activation inner = Activation.mix(Activation.RELU, Activation.prelu());
        Node in = Node.input(4);
        Node h = in.dense(6, Activation.split(Activation.prelu(), Activation.TANH, inner))
                .activation(Activation.mix(Activation.SILU, Activation.elu(0.5f)));
        Model m = Model.graph(in, h.dense(2)).seed(5).build();
        SplittableRandom r = new SplittableRandom(6);
        for (Var p : m.parameters()) {
            p.assign(Tensor.uniform(p.shape(), -0.9f, 0.9f, r));
        }
        Path file = dir.resolve("composite.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertEquals(m.parameterCount(), loaded.parameterCount());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray(), 1e-6f);
    }

    @Test
    void bindingRules() {
        Activation bound = Activation.split(Activation.RELU, Activation.TANH).bind(4);
        assertSame(bound, bound.bind(4));
        assertThrows(IllegalArgumentException.class, () -> bound.bind(6));
        assertThrows(IllegalStateException.class, () -> Activation.mix(Activation.RELU, Activation.TANH)
                .forward(Var.of(X)));
        assertThrows(IllegalArgumentException.class, () -> Activation.mix(Activation.RELU));
    }
}
