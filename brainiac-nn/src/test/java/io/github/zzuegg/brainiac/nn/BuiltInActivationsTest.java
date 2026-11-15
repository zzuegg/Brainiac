package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class BuiltInActivationsTest {

    /** Spread over [-3.4, 6.8], away from every kink (0, +-1, +-3, 6). */
    private static final Tensor X = Tensor.of(Shape.of(3, 4),
            -3.4f, -2.7f, -1.6f, -0.45f, 0.3f, 0.8f, 1.3f, 2.2f, 3.6f, 4.9f, 6.8f, -0.85f);

    @TempDir
    Path dir;

    static Stream<Activation> all() {
        return Stream.of(Activation.IDENTITY, Activation.RELU, Activation.SIGMOID, Activation.TANH,
                Activation.SOFTMAX, Activation.LOG_SOFTMAX, Activation.SILU, Activation.GELU, Activation.GELU_TANH,
                Activation.SOFTPLUS, Activation.MISH, Activation.SELU, Activation.RELU6, Activation.HARD_TANH,
                Activation.HARD_SIGMOID, Activation.HARD_SWISH, Activation.SOFTSIGN, Activation.leakyRelu(0.1f),
                Activation.elu(0.7f), Activation.softmax(0), Activation.logSoftmax(0));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("all")
    void gradientMatchesFiniteDifferences(Activation a) {
        Tensor weights = Tensor.uniform(Shape.of(3, 4), -1, 1, new SplittableRandom(2));
        GradCheck.Result r = GradCheck.check(v -> a.forward(v.get(0)).mul(weights).sum(), List.of(X),
                1e-3f, 2e-2f);
        assertTrue(r.passed(), a + ": " + r);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("all")
    void savesAndLoads(Activation a) throws IOException {
        Model m = Model.sequential().input(4).dense(3, a).dense(2).activation(a).seed(3).build();
        Path file = dir.resolve("m.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        Tensor in = Tensor.uniform(Shape.of(5, 4), -2, 2, new SplittableRandom(4));
        assertArrayEquals(m.predict(in).toArray(), loaded.predict(in).toArray());
        Activation restored = ((Dense) loaded.layers().get(0)).activation();
        assertEquals(a.config(), restored.config());
    }

    private static float at(Activation a, float x) {
        return a.forward(Var.of(Tensor.of(Shape.of(1, 1), x))).value().item();
    }

    @Test
    void knownValues() {
        assertEquals(0.8413447f, at(Activation.GELU, 1), 1e-6f);
        assertEquals(0.841192f, at(Activation.GELU_TANH, 1), 1e-5f);
        assertEquals(0.7310586f, at(Activation.SILU, 1), 1e-6f);
        assertEquals(0.8650984f, at(Activation.MISH, 1), 1e-6f);
        assertEquals(-1.1113307f, at(Activation.SELU, -1), 1e-6f);
        assertEquals(1.0507010f, at(Activation.SELU, 1), 1e-6f);
        assertEquals(-0.6321206f, at(Activation.elu(), -1), 1e-6f);
        assertEquals(-0.02f, at(Activation.leakyRelu(), -2), 1e-7f);
        assertEquals(6f, at(Activation.RELU6, 9), 0f);
        assertEquals(-1f, at(Activation.HARD_TANH, -4), 0f);
        assertEquals(0.75f, at(Activation.HARD_SIGMOID, 1.5f), 1e-7f);
        assertEquals(0.5f, at(Activation.SOFTSIGN, 1), 1e-7f);
        assertEquals((float) Math.log(2), at(Activation.SOFTPLUS, 0), 1e-7f);
    }

    @Test
    void largeInputsStayFinite() {
        Var x = Var.param(Tensor.of(Shape.of(1, 4), -500f, -80f, 80f, 500f));
        for (Activation a : all().toList()) {
            x.zeroGrad();
            Var y = a.forward(x);
            y.sum().backward();
            for (float v : y.value().toArray()) {
                assertFalse(Float.isNaN(v), a + " gave NaN");
            }
            for (float g : x.grad().toArray()) {
                assertFalse(Float.isNaN(g), a + " gave a NaN gradient");
            }
        }
    }

    @Test
    void logSoftmaxIsTheLogOfSoftmax() {
        float[] sm = Activation.SOFTMAX.forward(Var.of(X)).value().toArray();
        float[] lsm = Activation.LOG_SOFTMAX.forward(Var.of(X)).value().toArray();
        for (int i = 0; i < sm.length; i++) {
            assertEquals(Math.log(sm[i]), lsm[i], 1e-5);
        }
        float[] cols = Activation.softmax(0).forward(Var.of(X)).value().sum(0).toArray();
        assertArrayEquals(new float[] {1, 1, 1, 1}, cols, 1e-6f);
    }

    @Test
    void activationsPickSuitableInit() {
        assertSame(Init.HE, Activation.RELU.preferredInit());
        assertSame(Init.HE, Activation.leakyRelu().preferredInit());
        assertSame(Init.HE, Activation.GELU.preferredInit());
        assertSame(Init.LECUN, Activation.SELU.preferredInit());
        assertSame(Init.XAVIER, Activation.TANH.preferredInit());
    }
}
