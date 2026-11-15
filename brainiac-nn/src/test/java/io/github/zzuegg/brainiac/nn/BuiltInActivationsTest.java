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

    /** The fused activations agree with the same functions written out with Var operations. */
    @Test
    void fusedKernelsMatchComposedDefinitions() {
        Var x = Var.of(X);
        float c = (float) Math.sqrt(2 / Math.PI);
        assertClose(x.mul(x.sigmoid()), Activation.SILU, x);
        assertClose(x.mul(x.mul((float) (1 / Math.sqrt(2))).erf().add(1)).mul(0.5f), Activation.GELU, x);
        assertClose(x.mul(x.add(x.pow(3).mul(0.044715f)).mul(c).tanh().add(1)).mul(0.5f), Activation.GELU_TANH, x);
        assertClose(x.mul(x.softplus().tanh()), Activation.MISH, x);
        assertClose(Var.where(x.greater(0), x, x.minimum(0).exp().sub(1).mul(1.6732632f)).mul(1.0507010f),
                Activation.SELU, x);
        assertClose(Var.where(x.greater(0), x, x.minimum(0).exp().sub(1).mul(0.7f)), Activation.elu(0.7f), x);
        assertClose(x.relu().sub(x.neg().relu().mul(0.1f)), Activation.leakyRelu(0.1f), x);
        assertClose(x.clamp(0, 6), Activation.RELU6, x);
        assertClose(x.clamp(-1, 1), Activation.HARD_TANH, x);
        assertClose(x.div(6).add(0.5f).clamp(0, 1), Activation.HARD_SIGMOID, x);
        assertClose(x.mul(x.div(6).add(0.5f).clamp(0, 1)), Activation.HARD_SWISH, x);
        assertClose(x.div(x.abs().add(1)), Activation.SOFTSIGN, x);
    }

    private static void assertClose(Var expected, Activation a, Var x) {
        assertArrayEquals(expected.value().toArray(), a.forward(x).value().toArray(), 2e-6f, a.toString());
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
