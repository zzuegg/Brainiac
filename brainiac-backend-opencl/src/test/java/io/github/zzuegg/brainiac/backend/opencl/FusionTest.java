package io.github.zzuegg.brainiac.backend.opencl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.Optional;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Chains of element-wise operations are generated into single kernels; these check that the
 * generated kernels compute what the separate operations do, on the CPU and on the device.
 */
class FusionTest {

    private static Device gpu;

    @BeforeAll
    static void findDevice() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("opencl")).findFirst();
        assumeTrue(found.isPresent(), "no OpenCL device");
        gpu = found.get();
    }

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    private static void same(DType type, UnaryOperator<Tensor> op, Tensor input) {
        Tensor expected = op.apply(input.to(type));
        Tensor actual;
        try {
            actual = op.apply(input.to(type).to(gpu));
        } catch (UnsupportedOperationException e) {
            assumeTrue(false, type + " is not supported by " + gpu);
            return;
        }
        assertEquals(expected.shape(), actual.shape());
        double[] e = expected.toDoubleArray();
        double[] a = actual.toDoubleArray();
        double tolerance = type == DType.FLOAT64 ? 1e-6 : 1e-5;   // the CPU erf is good to 1.5e-7
        for (int i = 0; i < e.length; i++) {
            assertEquals(e[i], a[i], tolerance * Math.max(1, Math.abs(e[i])), "element " + i);
        }
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT64"})
    void chainsWithBroadcastingMatchTheCpu(DType type) {
        Tensor bias = random(2, 7).to(type);
        Tensor column = random(3, 5, 1).to(type);
        same(type, x -> {
            Tensor b = bias.to(x.device());
            Tensor c = column.to(x.device());
            Tensor y = x.mul(2f).add(b).relu().sub(c).tanh().mul(x).add(1f);
            return y.activation(ActivationOp.GELU, 0).div(y.abs().add(1f)).maximum(c.mul(0.5f));
        }, random(1, 5, 7));
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT64"})
    void reductionsOfChainsMatchTheCpu(DType type) {
        same(type, x -> x.mul(x).add(0.5f).sum(1), random(4, 6, 300, 3));
        same(type, x -> x.exp().mean(0), random(5, 40, 9));
        same(type, x -> x.sub(0.25f).max(2, false), random(6, 4, 5, 6));
        same(type, x -> x.neg().min(1, true), random(7, 3, 8));
        same(type, x -> x.square().sum(), random(8, 3, 4));
    }

    @Test
    void longChainsAreSplitAndSharedValuesReused() {
        same(DType.FLOAT32, x -> {
            Tensor y = x;
            for (int i = 0; i < 200; i++) {
                y = y.mul(y).mul(0.5f).add(y.mul(0.25f)).sub(0.1f).tanh();
            }
            return y;
        }, random(9, 64));
    }

    @Test
    void transposedAndSlicedInputsFuse() {
        same(DType.FLOAT32, x -> {
            Tensor t = x.add(1f).transpose();
            Tensor s = x.mul(3f).slice(1, 1, 3);
            return t.mul(t).sum(0).add(s.sum(1));
        }, random(10, 4, 6));
    }

    @Test
    void gradientsOfActivationsFuse() {
        same(DType.FLOAT32, x -> x.mul(2f).activationBackward(ActivationOp.SILU, 0, x.sigmoid()).add(x),
                random(11, 33));
    }

    @Test
    void resultsDoNotDependOnWhetherTheKernelIsCompiledYet() {
        // The first runs may use the separate kernels while the generated one compiles in the
        // background; every run must give exactly the same values.
        Tensor x = random(12, 257).to(gpu);
        Tensor w = random(13, 257).to(gpu);
        float[] first = null;
        for (int run = 0; run < 20; run++) {
            float[] r = x.mul(w).add(0.3f).sigmoid().mul(x.sub(w)).exp().sum(0).toArray();
            if (first == null) {
                first = r;
            } else {
                assertArrayEquals(first, r, "run " + run);
            }
        }
        assertTrue(Float.isFinite(first[0]));
    }

    @Test
    void halfPrecisionIsComputedStepByStep() {
        Tensor x = random(14, 50).to(DType.FLOAT16);
        Tensor cpu = x.mul(3f).add(1f).tanh();
        Tensor dev = x.to(gpu).mul(3f).add(1f).tanh();
        assertArrayEquals(cpu.toArray(), dev.toArray(), 2e-3f);
    }
}
