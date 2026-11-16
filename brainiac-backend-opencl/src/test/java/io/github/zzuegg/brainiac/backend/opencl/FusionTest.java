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
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.util.ArrayList;
import java.util.List;
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
    void matmulEpiloguesMatchTheCpu() {
        Tensor w = random(20, 7, 5);
        Tensor bias = random(21, 5);
        same(DType.FLOAT32, x -> x.matmul(w.to(x.device())).add(bias.to(x.device())).relu().mul(2f),
                random(22, 9, 7));
        // the product is also used on its own, through a view, and in a reduction
        same(DType.FLOAT32, x -> {
            Tensor z = x.matmul(w.to(x.device()));
            return z.tanh().add(z.transpose().transpose()).sum(0).add(z.sum(0));
        }, random(23, 9, 7));
        // batched, with a transposed operand read in place
        same(DType.FLOAT32, x -> x.matmul(x.transpose()).sigmoid(), random(24, 3, 4, 6));
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

    /** An optimizer-like step over parameters of many sizes: new values and the state they use. */
    private static List<Tensor> step(List<Tensor> params, List<Tensor> grads, Device device) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor g = grads.get(i).to(device);
            Tensor m = g.mul(0.1f);
            Tensor v = g.square().mul(0.001f);
            Tensor p = params.get(i).to(device).sub(m.div(v.sqrt().add(1e-8f)).mul(0.01f));
            out.add(p);
            out.add(m);
            out.add(v);
            if (i % 3 == 0) {
                out.add(m.transpose().add(1f));   // read through a view: needs its own range
            }
        }
        return out;
    }

    @Test
    void evaluatingTogetherMatchesOneByOne() {
        List<Tensor> params = new ArrayList<>();
        List<Tensor> grads = new ArrayList<>();
        for (int i = 0; i < 40; i++) {   // more than one kernel's arguments allow
            params.add(random(100 + i, 1 + i % 7, 3 + 5 * i));
            grads.add(random(200 + i, 1 + i % 7, 3 + 5 * i));
        }
        List<Tensor> cpu = step(params, grads, Device.cpu());
        for (int run = 0; run < 3; run++) {
            List<Tensor> separate = step(params, grads, gpu);
            List<Tensor> together = step(params, grads, gpu);
            Tensor.evaluate(together);
            for (int i = 0; i < cpu.size(); i++) {
                float[] e = cpu.get(i).toArray();
                assertArrayEquals(separate.get(i).toArray(), together.get(i).toArray(), "tensor " + i);
                float[] a = together.get(i).toArray();
                for (int j = 0; j < e.length; j++) {
                    assertEquals(e[j], a[j], 1e-5 * Math.max(1, Math.abs(e[j])), "tensor " + i + ", element " + j);
                }
            }
        }
    }

    @Test
    void changingScalarsAndFilledTensorsMatchTheCpu() {
        // scalars that change every step (like a bias correction) are kernel arguments
        for (int step = 1; step <= 5; step++) {
            float c = 1 - (float) Math.pow(0.9, step);
            same(DType.FLOAT32, x -> x.div(c).add(Tensor.full(Shape.of(3, 4), c, x.device())).mul(x),
                    random(30 + step, 3, 4));
        }
        same(DType.FLOAT64, x -> Tensor.zeros(Shape.of(2, 3), DType.FLOAT64, x.device()).add(x).sum(0),
                random(40, 2, 3));
        same(DType.FLOAT32, x -> Tensor.full(Shape.of(5), 2.5f, x.device()).sum(), random(41, 1));
    }

    /** Runs a chain ending in reductions through the backend directly. */
    private static float[][] chain(OpenClBackend b) {
        SplittableRandom r = new SplittableRandom(15);
        float[] xs = new float[40 * 300];
        float[] ws = new float[300];
        for (int i = 0; i < xs.length; i++) {
            xs[i] = (float) r.nextDouble(-1, 1);
        }
        for (int i = 0; i < ws.length; i++) {
            ws[i] = (float) r.nextDouble(-1, 1);
        }
        int[] shape = {40, 300};
        Storage x = b.upload(xs);
        Storage w = b.upload(ws);
        Storage y = b.allocateUninitialized(xs.length, DType.FLOAT32);
        b.binary(BinaryOp.MUL, x, new long[] {300, 1}, w, new long[] {0, 1}, shape, y);
        Storage z = b.allocateUninitialized(xs.length, DType.FLOAT32);
        b.activation(ActivationOp.GELU, 0, y, z);
        Storage rows = b.allocateUninitialized(40, DType.FLOAT32);
        b.reduce(ReduceOp.SUM, z, 40, 300, 1, rows);
        Storage cols = b.allocateUninitialized(300, DType.FLOAT32);
        b.reduce(ReduceOp.MEAN, z, 1, 40, 300, cols);
        Storage all = b.allocateUninitialized(1, DType.FLOAT32);
        b.reduce(ReduceOp.MAX, z, 1, 12_000, 1, all);
        Storage p = b.allocateUninitialized(40 * 40, DType.FLOAT32);
        b.matmul(z, x, 40, 300, 40, p);   // x read as 300 x 40: shapes only need to agree in size
        Storage q = b.allocateUninitialized(40 * 40, DType.FLOAT32);
        b.activation(ActivationOp.TANH, 0, p, q);
        return new float[][] {b.download(rows), b.download(cols), b.download(all), b.download(z), b.download(q),
                b.download(p)};
    }

    @Test
    void fusedAndUnfusedResultsAreIdentical() {
        String old = System.getProperty("brainiac.opencl.fusion");
        System.setProperty("brainiac.opencl.fusion", "false");
        OpenClBackend plain = new OpenClBackend();
        System.setProperty("brainiac.opencl.fusion", "true");
        OpenClBackend fused = new OpenClBackend();
        if (old == null) {
            System.clearProperty("brainiac.opencl.fusion");
        } else {
            System.setProperty("brainiac.opencl.fusion", old);
        }
        assumeTrue(plain.isAvailable() && fused.isAvailable());
        float[][] expected = chain(plain);
        for (int run = 0; run < 5; run++) {
            float[][] actual = chain(fused);
            for (int i = 0; i < expected.length; i++) {
                assertArrayEquals(expected[i], actual[i], "result " + i + ", run " + run);
            }
        }
    }

    @Test
    void halfPrecisionIsComputedStepByStep() {
        Tensor x = random(14, 50).to(DType.FLOAT16);
        Tensor cpu = x.mul(3f).add(1f).tanh();
        Tensor dev = x.to(gpu).mul(3f).add(1f).tanh();
        assertArrayEquals(cpu.toArray(), dev.toArray(), 2e-3f);
    }
}
