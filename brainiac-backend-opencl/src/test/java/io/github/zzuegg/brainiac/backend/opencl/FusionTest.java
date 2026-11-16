package io.github.zzuegg.brainiac.backend.opencl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
    void matmulOperandsComputedAsTheKernelLoadsThemMatchTheCpu() {
        Tensor w = random(25, 7, 5);
        // element-wise chains as operands, directly and through transposes and permutations
        same(DType.FLOAT32, x -> x.relu().mul(1.5f).add(0.25f).matmul(w.to(x.device()).tanh()), random(26, 9, 7));
        same(DType.FLOAT32, x -> x.sigmoid().transpose().matmul(x.exp()), random(27, 9, 7));
        same(DType.FLOAT32, x -> {
            Tensor y = x.square().sub(0.5f);
            return y.permute(0, 2, 1).matmul(y).sum(2).add(y.sum(1, true).matmul(y.abs().permute(0, 2, 1)).sum(2));
        }, random(28, 3, 6, 4));
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

    @Test
    void releasedBuffersAreReusedAndTheirTensorsRefused() {
        Tensor x = random(50, 1000).to(gpu);
        Tensor y = x.mul(2f);
        Tensor.evaluate(List.of(y));
        float[] expected = y.toArray();
        y.release();
        assertThrows(IllegalStateException.class, y::toArray);
        Tensor z = x.mul(2f);   // takes the released buffer
        assertArrayEquals(expected, z.toArray());
    }

    /** Five gradient steps, releasing each step's graph, as a trainer does. */
    private static float[] trainReleasing(Device device) {
        io.github.zzuegg.brainiac.autograd.Var w =
                io.github.zzuegg.brainiac.autograd.Var.param(random(70, 16, 8).to(device));
        Tensor x = random(71, 32, 16).to(device);
        for (int step = 0; step < 5; step++) {
            io.github.zzuegg.brainiac.autograd.Var input = io.github.zzuegg.brainiac.autograd.Var.of(x);
            // a view of the parameter, a constant computed this step and a matmul
            io.github.zzuegg.brainiac.autograd.Var shift = io.github.zzuegg.brainiac.autograd.Var.of(x.mul(0.5f));
            io.github.zzuegg.brainiac.autograd.Var loss = input.add(shift)
                    .matmul(w.reshape(8, 16).reshape(16, 8)).tanh().square().sum();
            loss.backward();
            Tensor old = w.value();
            w.assign(w.value().sub(w.grad().mul(0.01f)));
            w.zeroGrad();
            Tensor.evaluate(List.of(w.value()));
            loss.release();
            Tensor.release(List.of(old), List.of(w.value()));
        }
        return w.value().toArray();
    }

    @Test
    void releasingAStepsGraphKeepsParametersAndConstants() {
        float[] expected = trainReleasing(Device.cpu());
        float[] actual = trainReleasing(gpu);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 1e-4);
        }
    }

    /**
     * Three plain gradient steps of a small transformer, releasing each step's graph, the
     * replaced weights and the masks as the trainer does; returns the last loss and weights.
     */
    private static float[] trainTransformer(Device device) {
        io.github.zzuegg.brainiac.nn.Model model = io.github.zzuegg.brainiac.nn.Model.sequential().input(8, 12)
                .dense(16).positionalEncoding()
                .add(io.github.zzuegg.brainiac.nn.TransformerBlock.of(4).ffnSize(32))
                .add(io.github.zzuegg.brainiac.nn.TransformerBlock.of(4).ffnSize(32))
                .dense(4).seed(7).build().to(device);
        List<io.github.zzuegg.brainiac.autograd.Var> params = model.parameters();
        Tensor x = random(80, 6, 8, 12).to(device);
        Tensor target = random(81, 6, 8, 4).to(device);
        float last = 0;
        for (int step = 0; step < 3; step++) {
            io.github.zzuegg.brainiac.autograd.Var[] loss = new io.github.zzuegg.brainiac.autograd.Var[1];
            List<Tensor> work = Tensor.created(() -> loss[0] = model
                    .forward(io.github.zzuegg.brainiac.autograd.Var.of(x), true).sub(target).square().mean());
            loss[0].backward();
            List<Tensor> old = new ArrayList<>();
            List<Tensor> kept = new ArrayList<>(List.of(loss[0].value(), x, target));
            for (io.github.zzuegg.brainiac.autograd.Var p : params) {
                old.add(p.value());
                p.assign(p.value().sub(p.grad().mul(0.05f)));
                p.zeroGrad();
                kept.add(p.value());
            }
            Tensor.evaluate(kept);
            last = loss[0].value().item();
            loss[0].release();
            old.addAll(work);
            Tensor.release(old, kept);
        }
        List<Float> out = new ArrayList<>(List.of(last));
        for (float v : params.get(0).value().toArray()) {
            out.add(v);
        }
        float[] result = new float[out.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.get(i);
        }
        return result;
    }

    @Test
    void aTransformerTrainsLikeOnTheCpuWhileReleasingEachStep() {
        // deep pending graphs (attention around reductions) once recursed without end, and
        // released values once included the shared scalar constants
        float[] expected = trainTransformer(Device.cpu());
        float[] actual = trainTransformer(gpu);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 1e-3, "value " + i);
        }
    }

    /** Output and every parameter's gradient of a switching layer, for one batch. */
    private static List<float[]> switching(Device device, boolean hard) {
        io.github.zzuegg.brainiac.nn.Layer layer = io.github.zzuegg.brainiac.nn.SwitchingDense
                .of(24, io.github.zzuegg.brainiac.nn.Activation.RELU, io.github.zzuegg.brainiac.nn.Activation.TANH,
                        io.github.zzuegg.brainiac.nn.Activation.GELU, io.github.zzuegg.brainiac.nn.Activation.SIGMOID)
                .hard(hard).create(Shape.of(10), new SplittableRandom(9));
        for (io.github.zzuegg.brainiac.autograd.Var p : layer.parameters()) {
            p.assign(p.value().to(device));
        }
        io.github.zzuegg.brainiac.autograd.Var y = layer.forward(io.github.zzuegg.brainiac.autograd.Var.of(
                random(90, 16, 10).to(device)));
        y.mul(random(91, 16, 24).to(device)).sum().backward();
        List<float[]> out = new ArrayList<>(List.of(y.value().toArray()));
        for (io.github.zzuegg.brainiac.autograd.Var p : layer.parameters()) {
            out.add(p.grad() == null ? new float[0] : p.grad().toArray());
        }
        return out;
    }

    @Test
    void aSwitchingLayerMatchesTheCpuAndBlendsInOneKernel() throws ReflectiveOperationException {
        for (boolean hard : new boolean[] {false, true}) {
            List<float[]> expected = switching(Device.cpu(), hard);
            List<float[]> actual = switching(gpu, hard);
            for (int i = 0; i < expected.size(); i++) {
                assertEquals(expected.get(i).length, actual.get(i).length, "result " + i);
                for (int j = 0; j < expected.get(i).length; j++) {
                    float e = expected.get(i)[j];
                    assertEquals(e, actual.get(i)[j], 1e-4 * Math.max(1, Math.abs(e)), "result " + i + ", " + j);
                }
            }
        }
        // the forward pass: the two matrix products, then the blend of all four candidates at once
        io.github.zzuegg.brainiac.nn.Layer layer = io.github.zzuegg.brainiac.nn.SwitchingDense
                .of(32, io.github.zzuegg.brainiac.nn.Activation.RELU, io.github.zzuegg.brainiac.nn.Activation.TANH,
                        io.github.zzuegg.brainiac.nn.Activation.GELU, io.github.zzuegg.brainiac.nn.Activation.SIGMOID)
                .create(Shape.of(16), new SplittableRandom(10));
        for (io.github.zzuegg.brainiac.autograd.Var p : layer.parameters()) {
            p.assign(p.value().to(gpu));
        }
        Tensor x = random(92, 64, 16).to(gpu);
        Tensor.evaluate(layer.parameters().stream().map(io.github.zzuegg.brainiac.autograd.Var::value).toList());
        x.toArray();
        java.lang.reflect.Method backendOf = Device.class.getDeclaredMethod("backend");
        backendOf.setAccessible(true);
        OpenClBackend backend = (OpenClBackend) backendOf.invoke(gpu);
        java.lang.reflect.Field launches = OpenClBackend.class.getDeclaredField("launchIndex");
        launches.setAccessible(true);
        long before = launches.getLong(backend);
        layer.forward(io.github.zzuegg.brainiac.autograd.Var.of(x)).value().toArray();
        long used = launches.getLong(backend) - before;
        assertTrue(used <= 3, "forward took " + used + " launches");
    }

    @Test
    void reductionsOfTheSameShapeRunTogether() {
        for (int[] dims : new int[][] {{6, 300}, {700, 5}, {3, 2000}}) {
            Tensor input = random(60 + dims[0], dims);
            java.util.function.Function<Tensor, List<Tensor>> f = x -> {
                Tensor y = x.mul(1.5f).sub(0.2f);
                Tensor s = y.sum(1);
                return List.of(s, y.mul(x).sum(1), x.max(1, false), y.square().mean(1), x.min(0, false),
                        y.sub(s.reshape(dims[0], 1)).sum(1),   // depends on another reduction
                        // a variance over columns: reads its mean at its own output, in one kernel
                        x.sub(x.mean(0).reshape(1, dims[1])).square().mean(0),
                        x.sub(x.mean(0).reshape(1, dims[1])).mul(y).sum(0));
            };
            List<Tensor> cpu = f.apply(input);
            List<Tensor> separate = f.apply(input.to(gpu));
            List<float[]> separateValues = new ArrayList<>();
            for (Tensor t : separate) {
                separateValues.add(t.toArray());   // read one at a time, before anything is evaluated
            }
            List<Tensor> together = f.apply(input.to(gpu));
            Tensor.evaluate(together);
            for (int i = 0; i < cpu.size(); i++) {
                float[] e = cpu.get(i).toArray();
                float[] a = together.get(i).toArray();
                assertArrayEquals(separateValues.get(i), a, "result " + i);
                for (int j = 0; j < e.length; j++) {
                    assertEquals(e[j], a[j], 1e-4 * Math.max(1, Math.abs(e[j])), "result " + i + ", element " + j);
                }
            }
        }
    }

    @Test
    void randomMasksAreTheSameOnEveryDevice() {
        Shape shape = Shape.of(37, 29);
        float[] cpu = Tensor.random(shape, 1234L, Device.cpu()).toArray();
        float[] stored = Tensor.random(shape, 1234L, gpu).toArray();
        assertArrayEquals(cpu, stored);
        Tensor x = random(90, 37, 29);
        java.util.function.Function<Tensor, Tensor> dropout = t -> t.mul(Tensor.random(shape, 99L, t.device())
                .greater(Tensor.full(Shape.of(), 0.3f, t.device())).div(0.7f));
        assertArrayEquals(dropout.apply(x).toArray(), dropout.apply(x.to(gpu)).toArray(), 1e-6f);
        for (float v : cpu) {
            assertTrue(v >= 0 && v < 1);
        }
    }

    @Test
    void argmaxWithOtherReductionsMatchesTheCpu() {
        float[] values = new float[40 * 9];
        SplittableRandom r = new SplittableRandom(80);
        for (int i = 0; i < values.length; i++) {
            values[i] = r.nextInt(4);   // many ties: the first largest wins
        }
        values[9] = Float.NaN;   // NaN is never the largest
        Tensor input = Tensor.of(Shape.of(40, 9), values);
        java.util.function.Function<Tensor, List<Tensor>> f = x -> {
            Tensor y = x.mul(2f).sub(1f);
            return List.of(y.argmax(1), y.max(1, false), y.argmax(0));
        };
        List<Tensor> cpu = f.apply(input);
        List<Tensor> dev = f.apply(input.to(gpu));
        Tensor.evaluate(dev);
        for (int i = 0; i < cpu.size(); i++) {
            assertArrayEquals(cpu.get(i).toArray(), dev.get(i).toArray(), "result " + i);
        }
    }

    /** A small network of convolutions and poolings with its gradients, on one device. */
    private static List<float[]> convolutions(Device device) {
        io.github.zzuegg.brainiac.autograd.Var x = io.github.zzuegg.brainiac.autograd.Var.param(
                random(70, 3, 4, 9, 8).to(device));
        io.github.zzuegg.brainiac.autograd.Var w1 = io.github.zzuegg.brainiac.autograd.Var.param(
                random(71, 6, 4, 3, 3).to(device));
        io.github.zzuegg.brainiac.autograd.Var b1 = io.github.zzuegg.brainiac.autograd.Var.param(
                random(72, 6).to(device));
        io.github.zzuegg.brainiac.autograd.Var w2 = io.github.zzuegg.brainiac.autograd.Var.param(
                random(73, 4, 3, 2, 2).to(device));
        io.github.zzuegg.brainiac.autograd.Var y = io.github.zzuegg.brainiac.nn.Functional.conv(x, w1, b1,
                new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, 1).relu();
        io.github.zzuegg.brainiac.autograd.Var pooled = io.github.zzuegg.brainiac.nn.Functional.pool(y,
                io.github.zzuegg.brainiac.nn.Functional.PoolKind.MAX, new int[] {2, 2}, new int[] {2, 2},
                new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, false, false, 0);
        io.github.zzuegg.brainiac.autograd.Var z = io.github.zzuegg.brainiac.nn.Functional.conv(pooled, w2, null,
                new int[] {2, 1}, new int[] {0, 1}, new int[] {1, 0}, new int[] {1, 1}, 2);
        io.github.zzuegg.brainiac.autograd.Var avg = io.github.zzuegg.brainiac.nn.Functional.pool(z,
                io.github.zzuegg.brainiac.nn.Functional.PoolKind.AVG, new int[] {2, 2}, new int[] {1, 1},
                new int[] {1, 0}, new int[] {0, 1}, new int[] {1, 1}, false, true, 0);
        avg.square().mean().backward();
        return List.of(y.value().toArray(), z.value().toArray(), avg.value().toArray(), x.grad().toArray(),
                w1.grad().toArray(), b1.grad().toArray(), w2.grad().toArray());
    }

    @Test
    void convolutionWindowsComputedInKernelsMatchTheCpu() {
        List<float[]> cpu = convolutions(Device.cpu());
        List<float[]> first = convolutions(gpu);
        List<float[]> second = convolutions(gpu);   // with the kernels compiled
        for (int i = 0; i < cpu.size(); i++) {
            assertArrayEquals(first.get(i), second.get(i), "result " + i + " differs between runs");
            float[] e = cpu.get(i);
            float[] a = second.get(i);
            assertEquals(e.length, a.length);
            for (int j = 0; j < e.length; j++) {
                assertEquals(e[j], a[j], 1e-4 * Math.max(1, Math.abs(e[j])), "result " + i + ", element " + j);
            }
        }
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
        // few outputs and a long k: summed in pieces, whether or not a chain follows
        float[] ls = new float[8 * 4096];
        for (int i = 0; i < ls.length; i++) {
            ls[i] = (float) r.nextDouble(-1, 1);
        }
        Storage l = b.upload(ls);
        Storage lp = b.allocateUninitialized(8 * 8, DType.FLOAT32);
        b.matmul(l, l, 8, 4096, 8, lp);
        Storage lq = b.allocateUninitialized(8 * 8, DType.FLOAT32);
        b.activation(ActivationOp.TANH, 0, lp, lq);
        return new float[][] {b.download(rows), b.download(cols), b.download(all), b.download(z), b.download(q),
                b.download(p), b.download(lq)};
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
