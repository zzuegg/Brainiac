package io.github.zzuegg.brainiac.backend.vulkan;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;

import java.util.List;
import java.util.Optional;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Runs every operation on the Vulkan device and on the CPU and compares. Skipped when no
 * Vulkan driver is installed.
 */
class VulkanBackendTest {

    private static Device gpu;
    private static final Device CPU = Device.cpu();

    @BeforeAll
    static void findDevice() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("vulkan")).findFirst();
        if (found.isEmpty()) {
            System.out.println("no Vulkan device, skipping: " + new VulkanBackend().status());
        }
        assumeTrue(found.isPresent(), "no Vulkan device");
        gpu = found.get();
        System.out.println("testing on " + gpu);
    }

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    private static double tolerance(DType type) {
        return switch (type) {
            case FLOAT64 -> 1e-6;   // the CPU's erf is an approximation good to 1.5e-7
            case FLOAT32 -> 2e-5;
            case FLOAT16 -> 2e-2;
            case BFLOAT16 -> 8e-2;
        };
    }

    /** Runs an operation on the CPU and on the device in the given type and compares. */
    private static void same(DType type, String what, UnaryOperator<Tensor> op, Tensor input) {
        Tensor expected = op.apply(input.to(type));
        Tensor actual = op.apply(input.to(type).to(gpu));
        assertEquals(gpu, actual.device(), what);
        assertEquals(expected.shape(), actual.shape(), what);
        assertEquals(expected.dtype(), actual.dtype(), what);
        double[] e = expected.toDoubleArray();
        double[] a = actual.toDoubleArray();
        double scale = 1;
        for (double v : e) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < e.length; i++) {
            assertEquals(e[i], a[i], tolerance(type) * scale, what + " at " + i);
        }
    }

    private static boolean supported(DType type) {
        try {
            Tensor.zeros(Shape.of(1), type, gpu).add(1).toArray();
            return true;
        } catch (UnsupportedOperationException e) {
            return false;
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void everyOperationMatchesTheCpu(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        Tensor x = random(1, 5, 7);
        Tensor y = random(2, 5, 7);
        UnaryOperator<Tensor> onDevice = t -> t.to(type);
        same(type, "add", t -> t.add(onDevice.apply(y).to(t.device())), x);
        same(type, "row broadcast", t -> t.mul(onDevice.apply(y.slice(0, 0, 1)).to(t.device())), x);
        same(type, "column broadcast", t -> t.sub(onDevice.apply(y.slice(1, 0, 1)).to(t.device())), x);
        same(type, "scalar", t -> t.mul(3).add(1).div(2).sub(0.5f), x);
        same(type, "pow", t -> t.abs().pow(1.5f), x);
        same(type, "compare", t -> t.maximum(t.neg()).add(t.minimum(t.square())).add(t.greater(t.square())), x);
        same(type, "unary", t -> t.exp().add(t.tanh()).add(t.sigmoid()).add(t.square().add(1).log())
                .add(t.erf()).add(t.softplus()).add(t.abs().sqrt()).add(t.relu()).add(t.sign()).add(t.step()).add(t.mul(4).round()), x);
        for (ActivationOp op : ActivationOp.values()) {
            same(type, op.name(), t -> t.activation(op, 0.1f), x);
            same(type, op + " backward", t -> t.activationBackward(op, 0.1f, t.square()), x);
        }
        same(type, "fill", t -> Tensor.full(Shape.of(3), 2.5f, type, t.device()), x);
        same(type, "zeros", t -> Tensor.zeros(Shape.of(4), type, t.device()).add(t.sum()), x);
        same(type, "reductions", t -> Tensor.concat(1, t.sum(1, true), t.mean(1, true), t.max(1, true),
                t.min(1, true)), x);
        same(type, "reduce axis 0", t -> t.sum(0).add(t.max(0, false)), x);
        same(type, "argmax", t -> t.argmax(1).add(t.argmax(0).sum()), x);
        same(type, "slice, permute, concat", t -> Tensor.concat(0, t.transpose(), t.slice(1, 2, 7).transpose()), x);
        same(type, "permute 3d", t -> t.reshape(5, 7, 1).permute(2, 0, 1).reshape(7, 5).add(0), x);
        same(type, "gather", t -> t.gather(1, Tensor.of(6, 0, 6, 3).to(t.device())), x);
        same(type, "scatterAdd", t -> t.scatterAdd(1, Tensor.of(1, 1, 0, 3, 3, 3, 8).to(t.device()), 9), x);
        same(type, "matmul", t -> t.matmul(t.transpose()), x);
        same(type, "batched matmul", t -> t.reshape(1, 5, 7).expand(Shape.of(3, 5, 7)).mul(2)
                .matmul(t.transpose().reshape(1, 7, 5).expand(Shape.of(3, 7, 5)).add(0)), x);
        ConvGeometry g = ConvGeometry.sliding(1, 5, new int[] {7}, new int[] {3}, new int[] {2}, new int[] {1},
                new int[] {1}, new int[] {1}, false);
        same(type, "im2col", t -> t.reshape(1, 5, 7).im2col(g, 0.5f), x);
        same(type, "col2im", t -> t.reshape(1, 5, 7).im2col(g, 0).col2im(g), x);
        same(type, "convert", t -> t.to(DType.FLOAT32).to(type).add(t), x);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void uploadsRawValuesFromNativeMemory(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        // small (inline update), mid-size (staging ring) and large (own staging block)
        for (int n : new int[] {7, 5_000, 3_000_000}) {
            Tensor host = random(n, n).to(type);
            double[] values = host.toDoubleArray();
            try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
                java.lang.foreign.MemorySegment raw = arena.allocate((long) n * type.bytes());
                for (int i = 0; i < n; i++) {
                    switch (type) {
                        case FLOAT32 -> raw.setAtIndex(io.github.zzuegg.brainiac.tensor.spi.RawValues.FLOAT, i,
                                (float) values[i]);
                        case FLOAT64 -> raw.setAtIndex(io.github.zzuegg.brainiac.tensor.spi.RawValues.DOUBLE, i,
                                values[i]);
                        case FLOAT16 -> raw.setAtIndex(io.github.zzuegg.brainiac.tensor.spi.RawValues.SHORT, i,
                                Float.floatToFloat16((float) values[i]));
                        case BFLOAT16 -> raw.setAtIndex(io.github.zzuegg.brainiac.tensor.spi.RawValues.SHORT, i,
                                DType.floatToBfloat16((float) values[i]));
                    }
                }
                Tensor t = Tensor.of(Shape.of(n), raw, type, gpu);
                assertEquals(type, t.dtype());
                assertEquals(gpu, t.device());
                assertTrue(java.util.Arrays.equals(values, t.toDoubleArray()), type + " x " + n);
                assertTrue(java.util.Arrays.equals(values, Tensor.of(Shape.of(n), raw, type, CPU).toDoubleArray()));
            }
        }
    }

    @Test
    void largeShapesUseTilesAndChunkedReductions() {
        Tensor a = random(3, 67, 131);
        Tensor b = random(4, 131, 45);
        same(DType.FLOAT32, "odd matmul", t -> t.matmul(b.to(t.device())), a);
        // few outputs and a long k: summed in pieces
        Tensor wide = random(6, 9, 8200);
        Tensor tall = random(7, 8200, 21);
        same(DType.FLOAT32, "split matmul", t -> t.matmul(tall.to(t.device())), wide);
        same(DType.FLOAT32, "split batched matmul", t -> t.reshape(1, 9, 8200).expand(Shape.of(2, 9, 8200)).add(0)
                .matmul(tall.to(t.device()).reshape(1, 8200, 21).expand(Shape.of(2, 8200, 21)).mul(2)), wide);
        // 64 x 64 tiles: ragged edges, batches, transposed operands read in place, long k split
        Tensor m1 = random(8, 130, 70);
        Tensor m2 = random(9, 70, 200);
        same(DType.FLOAT32, "big-tile matmul", t -> t.matmul(m2.to(t.device())), m1);
        same(DType.FLOAT32, "big-tile transposed", t -> t.transpose().matmul(t), m1);
        same(DType.FLOAT32, "big-tile batched", t -> t.reshape(2, 65, 70).matmul(m2.to(t.device())
                .reshape(1, 70, 200).expand(Shape.of(2, 70, 200)).add(0)), m1);
        Tensor longK = random(10, 64, 9000);
        same(DType.FLOAT32, "big-tile split", t -> t.matmul(t.transpose()), longK);
        Tensor big = random(5, 300_000);
        same(DType.FLOAT32, "long sum", t -> t.sum().reshape(1).mul(1e-3f), big);
        same(DType.FLOAT32, "long max", Tensor::max, big);
        same(DType.FLOAT32, "long mean of rows", t -> t.reshape(3, 100_000).mean(1), big);
    }

    @Test
    void float64HasDoublePrecision() {
        assumeTrue(supported(DType.FLOAT64), "float64 not supported by " + gpu);
        // GLSL has no double exp or log; the shaders bring their own
        Tensor x = random(13, 64).mul(30).to(DType.FLOAT64);
        UnaryOperator<Tensor> f = t -> t.exp().add(t.abs().add(1e-3f).log()).add(t.tanh()).add(t.sigmoid())
                .add(t.abs().pow(0.7f));
        double[] e = f.apply(x).toDoubleArray();
        double[] a = f.apply(x.to(gpu)).toDoubleArray();
        for (int i = 0; i < e.length; i++) {
            assertEquals(e[i], a[i], 1e-12 * Math.max(1, Math.abs(e[i])), "at " + i);
        }
    }

    @Test
    void highRanksAndLargeUploadsMatchTheCpu() {
        // eight axes that cannot be merged: run as several dispatches over the outer axes
        Tensor x = random(11, 2, 3, 2, 3, 2, 3, 2, 3);
        same(DType.FLOAT32, "permute rank 8", t -> t.permute(7, 5, 3, 1, 0, 2, 4, 6).add(0), x);
        same(DType.FLOAT32, "broadcast rank 8", t -> t.permute(1, 0, 3, 2, 5, 4, 7, 6)
                .add(t.slice(0, 0, 1).permute(1, 0, 3, 2, 5, 4, 7, 6)), x);
        // through a staging buffer of its own
        Tensor big = random(12, 3_000_000);
        same(DType.FLOAT32, "large upload", t -> t.mul(2).slice(0, 2_999_000, 3_000_000), big);
    }

    @Test
    void networksTrainTheSameOnBothDevices() {
        float[] cpuLoss = train(CPU);
        float[] gpuLoss = train(gpu);
        for (int i = 0; i < cpuLoss.length; i++) {
            assertEquals(cpuLoss[i], gpuLoss[i], 1e-3 * Math.max(1, Math.abs(cpuLoss[i])), "step " + i);
        }
        assertTrue(cpuLoss[cpuLoss.length - 1] < cpuLoss[0], "training lowers the loss");
    }

    /** A small conv net trained for a few steps of plain gradient descent. */
    private static float[] train(Device device) {
        Model model = Model.sequential()
                .input(1, 8, 8)
                .add(Conv.of(4, 3, 3).padding(Padding.SAME))
                .activation(Activation.RELU)
                .maxPool(2, 2)
                .flatten()
                .dense(8, Activation.TANH)
                .dense(3, Activation.SOFTMAX)
                .seed(9)
                .build()
                .to(device);
        Tensor x = random(6, 16, 64).add(1).mul(0.5f).to(device);
        float[] labels = new float[16 * 3];
        for (int i = 0; i < 16; i++) {
            labels[i * 3 + i % 3] = 1;
        }
        Tensor y = Tensor.of(Shape.of(16, 3), labels).to(device);
        float[] losses = new float[8];
        for (int step = 0; step < losses.length; step++) {
            Var loss = Loss.CROSS_ENTROPY.compute(model.forward(Var.of(x), true), y);
            loss.backward();
            for (Var p : model.parameters()) {
                p.assign(p.value().sub(p.grad().mul(0.5f)));
                p.zeroGrad();
            }
            losses[step] = loss.value().item();
        }
        return losses;
    }

    @Test
    void embeddingsWithPaddingMasksMatchTheCpu() {
        io.github.zzuegg.brainiac.nn.Node ids = io.github.zzuegg.brainiac.nn.Node.input(6);
        io.github.zzuegg.brainiac.nn.Node mask = ids.layer(new io.github.zzuegg.brainiac.nn.PaddingMask(0));
        io.github.zzuegg.brainiac.nn.Node out = ids
                .add(io.github.zzuegg.brainiac.nn.Embedding.of(9, 8).paddingIndex(0).maxNorm(2))
                .positionalEncoding()
                .merge(io.github.zzuegg.brainiac.nn.TransformerBlock.of(2).keyMask(true), mask)
                .merge(io.github.zzuegg.brainiac.nn.MaskedPool.of(io.github.zzuegg.brainiac.nn.SequencePool.Kind.MAX),
                        mask)
                .dense(3, Activation.SOFTMAX);
        Model cpu = Model.graph(ids, out).seed(3).build();
        Model device = Model.graph(ids, out).seed(3).build().to(gpu);
        Tensor x = Tensor.of(Shape.of(2, 6), 1, 5, 8, 0, 0, 0, 2, 2, 7, 3, 1, 0);
        same(DType.FLOAT32, "embedding model", t -> (t.device().equals(gpu) ? device : cpu).predict(t), x);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT64"})
    void fusedChainsAndViewsMatchTheCpu(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        Tensor x = random(21, 6, 9);
        Tensor y = random(22, 9, 6);
        UnaryOperator<Tensor> longChain = t -> {
            Tensor r = t;
            for (int i = 0; i < 70; i++) {
                r = i % 7 == 0 ? r.tanh() : r.mul(0.9f).add(t);   // beyond one shader's limit
            }
            return r;
        };
        same(type, "long chain", longChain, x);
        same(type, "shared values", t -> {
            Tensor a = t.exp();
            Tensor b = a.mul(a).add(a.sigmoid());
            return b.sub(a).div(a.add(1));
        }, x);
        same(type, "transposed result", t -> t.exp().add(1).transpose().mul(2), x);
        same(type, "result read in place by a matmul", t -> t.mul(2).add(1).transpose().matmul(t.relu()), x);
        same(type, "permuted chain", t -> t.reshape(2, 3, 9).permute(2, 0, 1).tanh().permute(1, 2, 0)
                .reshape(6, 9).add(t), x);
        same(type, "slice of a result", t -> t.square().slice(1, 2, 5).add(t.slice(1, 4, 7).exp()), x);
        same(type, "broadcast of a result", t -> t.add(t.sum(1, true).exp().mul(0.1f)), x);
        same(type, "broadcast of a pending row", t -> t.mul(t.slice(0, 1, 2).abs().add(1)), x);
        same(type, "two results of one chain", t -> {
            Tensor a = t.mul(3).sub(1);
            Tensor b = a.relu();
            Tensor c = a.neg().relu();
            Tensor.evaluate(List.of(b, c));
            return b.sub(c).matmul(y.to(type).to(t.device()));
        }, x);
        same(type, "constants", t -> Tensor.full(Shape.of(6, 9), 0.25f, type, t.device()).mul(t)
                .add(Tensor.zeros(Shape.of(9), type, t.device())), x);
        if (type == DType.FLOAT32) {
            // the mask is generated inside the shader that uses it
            same(type, "random mask", t -> t.mul(Tensor.random(t.shape(), 77, t.device()).greater(
                    Tensor.full(t.shape(), 0.5f, type, t.device()))), x);
        }
    }

    @Test
    void elementWiseChainsRunAsOneDispatch() {
        VulkanBackend vk = VulkanBackend.loaded().orElseThrow();
        assumeTrue(vk.status().contains("fusion on"), vk.status());
        Tensor x = random(23, 64, 64).to(gpu);
        x.toArray();
        long before = vk.dispatchCount();
        Tensor r = x;
        for (int i = 0; i < 6; i++) {
            r = r.mul(1.01f).add(x).tanh();
        }
        r.transpose().toArray();
        assertEquals(1, vk.dispatchCount() - before, vk.statistics());
    }

    @Test
    void manyTemporaryTensorsAreFreed() {
        Tensor t = Tensor.zeros(Shape.of(1 << 18), gpu);
        for (int i = 0; i < 300; i++) {
            t = t.add(1);   // 1 MB each; earlier results become garbage
        }
        assertEquals(300f, t.max().item());
    }

    @Test
    void badIndicesAreRejected() {
        Tensor x = random(7, 3, 4).to(gpu);
        assertThrows(IndexOutOfBoundsException.class, () -> x.gather(1, Tensor.of(4).to(gpu)));
    }

    @Test
    void statusDescribesTheDevice() {
        String status = Device.available().stream().map(Device::name).filter(n -> n.startsWith("vulkan"))
                .findFirst().orElseThrow();
        assertTrue(status.length() > "vulkan".length(), status);
        assertTrue(List.of(DType.values()).size() == 4);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"8, 64, 1", "8, 64, 5", "4, 64, 1", "4, 32, 8", "3, 48, 2",
        "4, 3, 9", "8, 64, 20", "8, 1536, 1", "4, 32, 3", "2, 2, 4"})
    void packedMatricesMatchTheCpu(int bits, int group, int m) {
        packedMatches(bits, group, m, group > 192 ? 3072 : 192, 77);
        packedMatches(bits, group, m, 768, 256);
    }

    private static void packedMatches(int bits, int group, int m, int rows, int cols) {
        if (rows % group != 0) {
            return;
        }
        SplittableRandom r = new SplittableRandom(bits * 100L + m + rows);
        int limit = (1 << (bits - 1)) - 1;
        byte[] levels = new byte[rows * cols];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) r.nextInt(-limit, limit + 1);
        }
        float[] scales = new float[rows / group * cols];
        for (int i = 0; i < scales.length; i++) {
            scales[i] = (float) r.nextDouble(0.001, 0.05);
        }
        PackedMatrix w = PackedMatrix.of(levels, scales, rows, cols, bits, group);
        Tensor x = random(3, m, rows);
        Tensor onCpu = Tensor.packed(w, CPU);
        Tensor onGpu = onCpu.to(gpu);
        String what = bits + " bits, groups of " + group + ", " + m + " x " + rows + " x " + cols;
        assertTrue(onGpu.isPacked());
        org.junit.jupiter.api.Assertions.assertArrayEquals(onCpu.toArray(), onGpu.toArray(), 1e-6f, what);
        float[] expected = x.matmul(onCpu).toArray();
        double scale = 1;
        for (float v : expected) {
            scale = Math.max(scale, Math.abs(v));
        }
        // sums in another order: errors grow with the largest value
        float close = (float) (5e-5 * scale);
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, x.to(gpu).matmul(onGpu).toArray(), close, what);
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, x.to(gpu).matmul(onGpu.to(CPU).to(gpu)).toArray(),
                close, what);
        for (DType type : List.of(DType.FLOAT64, DType.FLOAT16, DType.BFLOAT16)) {
            if (supported(type)) {
                // inputs rounded to 16 bits: errors relative to the largest output
                float[] actual = x.to(type).to(gpu).matmul(onGpu).toArray();
                double tolerance = (type == DType.FLOAT64 ? 1e-5 : type == DType.FLOAT16 ? 4e-3 : 3e-2) * scale;
                for (int i = 0; i < expected.length; i++) {
                    assertEquals(expected[i], actual[i], tolerance, what + ", " + type + " at " + i);
                }
            }
        }
    }
}
