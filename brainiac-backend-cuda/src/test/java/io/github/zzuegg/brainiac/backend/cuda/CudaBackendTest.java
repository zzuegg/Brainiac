package io.github.zzuegg.brainiac.backend.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
 * Runs every operation on the CUDA device and on the CPU and compares. Skipped when there is no
 * NVIDIA driver and device.
 */
class CudaBackendTest {

    private static Device gpu;
    private static final Device CPU = Device.cpu();

    @BeforeAll
    static void findDevice() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("cuda")).findFirst();
        if (found.isEmpty()) {
            System.out.println("no CUDA device, skipping: " + new CudaBackend().status());
        }
        assumeTrue(found.isPresent(), "no CUDA device");
        gpu = found.get();
        System.out.println("testing on " + gpu + ": " + CudaBackend.loaded().map(CudaBackend::status).orElse("?"));
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
        same(type, "trigonometric", t -> t.mul(3).sin().add(t.mul(2).cos()), x);
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
                Tensor x = random(13, 64).mul(30).to(DType.FLOAT64);
        UnaryOperator<Tensor> f = t -> t.exp().add(t.abs().add(1e-3f).log()).add(t.tanh()).add(t.sigmoid())
                .add(t.abs().pow(0.7f)).add(t.sin()).add(t.mul(100).cos());
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

    /** Uploads and downloads larger than the staging buffer go through in pieces, in order. */
    @Test
    void largeUploadsArriveWhole() {
        float[] data = new float[(80 << 20) / 4 + 12345];
        SplittableRandom r = new SplittableRandom(81);
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) r.nextDouble();
        }
        Tensor t = Tensor.of(Shape.of(data.length), data, gpu);
        assertArrayEquals(data, t.toArray());
        java.lang.foreign.MemorySegment seg = java.lang.foreign.MemorySegment.ofArray(data);
        assertArrayEquals(data, Tensor.of(Shape.of(data.length), seg, DType.FLOAT32, gpu).toArray());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT16"})
    void reductionsMatchTheCpu(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        for (int[] dims : new int[][] {{1500, 40}, {9, 300}, {2000, 33}, {7, 31}, {1100, 700}, {3, 5000}}) {
            Tensor x = random(dims[0] * 7L + dims[1], dims);
            String at = dims[0] + "x" + dims[1];
            same(type, "sum " + at, t -> t.sum(1, true), x);
            same(type, "mean " + at, t -> t.mean(1, true), x);
            same(type, "max " + at, t -> t.max(1, true), x);
            same(type, "min " + at, t -> t.min(1, true), x);
            same(type, "sum over rows " + at, t -> t.sum(0, true), x);
        }
    }

    /** Few long rows (the logits of a sampled token): by work groups, in pieces; the first largest wins. */
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT16"})
    void argmaxOfLongRowsMatchesTheCpu(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        for (int[] dims : new int[][] {{1, 151_936}, {3, 5000}, {8, 1000}, {2, 300}, {700, 257}}) {
            Tensor x = random(dims[0] * 31L + dims[1], dims);
            same(type, "argmax " + dims[0] + "x" + dims[1], t -> t.argmax(1).to(DType.FLOAT32), x);
            same(type, "argmax over rows " + dims[0] + "x" + dims[1], t -> t.argmax(0).to(DType.FLOAT32),
                    x.slice(1, 0, Math.min(dims[1], 300)));
        }
        // ties go to the first; NaN is never the largest, and a leading NaN gives 0
        float[] row = new float[4000];
        row[1234] = 5;
        row[3999] = 5;
        row[2000] = Float.NaN;
        assertEquals(1234f, Tensor.of(row).reshape(1, 4000).to(gpu).argmax(1).to(DType.FLOAT32).item());
        row[0] = Float.NaN;
        assertEquals(0f, Tensor.of(row).reshape(1, 4000).to(gpu).argmax(1).to(DType.FLOAT32).item());
    }

    /** Transposed and sliced operands, read in place by the tiled product. */
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT16"})
    void productsOfTransposedOperandsMatchTheCpu(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        for (int[] mkn : new int[][] {{70, 33, 45}, {130, 1500, 20}, {9, 64, 300}, {200, 70, 130}}) {
            int m = mkn[0];
            int k = mkn[1];
            int n = mkn[2];
            Tensor at = random(m + 3L * k, k, m);   // a, stored transposed
            Tensor bt = random(n + 5L * k, n, k);   // b, stored transposed
            Tensor a = random(m + 7L * k, m, k);
            Tensor b = random(n + 11L * k, k, n);
            String at_ = m + "x" + k + "x" + n;
            same(type, "aT b " + at_, t -> t.transpose().matmul(b.to(type).to(t.device())), at);
            same(type, "a bT " + at_, t -> a.to(type).to(t.device()).matmul(t.transpose()), bt);
            same(type, "aT bT " + at_, t -> t.transpose().matmul(bt.to(type).to(t.device()).transpose()), at);
            Tensor batched = random(m * 13L + k, 3, k, m);
            same(type, "batched aT b " + at_, t -> t.transpose().matmul(b.to(type).to(t.device())
                    .reshape(1, k, n).expand(Shape.of(3, k, n)).add(0)), batched);
            same(type, "sliced a " + at_, t -> t.slice(1, 1, k - 1).matmul(b.to(type).to(t.device())
                    .slice(0, 1, k - 1)), a);
        }
    }

    /** A few rows times a transposed matrix (the tied output head while decoding), read in place. */
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT16"})
    void fewRowsTimesATransposedMatrixMatchTheCpu(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        for (int[] mkn : new int[][] {{1, 64, 300}, {2, 33, 70}, {5, 128, 1001}, {8, 70, 9}, {3, 4, 5}}) {
            int m = mkn[0];
            int k = mkn[1];
            int n = mkn[2];
            Tensor a = random(m + 17L * k, m, k);
            Tensor e = random(n + 19L * k, n, k);   // stored [n, k]
            Tensor rows = random(n + 23L * k, n + 2, k);
            Tensor wide = random(n + 29L * k, n, k + 4);
            String at = m + "x" + k + "x" + n;
            same(type, "a eT " + at, t -> a.to(type).to(t.device()).matmul(t.transpose()), e);
            same(type, "a sliced rows eT " + at, t -> a.to(type).to(t.device()).matmul(t.slice(0, 1, n + 1)
                    .transpose()), rows);
            same(type, "a sliced columns eT " + at, t -> a.to(type).to(t.device()).matmul(t.slice(1, 4, k + 4)
                    .transpose()), wide);
            same(type, "a odd sliced columns eT " + at, t -> a.to(type).to(t.device()).matmul(t.slice(1, 1, k + 1)
                    .transpose()), wide);
        }
    }

    /** Embedding gradients of a small vocabulary: long index lists, added up in slices. */
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT16"})
    void scatterAddOfLongIndexListsMatchesTheCpu(DType type) {
        assumeTrue(supported(type), type + " not supported by " + gpu);
        for (int[] dims : new int[][] {{4000, 16, 5}, {3000, 130, 65}, {700, 3, 2}, {2, 4000, 7}}) {
            int count = dims[0];
            int length = dims[2];
            float[] idx = new float[count];
            java.util.Random r = new java.util.Random(count);
            for (int i = 0; i < count; i++) {
                idx[i] = i % 3 == 0 ? 0 : r.nextInt(length);   // one frequent target
            }
            Tensor indices = Tensor.of(idx);
            Tensor x = random(count + length, count, dims[1]);
            same(type, "scatterAdd " + count + "x" + dims[1] + " into " + length,
                    t -> t.scatterAdd(0, indices.to(t.device()), length), x);
            Tensor y = random(count * 3L, 2, count, dims[1]);
            same(type, "scatterAdd middle " + count + "x" + dims[1] + " into " + length,
                    t -> t.scatterAdd(1, indices.to(t.device()), length), y);
        }
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
        String status = Device.available().stream().map(Device::name).filter(n -> n.startsWith("cuda"))
                .findFirst().orElseThrow();
        assertTrue(status.length() > "cuda".length(), status);
        assertTrue(CudaBackend.loaded().orElseThrow().status().contains("NVRTC"));
        assertTrue(List.of(DType.values()).size() == 4);
    }

    @ParameterizedTest
    /** One kernel for RMSNorm, against the operations Llama builds it from; the input a pending sum. */
    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"1, 576", "5, 2048", "3, 100", "2, 7"})
    void rmsNormMatchesItsOperations(int rows, int cols) {
        Tensor a = random(41, rows, cols);
        Tensor b = random(42, rows, cols);
        Tensor w = random(43, cols);
        for (DType type : List.of(DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16)) {
            if (!supported(type)) {
                continue;
            }
            Var x = Var.of(a.to(type).add(b.to(type)));
            Var x32 = x.to(DType.FLOAT32);
            float[] expected = x32.div(x32.square().mean(1, true).add(1e-5f).sqrt()).to(type).mul(Var.of(w.to(type)))
                    .value().toArray();
            Tensor xg = a.to(type).to(gpu).add(b.to(type).to(gpu));
            assertTrue(xg.hasRmsNorm());
            float[] fused = xg.rmsNorm(w.to(gpu), 1e-5f).toArray();
            float tolerance = type == DType.FLOAT32 ? 1e-5f : type == DType.FLOAT16 ? 4e-3f : 3e-2f;
            for (int i = 0; i < fused.length; i++) {
                assertEquals(expected[i], fused[i], tolerance * (1 + Math.abs(expected[i])), type + " at " + i);
            }
        }
    }

    /** Decoding's products of a few rows: one pass over the matrix, in pieces of rows added up. */
    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"1, 896, 896", "1, 4864, 896", "2, 64, 7", "3, 1000, 130", "5, 17, 4",
        "8, 2048, 516", "1, 3, 1"})
    void productsOfFewRowsMatchTheCpu(int m, int k, int n) {
        Tensor a = random(61, m, k);
        Tensor b = random(62, k, n);
        for (DType type : List.of(DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16)) {
            if (!supported(type)) {
                continue;
            }
            float[] expected = a.to(type).to(DType.FLOAT32).matmul(b.to(type).to(DType.FLOAT32)).toArray();
            float[] actual = a.to(type).to(gpu).matmul(b.to(type).to(gpu)).toArray();
            double tolerance = type == DType.FLOAT32 ? 1e-4 : type == DType.FLOAT16 ? 4e-3 : 3e-2;
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], actual[i], tolerance * (Math.sqrt(k) + Math.abs(expected[i])),
                        type + " " + m + "x" + k + "x" + n + " at " + i);
            }
        }
    }

    /** Fourier transforms in shared memory for powers of two up to 2048, on the host otherwise. */
    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 2, 8, 64, 2048, 12, 4096})
    void fourierTransformsMatchTheCpu(int n) {
        Tensor x = random(71, 3, n, 2);
        for (boolean inverse : new boolean[] {false, true}) {
            float[] expected = x.fft(inverse).toArray();
            float[] actual = x.to(gpu).fft(inverse).toArray();
            float tolerance = (inverse ? 1e-5f : 1e-5f * n) * (float) Math.log(n + 1) + 1e-5f;
            assertArrayEquals(expected, actual, tolerance, (inverse ? "inverse " : "") + n);
        }
        if (supported(DType.FLOAT16)) {
            Tensor h = x.to(DType.FLOAT16);
            assertArrayEquals(h.fft(false).toArray(), h.to(gpu).fft(false).toArray(), 2e-3f * n + 1e-2f, "f16 " + n);
        }
    }

    /** Float16 products: on cooperative matrices where the device has them (see the status). */
    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"1, 16, 16, 16", "1, 64, 64, 64", "1, 17, 33, 70", "3, 100, 48, 130",
        "1, 16, 4096, 24", "2, 300, 520, 260", "1, 1, 64, 64", "1, 128, 5, 128"})
    void float16ProductsMatchTheCpu(int batch, int m, int k, int n) {
        assumeTrue(supported(DType.FLOAT16), "float16 not supported by " + gpu);
        Tensor a = random(5, batch, m, k).to(DType.FLOAT16);
        Tensor b = random(6, batch, k, n).to(DType.FLOAT16);
        float[] expected = a.to(DType.FLOAT32).matmul(b.to(DType.FLOAT32)).toArray();   // exact sums of the halves
        float[] actual = a.to(gpu).matmul(b.to(gpu)).toArray();
        // rounding the result to float16, and sums in float32 in another order
        double tolerance = 2e-3 * Math.sqrt(k) + 1e-3;
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance + Math.abs(expected[i]) * 1e-3,
                    batch + "x" + m + "x" + k + "x" + n + " at " + i + "; " + gpu);
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"256, 64", "16, 1024", "64, 4", "8, 36", "12, 6"})
    void quantizesOnTheDeviceAsTheCpuDoes(int rows, int cols) {
        Tensor x = random(11, rows, cols).mul(3);
        x = x.mul(Tensor.uniform(Shape.of(1, cols), 0, 1, new SplittableRandom(2)));   // a column of near zeros
        int group = rows % 8 == 0 ? rows / 2 : rows;
        PackedMatrix expected = PackedMatrix.quantize(x.toArray(), rows, cols, 8, group);
        Tensor q = x.to(gpu).quantized(8, group);
        assertTrue(q.isPacked());
        float[] values = new float[rows * cols];
        expected.dequantize(values);
        float[] actual = q.toArray();
        float[] scales = expected.scales();
        for (int i = 0; i < values.length; i++) {
            // the GPU divides to within a few ulps: a level may round the other way at .5
            float step = scales[i / cols / group * cols + i % cols];
            assertEquals(values[i], actual[i], step * 1.001f + 1e-6f, rows + "x" + cols + " at " + i);
        }
        Tensor back = q.to(CPU);   // levels read back from the device
        assertArrayEquals(actual, back.toArray(), 0f);
        Tensor again = Tensor.packed(PackedMatrix.quantize(new float[rows * cols], rows, cols, 8, group), gpu);
        assertArrayEquals(new float[rows * cols], again.toArray(), 0f);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"8, 64, 1", "8, 64, 5", "4, 64, 1", "4, 32, 8", "3, 48, 2",
        "4, 3, 9", "8, 64, 20", "8, 1536, 1", "4, 32, 3", "2, 2, 4"})
    void packedMatricesMatchTheCpu(int bits, int group, int m) {
        packedMatches(bits, group, m, group > 192 ? 3072 : 192, 77, false);
        packedMatches(bits, group, m, 768, 256, false);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"4, 32, 1", "5, 32, 3", "6, 16, 8", "2, 16, 2", "8, 32, 12"})
    void packedMatricesWithOffsetsMatchTheCpu(int bits, int group, int m) {
        packedMatches(bits, group, m, 192, 77, true);
        packedMatches(bits, group, m, 768, 256, true);
    }

    /** Scales and offsets factored as GGUF's k-quants store them, kept factored on the device. */
    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"4, 32, 1, true", "5, 32, 3, true", "6, 16, 8, false",
            "2, 16, 2, true", "3, 16, 4, false", "8, 32, 12, false", "4, 32, 2, false", "5, 32, 1, false"})
    void packedMatricesWithFactoredScalesMatchTheCpu(int bits, int group, int m, boolean withOffsets) {
        packedMatches(bits, group, m, 192, 77, withOffsets, true);
        packedMatches(bits, group, m, 768, 256, withOffsets, true);
    }

    /** 8-bit levels of at most 16 values (IQ4_NL's table), kept as 4-bit indices on the device. */
    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 3, 8, 12})
    void packedMatricesOfFewLevelsMatchTheCpu(int m) {
        int[] table = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        for (int[] dims : new int[][] {{192, 77}, {768, 256}}) {
            int rows = dims[0];
            int cols = dims[1];
            SplittableRandom r = new SplittableRandom(m + rows);
            byte[] levels = new byte[rows * cols];
            for (int i = 0; i < levels.length; i++) {
                levels[i] = (byte) table[r.nextInt(16)];
            }
            float[] scales = new float[rows / 32 * cols];
            for (int i = 0; i < scales.length; i++) {
                scales[i] = Float.float16ToFloat(Float.floatToFloat16((float) r.nextDouble(1e-4, 1e-3)));
            }
            Tensor onCpu = Tensor.packed(PackedMatrix.of(levels, scales, rows, cols, 8, 32), CPU);
            Tensor onGpu = onCpu.to(gpu);
            Tensor x = random(5, m, rows);
            float[] expected = x.matmul(onCpu).toArray();
            double scale = 1;
            for (float v : expected) {
                scale = Math.max(scale, Math.abs(v));
            }
            String what = "table levels, " + m + " x " + rows + " x " + cols;
            org.junit.jupiter.api.Assertions.assertArrayEquals(onCpu.toArray(), onGpu.toArray(), 1e-6f, what);
            org.junit.jupiter.api.Assertions.assertArrayEquals(expected, x.to(gpu).matmul(onGpu).toArray(),
                    (float) (5e-5 * scale), what);
            assertTrue(onGpu.bytes()
                    < (long) rows * cols * 3 / 4, what + ": 4-bit indices and float16 scales");
        }
    }

    private static void packedMatches(int bits, int group, int m, int rows, int cols, boolean withOffsets) {
        packedMatches(bits, group, m, rows, cols, withOffsets, false);
    }

    private static void packedMatches(int bits, int group, int m, int rows, int cols, boolean withOffsets,
            boolean factored) {
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
        float[] offsets = null;
        if (withOffsets) {
            offsets = new float[scales.length];
            for (int i = 0; i < offsets.length; i++) {
                offsets[i] = (float) r.nextDouble(-0.2, 0.2);
            }
        }
        PackedMatrix w = PackedMatrix.of(levels, scales, offsets, rows, cols, bits, group);
        if (factored) {
            // codes per group times float16 values per super-group; offsets add a multiple of the scale
            int superGroup = rows % 256 == 0 ? 256 : 2 * group;
            boolean codes = bits != 4 || withOffsets;   // 4 bits without offsets: one value per group, as Q4_0
            if (!codes) {
                superGroup = group;
            }
            byte[] scaleCodes = codes ? new byte[scales.length] : null;
            byte[] offsetCodes = codes && withOffsets ? new byte[scales.length] : null;
            float[] superScales = new float[rows / superGroup * cols];
            float[] superOffsets = withOffsets ? new float[superScales.length] : null;
            for (int i = 0; i < scales.length; i++) {
                if (scaleCodes != null) {
                    scaleCodes[i] = (byte) (bits == 6 || bits == 3 ? r.nextInt(-32, 32) : r.nextInt(0, 64));
                }
                if (offsetCodes != null) {
                    offsetCodes[i] = (byte) r.nextInt(0, 64);
                }
            }
            for (int i = 0; i < superScales.length; i++) {
                superScales[i] = Float.float16ToFloat(Float.floatToFloat16((float) r.nextDouble(1e-4, 1e-3)));
                if (superOffsets != null) {
                    superOffsets[i] = Float.float16ToFloat(Float.floatToFloat16((float) -r.nextDouble(1e-4, 3e-3)));
                }
            }
            int shift = withOffsets ? 1 << (bits - 1) : 0;
            w = PackedMatrix.of(levels, new PackedMatrix.Factors(scaleCodes, superScales, offsetCodes, superOffsets,
                    superGroup, shift), rows, cols, bits, group);
        }
        Tensor x = random(3, m, rows);
        Tensor onCpu = Tensor.packed(w, CPU);
        Tensor onGpu = onCpu.to(gpu);
        String what = bits + " bits, groups of " + group + ", " + m + " x " + rows + " x " + cols
                + (factored ? ", factored" : "");
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
                Tensor product = x.to(type).to(gpu).matmul(onGpu);
                assertEquals(type, product.dtype(), "packed weights multiply in the activations' type");
                float[] actual = product.toArray();
                double tolerance = (type == DType.FLOAT64 ? 1e-5 : type == DType.FLOAT16 ? 4e-3 : 3e-2) * scale;
                for (int i = 0; i < expected.length; i++) {
                    assertEquals(expected[i], actual[i], tolerance, what + ", " + type + " at " + i);
                }
            }
        }
    }
}
