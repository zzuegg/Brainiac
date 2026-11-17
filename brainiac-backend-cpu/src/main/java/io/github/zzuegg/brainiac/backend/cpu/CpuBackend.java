package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.PackedStorage;
import io.github.zzuegg.brainiac.tensor.spi.QuantizedRows;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.RawValues;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.util.Arrays;

/**
 * Runs tensor operations on the CPU in plain Java. Large operations are split across
 * threads (set {@code -Dbrainiac.cpu.threads=N} to limit them); results are the same for
 * any number of threads. Storage is a Java array, so a tensor holds at most about 2
 * billion values.
 *
 * <p>All four {@link DType}s are supported. Float32 runs the fast (SIMD where available)
 * kernels; float64, float16 and bfloat16 run simpler kernels in double precision, and
 * float16 and bfloat16 are rounded once when results are stored.
 */
public final class CpuBackend implements Backend {

    private static final boolean SIMD = simdAvailable();

    /** Creates the backend; called by {@link java.util.ServiceLoader}. */
    public CpuBackend() {
    }

    @Override
    public String name() {
        return "cpu";
    }

    @Override
    public Device.Kind kind() {
        return Device.Kind.CPU;
    }

    @Override
    public Storage allocate(long size) {
        return allocate(size, DType.FLOAT32);
    }

    @Override
    public boolean supports(DType dtype) {
        return true;
    }

    @Override
    public Storage allocate(long size, DType dtype) {
        if (size > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("the CPU backend holds at most " + (Integer.MAX_VALUE - 8)
                    + " values per tensor, requested " + size);
        }
        return CpuStorage.allocate((int) size, dtype);
    }

    @Override
    public Storage upload(float[] data) {
        return new CpuStorage(data.clone());
    }

    @Override
    public Storage upload(double[] data, DType dtype) {
        CpuStorage s = CpuStorage.allocate(data.length, dtype);
        s.write(data);
        return s;
    }

    @Override
    public Storage upload(java.lang.foreign.MemorySegment data, DType dtype) {
        long n = data.byteSize() / dtype.bytes();
        CpuStorage s = (CpuStorage) allocate(n, dtype);
        switch (dtype) {
            case FLOAT32 -> java.lang.foreign.MemorySegment.copy(data, RawValues.FLOAT, 0, s.data, 0, s.data.length);
            case FLOAT64 -> java.lang.foreign.MemorySegment.copy(data, RawValues.DOUBLE, 0, s.wide, 0, s.wide.length);
            case FLOAT16, BFLOAT16 ->
                    java.lang.foreign.MemorySegment.copy(data, RawValues.SHORT, 0, s.half, 0, s.half.length);
        }
        return s;
    }

    @Override
    public float[] download(Storage src) {
        return cpu(src).readFloat();
    }

    @Override
    public double[] downloadDouble(Storage src) {
        CpuStorage s = cpu(src);
        return s.wide != null ? s.wide.clone() : s.read();
    }

    @Override
    public void convert(Storage in, Storage out) {
        cpu(out).write(cpu(in).read());
    }

    @Override
    public void fill(Storage dst, float value) {
        CpuStorage s = cpu(dst);
        if (s.data != null) {
            Arrays.fill(s.data, value);
        } else {
            double[] v = s.target();
            Arrays.fill(v, value);
            s.write(v);
        }
    }

    @Override
    public void unary(UnaryOp op, Storage in, Storage out) {
        if (wide(in, out)) {
            CpuStorage o = cpu(out);
            double[] y = o.target();
            WideKernels.unary(op, cpu(in).read(), y);
            o.write(y);
            return;
        }
        float[] x = data(in);
        float[] y = data(out);
        long cost = switch (op) {
            case EXP, LOG, TANH, SIGMOID, ERF, SOFTPLUS, SIN, COS -> 20;
            case SQRT -> 4;
            default -> 1;
        };
        Parallel.forRange(x.length, cost, (from, to) -> Kernels.unary(op, x, y, from, to));
    }

    @Override
    public void activation(ActivationOp op, float param, Storage in, Storage out) {
        if (wide(in, out)) {
            CpuStorage o = cpu(out);
            double[] y = o.target();
            WideKernels.activation(op, param, cpu(in).read(), y);
            o.write(y);
            return;
        }
        float[] x = data(in);
        float[] y = data(out);
        if (SIMD && VectorActivationKernels.supports(op)) {
            Parallel.forRange(x.length, ActivationKernels.cost(op),
                    (from, to) -> VectorActivationKernels.forward(op, param, x, y, from, to));
        } else {
            Parallel.forRange(x.length, ActivationKernels.cost(op),
                    (from, to) -> ActivationKernels.forward(op, param, x, y, from, to));
        }
    }

    @Override
    public void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        if (wide(in, grad, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.activationBackward(op, param, cpu(in).read(), cpu(grad).read(), z);
            o.write(z);
            return;
        }
        float[] x = data(in);
        float[] g = data(grad);
        float[] z = data(out);
        if (SIMD && VectorActivationKernels.supports(op)) {
            Parallel.forRange(x.length, ActivationKernels.cost(op),
                    (from, to) -> VectorActivationKernels.backward(op, param, x, g, z, from, to));
        } else {
            Parallel.forRange(x.length, ActivationKernels.cost(op),
                    (from, to) -> ActivationKernels.backward(op, param, x, g, z, from, to));
        }
    }

    @Override
    public void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape, Storage out) {
        if (wide(a, b, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.binary(op, cpu(a).read(), aStrides, cpu(b).read(), bStrides, shape, z);
            o.write(z);
            return;
        }
        float[] x = data(a);
        float[] y = data(b);
        float[] z = data(out);
        int rank = shape.length;
        long[] contiguous = contiguousStrides(shape);
        long cost = op == BinaryOp.POW ? 20 : 1;
        boolean aContiguous = Arrays.equals(aStrides, contiguous);
        boolean bContiguous = Arrays.equals(bStrides, contiguous);

        boolean simd = SIMD && VectorActivationKernels.supports(op);
        if (aContiguous && bContiguous) {
            if (simd) {
                Parallel.forRange(z.length, cost, (from, to) -> VectorActivationKernels.binary(op, x, y, z, from, to));
            } else {
                Parallel.forRange(z.length, cost, (from, to) -> Kernels.binary(op, x, y, z, from, to));
            }
        } else if (aContiguous && isZero(bStrides)) {
            float s = y[0];
            if (simd) {
                Parallel.forRange(z.length, cost,
                        (from, to) -> VectorActivationKernels.binaryScalarRight(op, x, s, z, from, to));
            } else {
                Parallel.forRange(z.length, cost, (from, to) -> Kernels.binaryScalarRight(op, x, s, z, from, to));
            }
        } else if (bContiguous && isZero(aStrides)) {
            float s = x[0];
            Parallel.forRange(z.length, cost, (from, to) -> Kernels.binaryScalarLeft(op, s, y, z, from, to));
        } else if (aContiguous && rank >= 2 && isRowBroadcast(bStrides)) {
            int n = shape[rank - 1];
            if (simd) {
                Parallel.forRange(z.length / n, cost * n,
                        (from, to) -> VectorActivationKernels.binaryRows(op, x, y, z, n, from, to));
            } else {
                Parallel.forRange(z.length / n, cost * n, (from, to) -> Kernels.binaryRows(op, x, y, z, n, from, to));
            }
        } else {
            Parallel.forRange(z.length, cost + 4,
                    (from, to) -> Kernels.binaryStrided(op, x, aStrides, y, bStrides, shape, z, from, to));
        }
    }

    /** Strides of a vector broadcast over all leading axes: {@code [0, ..., 0, 1]}. */
    private static boolean isRowBroadcast(long[] strides) {
        for (int d = 0; d < strides.length - 1; d++) {
            if (strides[d] != 0) {
                return false;
            }
        }
        return strides[strides.length - 1] == 1;
    }

    @Override
    public void reduce(ReduceOp op, Storage in, long outer, long length, long inner, Storage out) {
        if (wide(in, out)) {
            CpuStorage o = cpu(out);
            double[] y = o.target();
            WideKernels.reduce(op, cpu(in).read(), (int) outer, (int) length, (int) inner, y);
            o.write(y);
            return;
        }
        float[] x = data(in);
        float[] y = data(out);
        int out0 = (int) outer;
        int len = (int) length;
        int inn = (int) inner;
        if (out0 >= Parallel.threads() || inn == 1) {
            Parallel.forRange(out0, (long) len * inn, (from, to) -> {
                for (int o = from; o < to; o++) {
                    Kernels.reduce(op, x, y, o, len, inn, 0, inn);
                }
            });
        } else {
            for (int o = 0; o < out0; o++) {
                int block = o;
                Parallel.forRange(inn, len, (from, to) -> Kernels.reduce(op, x, y, block, len, inn, from, to));
            }
        }
    }

    @Override
    public void matmul(Storage a, Storage b, int m, int k, int n, Storage out) {
        if (wide(a, b, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.matmul(cpu(a).read(), cpu(b).read(), z, 1, m, k, n, true);
            o.write(z);
            return;
        }
        float[] x = data(a);
        float[] y = data(b);
        float[] z = data(out);
        Arrays.fill(z, 0, m * n, 0f);
        if (m < 2 * Parallel.threads() && n >= 2 * COLUMN_CHUNK) {
            // too few rows to share among the cores: split the columns instead
            int chunks = Math.ceilDiv(n, COLUMN_CHUNK);
            Parallel.forRange(chunks, 2L * m * k * COLUMN_CHUNK, (from, to) -> {
                int j0 = from * COLUMN_CHUNK;
                int j1 = Math.min(n, to * COLUMN_CHUNK);
                if (SIMD) {
                    VectorKernels.matmulColumns(x, y, z, m, k, n, j0, j1);
                } else {
                    Kernels.matmulColumns(x, y, z, m, k, n, j0, j1);
                }
            });
        } else if (SIMD) {
            Parallel.forRange(m, 2L * k * n, (from, to) -> VectorKernels.matmulRows(x, y, z, k, n, from, to));
        } else {
            Parallel.forRange(m, 2L * k * n, (from, to) -> Kernels.matmulRows(x, y, z, k, n, from, to));
        }
    }

    /** Columns per task when a product of few rows is split by columns. */
    private static final int COLUMN_CHUNK = 128;

    /** Rows up to which a product reads packed levels directly; above, it unpacks first. */
    private static final int PACKED_ROWS = Integer.getInteger("brainiac.cpu.packedRows", 16);

    @Override
    public void unpack(PackedStorage packed, Storage out) {
        CpuStorage o = cpu(out);
        if (packed instanceof QuantizedRows rows) {
            if (o.data != null) {
                rows.dequantize(o.data);
                return;
            }
            Backend.super.unpack(packed, out);
            return;
        }
        PackedMatrix m = unpackToHost(packed);
        if (o.data != null) {
            m.dequantize(o.data);
        } else {
            float[] f = new float[Math.toIntExact(m.size())];
            m.dequantize(f);
            double[] v = o.target();
            for (int i = 0; i < f.length; i++) {
                v[i] = f[i];
            }
            o.write(v);
        }
    }

    @Override
    public void matmulPacked(Storage a, PackedStorage b, int m, int k, int n, Storage out) {
        if (m > PACKED_ROWS || wide(a, out)) {
            Backend.super.matmulPacked(a, b, m, k, n, out);
            return;
        }
        PackedMatrix w = unpackToHost(b);
        float[] x = data(a);
        float[] z = data(out);
        int chunks = Math.ceilDiv(n, PackedKernels.CHUNK);
        boolean simd = SIMD && VectorPackedKernels.usable();
        Parallel.forRange(chunks, 2L * m * k * PackedKernels.CHUNK, (from, to) -> {
            for (int c = from; c < to; c++) {
                int j0 = c * PackedKernels.CHUNK;
                int j1 = Math.min(n, j0 + PackedKernels.CHUNK);
                if (simd) {
                    VectorPackedKernels.matmul(x, w, z, m, j0, j1);
                } else {
                    PackedKernels.matmul(x, w, z, m, j0, j1);
                }
                if (w.offsets() != null) {
                    PackedKernels.addOffsets(x, w, z, m, j0, j1);
                }
            }
        });
    }

    @Override
    public void matmulBatched(Storage a, Storage b, int batch, int m, int k, int n, Storage out) {
        if (wide(a, b, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.matmul(cpu(a).read(), cpu(b).read(), z, batch, m, k, n, false);
            o.write(z);
            return;
        }
        float[] x = data(a);
        float[] y = data(b);
        float[] z = data(out);
        Parallel.forRange(batch * m, 2L * k * n, (from, to) -> WindowKernels.matmulBatched(x, y, z, m, k, n, from, to));
    }

    @Override
    public void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        if (wide(in, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.im2col(cpu(in).read(), new WindowKernels.Plan(geometry), geometry.batch(), padValue, z);
            o.write(z);
            return;
        }
        float[] x = data(in);
        float[] z = data(out);
        WindowKernels.Plan plan = new WindowKernels.Plan(geometry);
        int[] table = plan.table();
        int rows = geometry.batch() * plan.positions;
        Parallel.forRange(rows, (long) plan.channels * plan.kernelSize,
                (from, to) -> WindowKernels.im2col(x, plan, table, padValue, z, from, to));
    }

    @Override
    public void col2im(Storage cols, ConvGeometry geometry, Storage out) {
        if (wide(cols, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.col2im(cpu(cols).read(), new WindowKernels.Plan(geometry), geometry.batch(), z);
            o.write(z);
            return;
        }
        float[] x = data(cols);
        float[] z = data(out);
        WindowKernels.Plan plan = new WindowKernels.Plan(geometry);
        int[] table = plan.table();
        int planes = geometry.batch() * plan.channels;
        Parallel.forRange(planes, (long) plan.positions * plan.kernelSize,
                (from, to) -> WindowKernels.col2im(x, plan, table, z, from, to));
    }

    @Override
    public void gather(Storage in, long outer, long length, long inner, Storage indices, int count, Storage out) {
        int[] idx = indices(indices, count, (int) length);
        if (wide(in, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.gather(cpu(in).read(), (int) outer, (int) length, (int) inner, idx, z);
            o.write(z);
            return;
        }
        float[] x = data(in);
        float[] z = data(out);
        int in0 = (int) inner;
        int len = (int) length;
        Parallel.forRange((int) outer * count, in0, (from, to) -> WindowKernels.gather(x, len, in0, idx, z, from, to));
    }

    @Override
    public void scatterAdd(Storage src, long outer, int count, long inner, Storage indices, long length,
            Storage out) {
        int[] idx = indices(indices, count, (int) length);
        if (wide(src, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.target();
            WideKernels.scatterAdd(cpu(src).read(), (int) outer, (int) length, (int) inner, idx, z);
            o.write(z);
            return;
        }
        float[] x = data(src);
        float[] z = data(out);
        int out0 = (int) outer;
        int len = (int) length;
        int in0 = (int) inner;
        // columns of the inner block never collide, so they split safely across threads
        Parallel.forRange(in0, (long) out0 * (count + len),
                (from, to) -> WindowKernels.scatterAdd(x, out0, len, in0, idx, z, from, to));
    }

    @Override
    public boolean hasAttention(int[] shape, DType type, boolean backward) {
        return true;
    }

    @Override
    public void attention(Storage q, Storage k, Storage v, int[] shape, float scale, boolean causal, Storage out,
            Storage lse) {
        CpuStorage o = cpu(out);
        CpuStorage l = cpu(lse);
        double[] z = o.target();
        double[] e = l.target();
        AttentionKernels.forward(cpu(q).read(), cpu(k).read(), cpu(v).read(), shape, scale, causal, z, e);
        o.write(z);
        l.write(e);
    }

    @Override
    public void attentionBackward(Storage q, Storage k, Storage v, Storage out, Storage lse, Storage grad,
            int[] shape, float scale, boolean causal, Storage dq, Storage dk, Storage dv) {
        CpuStorage gq = cpu(dq);
        CpuStorage gk = cpu(dk);
        CpuStorage gv = cpu(dv);
        double[] zq = gq.target();
        double[] zk = gk.target();
        double[] zv = gv.target();
        AttentionKernels.backward(cpu(q).read(), cpu(k).read(), cpu(v).read(), cpu(out).read(), cpu(lse).read(),
                cpu(grad).read(), shape, scale, causal, zq, zk, zv);
        gq.write(zq);
        gk.write(zk);
        gv.write(zv);
    }

    private static int[] indices(Storage storage, int count, int length) {
        CpuStorage s = cpu(storage);
        float[] values = s.data != null ? s.data : s.readFloat();
        int[] idx = new int[count];
        for (int i = 0; i < count; i++) {
            idx[i] = WindowKernels.index(values[i], length);
        }
        return idx;
    }

    /**
     * Returns whether matrix multiplication, element-wise arithmetic and the arithmetic-only
     * activations use SIMD instructions through the Vector API.
     * That needs the {@code jdk.incubator.vector} module in the JVM (add it with
     * {@code --add-modules jdk.incubator.vector}); {@code -Dbrainiac.cpu.simd=false} turns
     * it off. Both paths give the same results for any thread count, but may differ from
     * each other in the last bit.
     *
     * @return {@code true} if SIMD kernels are used
     */
    public static boolean simd() {
        return SIMD;
    }

    private static boolean simdAvailable() {
        if (!Boolean.parseBoolean(System.getProperty("brainiac.cpu.simd", "true"))) {
            return false;
        }
        if (ModuleLayer.boot().findModule("jdk.incubator.vector").isEmpty()) {
            return false;
        }
        try {
            return VectorKernels.lanes() >= 4;
        } catch (LinkageError e) {
            return false;
        }
    }

    @Override
    public void copy(Storage in, long inOffset, long[] inStrides, int[] shape, Storage out, long outOffset,
            long[] outStrides) {
        if (wide(in, out)) {
            CpuStorage o = cpu(out);
            double[] z = o.wide != null ? o.wide : o.read();   // keep what the copy does not cover
            WideKernels.copy(cpu(in).read(), inOffset, inStrides, shape, z, outOffset, outStrides);
            o.write(z);
            return;
        }
        float[] x = data(in);
        float[] z = data(out);
        int rank = shape.length;
        long count = 1;
        for (int d : shape) {
            count *= d;
        }
        if (count == 0) {
            return;
        }
        if (rank == 0) {
            z[(int) outOffset] = x[(int) inOffset];
            return;
        }
        int rows = rank >= 2 ? shape[rank - 2] : 1;
        int cols = shape[rank - 1];
        int leading = (int) (count / ((long) rows * cols));
        int tilesPerMatrix = Math.ceilDiv(rows, Kernels.COPY_TILE);
        Parallel.forRange(leading * tilesPerMatrix, (long) Kernels.COPY_TILE * cols, (from, to) ->
                Kernels.copyTiles(x, inOffset, inStrides, shape, z, outOffset, outStrides, from, to));
    }

    private static long[] contiguousStrides(int[] shape) {
        long[] s = new long[shape.length];
        long step = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = step;
            step *= shape[i];
        }
        return s;
    }

    private static boolean isZero(long[] strides) {
        for (long s : strides) {
            if (s != 0) {
                return false;
            }
        }
        return true;
    }

    private static float[] data(Storage s) {
        return cpu(s).data;
    }

    private static CpuStorage cpu(Storage s) {
        if (s instanceof CpuStorage c) {
            return c;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to the CPU backend");
    }

    /** @return whether any storage is not float32, which takes the double-precision path */
    private static boolean wide(Storage... storages) {
        for (Storage s : storages) {
            if (cpu(s).dtype != DType.FLOAT32) {
                return true;
            }
        }
        return false;
    }
}
