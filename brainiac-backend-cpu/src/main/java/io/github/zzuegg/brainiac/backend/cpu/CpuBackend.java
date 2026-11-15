package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.util.Arrays;

/**
 * Runs tensor operations on the CPU in plain Java. Large operations are split across
 * threads (set {@code -Dbrainiac.cpu.threads=N} to limit them); results are the same for
 * any number of threads. Storage is a {@code float[]}, so a tensor holds at most about 2
 * billion values.
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
        if (size > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("the CPU backend holds at most " + (Integer.MAX_VALUE - 8)
                    + " values per tensor, requested " + size);
        }
        return new CpuStorage(new float[(int) size]);
    }

    @Override
    public Storage upload(float[] data) {
        return new CpuStorage(data.clone());
    }

    @Override
    public float[] download(Storage src) {
        return data(src).clone();
    }

    @Override
    public void fill(Storage dst, float value) {
        Arrays.fill(data(dst), value);
    }

    @Override
    public void unary(UnaryOp op, Storage in, Storage out) {
        float[] x = data(in);
        float[] y = data(out);
        long cost = switch (op) {
            case EXP, LOG, TANH, SIGMOID, ERF, SOFTPLUS -> 20;
            case SQRT -> 4;
            default -> 1;
        };
        Parallel.forRange(x.length, cost, (from, to) -> Kernels.unary(op, x, y, from, to));
    }

    @Override
    public void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape, Storage out) {
        float[] x = data(a);
        float[] y = data(b);
        float[] z = data(out);
        int rank = shape.length;
        long[] contiguous = contiguousStrides(shape);
        long cost = op == BinaryOp.POW ? 20 : 1;
        boolean aContiguous = Arrays.equals(aStrides, contiguous);
        boolean bContiguous = Arrays.equals(bStrides, contiguous);

        if (aContiguous && bContiguous) {
            Parallel.forRange(z.length, cost, (from, to) -> Kernels.binary(op, x, y, z, from, to));
        } else if (aContiguous && isZero(bStrides)) {
            float s = y[0];
            Parallel.forRange(z.length, cost, (from, to) -> Kernels.binaryScalarRight(op, x, s, z, from, to));
        } else if (bContiguous && isZero(aStrides)) {
            float s = x[0];
            Parallel.forRange(z.length, cost, (from, to) -> Kernels.binaryScalarLeft(op, s, y, z, from, to));
        } else if (aContiguous && rank >= 2 && isRowBroadcast(bStrides)) {
            int n = shape[rank - 1];
            Parallel.forRange(z.length / n, cost * n, (from, to) -> Kernels.binaryRows(op, x, y, z, n, from, to));
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
        float[] x = data(a);
        float[] y = data(b);
        float[] z = data(out);
        Arrays.fill(z, 0, m * n, 0f);
        if (SIMD) {
            Parallel.forRange(m, 2L * k * n, (from, to) -> VectorKernels.matmulRows(x, y, z, k, n, from, to));
        } else {
            Parallel.forRange(m, 2L * k * n, (from, to) -> Kernels.matmulRows(x, y, z, k, n, from, to));
        }
    }

    /**
     * Returns whether matrix multiplication uses SIMD instructions through the Vector API.
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
        if (s instanceof CpuStorage c) {
            return c.data;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to the CPU backend");
    }
}
