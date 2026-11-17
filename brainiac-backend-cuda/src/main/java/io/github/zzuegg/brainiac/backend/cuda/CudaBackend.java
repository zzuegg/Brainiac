package io.github.zzuegg.brainiac.backend.cuda;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.RawValues;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * A GPU backend for NVIDIA GPUs on CUDA, called through the Foreign Function and Memory API:
 * no JNI and no libraries beyond NVIDIA's own. It needs
 * <ul>
 * <li>the NVIDIA driver ({@code nvcuda.dll} or {@code libcuda.so.1}),</li>
 * <li>NVRTC, CUDA's runtime compiler, which compiles the kernels for the GPU at first use (the
 * result is cached in {@code ~/.cache/brainiac/cuda}),</li>
 * <li>optionally cuBLAS, for the matrix products; without it a simpler kernel of its own runs.</li>
 * </ul>
 * NVRTC and cuBLAS come with the CUDA toolkit (11.2 or later) or NVIDIA's CUDA runtime
 * container images. Without a driver or a device the backend is unavailable and Brainiac
 * uses the other devices; {@link #status()} says why.
 *
 * <p>Stores float32, float64, float16 and bfloat16 tensors; float16 and bfloat16 are computed
 * in float32 (their products in cuBLAS with float32 accumulation), float64 in double precision.
 * Every operation is queued on one stream, so the host only waits when it reads values back.
 *
 * <p>Run with {@code --enable-native-access=io.github.zzuegg.brainiac.backend.cuda} (or
 * {@code ALL-UNNAMED} on the class path). System properties:
 * <ul>
 * <li>{@code brainiac.cuda.device}: the device index, default 0</li>
 * <li>{@code brainiac.cuda.memory}: the memory budget in MB</li>
 * <li>{@code brainiac.cuda.debug=true}: wait after every operation, so errors name the
 * operation that caused them</li>
 * <li>{@code brainiac.cuda.critical=true}: link the per-operation driver and cuBLAS calls as
 * critical downcalls, which makes each call cheaper</li>
 * <li>{@code brainiac.cuda.tf32=true}: float32 products on tensor cores in TensorFloat-32</li>
 * <li>{@code brainiac.cuda.driver}, {@code brainiac.cuda.nvrtc}, {@code brainiac.cuda.cublas}:
 * the library files ({@code brainiac.cuda.cublas=none}: no cuBLAS)</li>
 * <li>{@code brainiac.cuda.cache}: where compiled kernels are kept, {@code none} for nowhere</li>
 * <li>{@code brainiac.cuda.statistics=true}: print {@link #statistics()} at exit</li>
 * </ul>
 */
public final class CudaBackend implements Backend {

    private static final Cleaner CLEANER = Cleaner.create();

    private CudaContext cuda;
    private boolean initialized;
    private String unavailable;
    private long initNanos;

    private final Map<String, Tables> tables = new LinkedHashMap<>(16, 0.75f, true) {
        private static final long serialVersionUID = 1L;

        @Override
        protected boolean removeEldestEntry(Map.Entry<String, Tables> eldest) {
            return size() > 32;
        }
    };

    /** Index tables of a window geometry, for {@link #im2col} and {@link #col2im}. */
    private record Tables(CudaStorage im2col, CudaStorage offsets, CudaStorage entries) {
    }

    private static CudaBackend first;

    /**
     * The first instance that found a device, normally the one behind the {@code cuda}
     * {@link Device}; for {@link #statistics()}.
     *
     * @return the instance, if any
     */
    public static synchronized Optional<CudaBackend> loaded() {
        return Optional.ofNullable(first);
    }

    /** Creates the backend; called by {@link java.util.ServiceLoader}. Does not touch the driver. */
    public CudaBackend() {
    }

    @Override
    public String name() {
        return cuda == null ? "cuda" : "cuda (" + cuda.deviceName + ")";
    }

    @Override
    public Device.Kind kind() {
        return Device.Kind.GPU;
    }

    @Override
    public synchronized boolean isAvailable() {
        if (!initialized) {
            initialized = true;
            long start = System.nanoTime();
            try {
                cuda = new CudaContext();
                initNanos = System.nanoTime() - start;
                synchronized (CudaBackend.class) {
                    if (first == null) {
                        first = this;
                    }
                }
                if (Boolean.getBoolean("brainiac.cuda.statistics")) {
                    Runtime.getRuntime().addShutdownHook(new Thread(() -> System.err.println(name() + ": "
                            + statistics()), "cuda-statistics"));
                }
            } catch (RuntimeException | LinkageError e) {
                unavailable = e.getMessage() == null ? e.toString() : e.getMessage();
                cuda = null;
                if (CudaContext.DEBUG) {
                    e.printStackTrace();
                }
            }
        }
        return cuda != null;
    }

    /**
     * Describes the device and libraries in use, or why the backend is unavailable.
     *
     * @return for example {@code "NVIDIA GeForce RTX 4090, compute 8.9, 24564 MB, ..."}
     */
    public synchronized String status() {
        if (!isAvailable()) {
            return "unavailable: " + unavailable;
        }
        CudaContext c = cuda;
        int driver = c.cu.driverVersion();
        return c.deviceName + ", compute " + c.computeCapability / 10 + "." + c.computeCapability % 10 + ", "
                + c.multiprocessors + " SMs, " + (c.deviceMemory >> 20) + " MB, " + (c.budget >> 20)
                + " MB budget; driver CUDA " + driver / 1000 + "." + driver % 1000 / 10 + " (" + c.cu.library
                + "); NVRTC " + c.nvrtc.version() + " (" + c.nvrtc.library + ", compute_" + c.architecture + ", "
                + (c.cachedKernels ? "cached" : "compiled in " + c.compileNanos / 1_000_000 + " ms") + "); cuBLAS "
                + (c.cublas != null ? c.cublas.version() + " (" + c.cublas.library + ")"
                        : "unavailable: " + c.cublasUnavailable)
                + (CudaContext.DEBUG ? "; debug" : "") + (Binder.CRITICAL ? "; critical downcalls" : "");
    }

    /**
     * Counts of work done so far, for tuning.
     *
     * @return launches, products, waits, copies and memory use
     */
    public synchronized String statistics() {
        if (cuda == null) {
            return "not initialized";
        }
        CudaContext c = cuda;
        return "init " + initNanos / 1_000_000 + " ms, " + c.launches + " kernel launches, " + c.products
                + " cuBLAS products, " + c.waits + " waits, " + (c.uploadedBytes >> 20) + " MB uploaded, "
                + (c.downloadedBytes >> 20) + " MB downloaded, " + (c.inUse() >> 20) + " MB in use, "
                + (c.peak() >> 20) + " MB peak, " + (c.reserved() >> 20) + " MB reserved, " + c.collections
                + " collections";
    }

    /** Waits until the device has run everything queued, for timing. */
    public synchronized void finish() {
        if (cuda != null) {
            cuda.bind();
            cuda.finish();
        }
    }

    @Override
    public synchronized Device.Memory memory() {
        if (cuda == null) {
            return null;
        }
        return new Device.Memory(cuda.inUse(), cuda.peak(), cuda.reserved(), cuda.budget);
    }

    @Override
    public synchronized void resetPeakMemory() {
        if (cuda != null) {
            cuda.resetPeak();
        }
    }

    // ------------------------------------------------------------------ storage

    /** The context, current on this thread. */
    private CudaContext ctx() {
        if (!isAvailable()) {
            throw new IllegalStateException("the cuda backend is unavailable: " + unavailable);
        }
        cuda.bind();
        return cuda;
    }

    private CudaStorage raw(long size, DType dtype, float[] host) {
        CudaContext c = ctx();
        return new CudaStorage(this, c, size, dtype, host, CLEANER);
    }

    private CudaStorage cs(Storage s) {
        if (s instanceof CudaStorage c && c.owner == this) {
            return c;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to " + name());
    }

    /** The kernel compiled for a number format. */
    private static String kernel(String name, DType dtype) {
        return name + switch (dtype) {
            case FLOAT32 -> "_f32";
            case FLOAT64 -> "_f64";
            case FLOAT16 -> "_f16";
            case BFLOAT16 -> "_bf16";
        };
    }

    /** {@code s} in another format: itself if it already is, else a converted copy to {@link #done}. */
    private CudaStorage as(CudaStorage s, DType dtype) {
        if (s.dtype == dtype) {
            return s;
        }
        CudaStorage c = raw(s.size, dtype, null);
        convert(s, c);
        return c;
    }

    /** Releases conversions made by {@link #as}: those that differ from the originals. */
    private static void done(CudaStorage converted, CudaStorage original) {
        if (converted != original) {
            converted.release();
        }
    }

    @Override
    public boolean supports(DType dtype) {
        return isAvailable();
    }

    @Override
    public synchronized Storage allocate(long size) {
        return allocate(size, DType.FLOAT32);
    }

    @Override
    public synchronized Storage allocate(long size, DType dtype) {
        CudaStorage s = raw(size, dtype, null);
        long bytes = s.bytes();
        if (bytes % 4 == 0) {
            cuda.fill32(s.address(), 0, bytes / 4);
        } else {
            cuda.fill8(s.address(), (byte) 0, bytes);
        }
        return s;
    }

    @Override
    public synchronized Storage allocateUninitialized(long size, DType dtype) {
        return raw(size, dtype, null);
    }

    @Override
    public synchronized void release(Storage storage) {
        if (storage instanceof CudaStorage s && s.owner == this) {
            s.release();
        }
    }

    @Override
    public synchronized Storage upload(float[] data) {
        CudaStorage s = raw(data.length, DType.FLOAT32, data.length <= 4096 ? data.clone() : null);
        cuda.upload(s.address(), MemorySegment.ofArray(data));
        return s;
    }

    @Override
    public synchronized Storage upload(double[] data, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> {
                float[] f = new float[data.length];
                for (int i = 0; i < f.length; i++) {
                    f[i] = (float) data[i];
                }
                yield upload(f);
            }
            case FLOAT64 -> {
                CudaStorage s = raw(data.length, dtype, null);
                cuda.upload(s.address(), MemorySegment.ofArray(data));
                yield s;
            }
            case FLOAT16, BFLOAT16 -> {
                short[] h = new short[data.length];
                for (int i = 0; i < h.length; i++) {
                    float v = (float) data[i];
                    h[i] = dtype == DType.FLOAT16 ? Float.floatToFloat16(v) : DType.floatToBfloat16(v);
                }
                CudaStorage s = raw(data.length, dtype, null);
                cuda.upload(s.address(), MemorySegment.ofArray(h));
                yield s;
            }
        };
    }

    @Override
    public synchronized Storage upload(MemorySegment data, DType dtype) {
        if (!RawValues.nativeLittleEndian()) {
            return Backend.super.upload(data, dtype);
        }
        CudaStorage s = raw(data.byteSize() / dtype.bytes(), dtype, null);
        cuda.upload(s.address(), data.asSlice(0, s.bytes()));
        return s;
    }

    private CudaStorage uploadInts(int[] data) {
        CudaStorage s = raw(Math.max(1, data.length), DType.FLOAT32, null);
        cuda.upload(s.address(), MemorySegment.ofArray(data));
        return s;
    }

    @Override
    public synchronized float[] download(Storage src) {
        CudaStorage s = cs(src);
        ctx();
        float[] out = new float[Math.toIntExact(s.size)];
        if (s.dtype == DType.FLOAT32) {
            cuda.download(MemorySegment.ofArray(out), s.address());
        } else {
            double[] d = downloadDouble(s);
            for (int i = 0; i < out.length; i++) {
                out[i] = (float) d[i];
            }
        }
        return out;
    }

    @Override
    public synchronized double[] downloadDouble(Storage src) {
        CudaStorage s = cs(src);
        ctx();
        int n = Math.toIntExact(s.size);
        double[] out = new double[n];
        switch (s.dtype) {
            case FLOAT64 -> cuda.download(MemorySegment.ofArray(out), s.address());
            case FLOAT32 -> {
                float[] f = new float[n];
                cuda.download(MemorySegment.ofArray(f), s.address());
                for (int i = 0; i < n; i++) {
                    out[i] = f[i];
                }
            }
            case FLOAT16, BFLOAT16 -> {
                short[] h = new short[n];
                cuda.download(MemorySegment.ofArray(h), s.address());
                for (int i = 0; i < n; i++) {
                    out[i] = s.dtype == DType.FLOAT16 ? Float.float16ToFloat(h[i]) : DType.bfloat16ToFloat(h[i]);
                }
            }
        }
        return out;
    }

    @Override
    public synchronized void convert(Storage in, Storage out) {
        CudaStorage x = cs(in);
        CudaStorage y = cs(out);
        CudaContext c = ctx();
        if (x.dtype == y.dtype) {
            c.copy(y.address(), x.address(), Math.min(x.bytes(), y.bytes()));
        } else if (x.dtype == DType.FLOAT32) {
            c.args().ptr(x.address()).ptr(y.address()).i64(y.size);
            c.launch(kernel("from_float", y.dtype), y.size);
        } else if (y.dtype == DType.FLOAT32) {
            c.args().ptr(x.address()).ptr(y.address()).i64(y.size);
            c.launch(kernel("to_float", x.dtype), y.size);
        } else {
            // through float32, which is also how the CPU rounds between these formats
            CudaStorage f = raw(x.size, DType.FLOAT32, null);
            convert(x, f);
            convert(f, y);
            f.release();
        }
    }

    @Override
    public synchronized void fill(Storage dst, float value) {
        CudaStorage s = cs(dst);
        CudaContext c = ctx();
        switch (s.dtype) {
            case FLOAT32 -> c.fill32(s.address(), Float.floatToRawIntBits(value), s.size);
            case FLOAT16 -> c.fill16(s.address(), Float.floatToFloat16(value), s.size);
            case BFLOAT16 -> c.fill16(s.address(), DType.floatToBfloat16(value), s.size);
            case FLOAT64 -> {
                if (Float.floatToRawIntBits(value) == 0) {
                    c.fill32(s.address(), 0, s.size * 2);
                } else {
                    c.args().ptr(s.address()).i64(s.size).f64(value);
                    c.launch(kernel("fill", s.dtype), s.size);
                }
            }
        }
    }

    @Override
    public synchronized Storage random(long size, long seed) {
        CudaStorage s = raw(size, DType.FLOAT32, null);
        cuda.args().ptr(s.address()).i64(size).i64(seed);
        cuda.launch("random_values", size);
        return s;
    }

    // ------------------------------------------------------------------ element-wise

    @Override
    public synchronized void unary(UnaryOp op, Storage in, Storage out) {
        CudaStorage y = cs(out);
        CudaContext c = ctx();
        CudaStorage x = as(cs(in), y.dtype);
        c.args().ptr(x.address()).ptr(y.address()).i64(y.size).i32(op.ordinal());
        c.launch(kernel("unary", y.dtype), y.size);
        done(x, cs(in));
    }

    @Override
    public synchronized void activation(ActivationOp op, float param, Storage in, Storage out) {
        CudaStorage y = cs(out);
        CudaContext c = ctx();
        CudaStorage x = as(cs(in), y.dtype);
        c.args().ptr(x.address()).ptr(y.address()).i64(y.size).i32(op.ordinal()).f32(param);
        c.launch(kernel("activation", y.dtype), y.size);
        done(x, cs(in));
    }

    @Override
    public synchronized void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        CudaStorage y = cs(out);
        CudaContext c = ctx();
        CudaStorage x = as(cs(in), y.dtype);
        CudaStorage g = as(cs(grad), y.dtype);
        c.args().ptr(x.address()).ptr(g.address()).ptr(y.address()).i64(y.size).i32(op.ordinal()).f32(param);
        c.launch(kernel("activation_backward", y.dtype), y.size);
        done(x, cs(in));
        done(g, cs(grad));
    }

    @Override
    public synchronized void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape,
            Storage out) {
        CudaStorage z = cs(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        CudaContext c = ctx();
        CudaStorage x = as(cs(a), z.dtype);
        CudaStorage y = as(cs(b), z.dtype);
        long[] contiguous = Axes.contiguous(shape);
        if (Arrays.equals(aStrides, contiguous) && Arrays.equals(bStrides, contiguous)) {
            c.args().ptr(x.address()).ptr(y.address()).ptr(z.address()).i64(n).i32(op.ordinal());
            c.launch(kernel("binary_contiguous", z.dtype), n);
        } else {
            Axes ax = Axes.merge(shape, aStrides, bStrides, contiguous);
            ax.forEachBlock((offsets, inner, strides, count) -> {
                c.args().ptr(x.address(offsets[0])).ptr(y.address(offsets[1])).ptr(z.address(offsets[2])).i64(count)
                        .i32(op.ordinal()).strided(inner.length, inner, strides[0], strides[1]);
                c.launch(kernel("binary", z.dtype), count);
            });
        }
        done(x, cs(a));
        done(y, cs(b));
    }

    @Override
    public void copy(Storage in, long[] strides, int[] shape, Storage out) {
        copy(in, 0, strides, shape, out, 0, Axes.contiguous(shape));
    }

    @Override
    public synchronized void copy(Storage in, long inOffset, long[] inStrides, int[] shape, Storage out,
            long outOffset, long[] outStrides) {
        CudaStorage z = cs(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        CudaContext c = ctx();
        CudaStorage x = as(cs(in), z.dtype);
        int bytes = z.dtype.bytes();
        Axes ax = Axes.merge(shape, inStrides, outStrides);
        if (ax.rank() == 1 && ax.strides[0][0] == 1 && ax.strides[1][0] == 1) {
            c.copy(z.address(outOffset), x.address(inOffset), n * bytes);
        } else {
            ax.forEachBlock((offsets, inner, strides, count) -> {
                c.args().ptr(x.address(inOffset + offsets[0])).ptr(z.address(outOffset + offsets[1])).i64(count)
                        .strided(inner.length, inner, strides[0], strides[1]);
                c.launch(kernel("copy", z.dtype), count);
            });
        }
        done(x, cs(in));
    }

    // ------------------------------------------------------------------ reductions

    @Override
    public synchronized void reduce(ReduceOp op, Storage in, long outer, long length, long inner, Storage out) {
        CudaStorage y = cs(out);
        long rows = outer * inner;
        if (rows == 0) {
            return;
        }
        CudaContext c = ctx();
        CudaStorage x = as(cs(in), y.dtype);
        if (op == ReduceOp.ARGMAX) {
            c.args().ptr(x.address()).ptr(y.address()).i64(rows).i64(length).i64(inner);
            c.launch(kernel("argmax", y.dtype), rows);
        } else {
            reduce(c, op.ordinal(), x, y, rows, length, inner);
        }
        done(x, cs(in));
    }

    private void reduce(CudaContext c, int op, CudaStorage x, CudaStorage y, long rows, long length, long inner) {
        long fill = 8L * c.multiprocessors * (CudaContext.GROUP / 32);   // warps that keep the device busy
        if (rows < fill / 8 && length > 1024) {
            // few long rows: a block each
            c.args().ptr(x.address()).ptr(y.address()).i64(rows).i64(length).i64(inner).i32(op);
            c.launch(kernel("reduce_block", y.dtype), (int) Math.min(rows, 1 << 20), 1, 1);
        } else if (inner == 1 && length >= 32) {
            // contiguous rows: a warp each
            c.args().ptr(x.address()).ptr(y.address()).i64(rows).i64(length).i32(op);
            c.launch(kernel("reduce_warp", y.dtype), (int) Math.min((rows + 7) / 8, 1 << 20), 1, 1);
        } else {
            c.args().ptr(x.address()).ptr(y.address()).i64(rows).i64(length).i64(inner).i32(op);
            c.launch(kernel("reduce", y.dtype), rows);
        }
    }

    // ------------------------------------------------------------------ matrix products

    @Override
    public void matmul(Storage a, Storage b, int m, int k, int n, Storage out) {
        matmulBatched(a, b, 1, m, k, n, out);
    }

    @Override
    public synchronized void matmulBatched(Storage a, Storage b, int batch, int m, int k, int n, Storage out) {
        CudaStorage z = cs(out);
        if ((long) batch * m * n == 0) {
            return;
        }
        CudaContext c = ctx();
        if (k == 0) {
            fill(z, 0);
            return;
        }
        CudaStorage x = as(cs(a), z.dtype);
        CudaStorage y = as(cs(b), z.dtype);
        DType t = z.dtype;
        boolean blas = c.cublas != null
                && (t == DType.FLOAT32 || t == DType.FLOAT64
                        || (t == DType.FLOAT16 && c.computeCapability >= 53)
                        || (t == DType.BFLOAT16 && c.computeCapability >= 80));
        if (blas) {
            c.cublas.gemm(t, false, false, m, n, k, x.address(), k, (long) m * k, y.address(), n, (long) k * n,
                    z.address(), (long) m * n, batch);
            c.products++;
            if (CudaContext.DEBUG) {
                c.debugWait("cuBLAS " + t + " product " + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n + "]");
            }
        } else if (c.cublas != null && (t == DType.FLOAT16 || t == DType.BFLOAT16)) {
            // 16-bit products the device's cuBLAS cannot do: in float32
            CudaStorage xf = as(x, DType.FLOAT32);
            CudaStorage yf = as(y, DType.FLOAT32);
            CudaStorage zf = raw(z.size, DType.FLOAT32, null);
            matmulBatched(xf, yf, batch, m, k, n, zf);
            convert(zf, z);
            xf.release();
            yf.release();
            zf.release();
        } else {
            int tiles = 16;
            for (int done = 0; done < batch; done += 65535) {
                int part = Math.min(65535, batch - done);
                c.args().ptr(x.address((long) done * m * k)).ptr(y.address((long) done * k * n))
                        .ptr(z.address((long) done * m * n)).i32(m).i32(k).i32(n).i64((long) m * k).i64((long) k * n)
                        .i64((long) m * n);
                c.launch(kernel("matmul", t), (n + tiles - 1) / tiles, (m + tiles - 1) / tiles, part);
            }
        }
        done(x, cs(a));
        done(y, cs(b));
    }

    // ------------------------------------------------------------------ RMSNorm

    @Override
    public boolean hasRmsNorm(DType type) {
        return type != DType.FLOAT64 && isAvailable();
    }

    @Override
    public synchronized void rmsNorm(Storage x, Storage weight, long rows, int cols, float eps, Storage out) {
        CudaStorage y = cs(out);
        CudaContext c = ctx();
        CudaStorage xs = as(cs(x), y.dtype);
        CudaStorage ws = as(cs(weight), y.dtype);
        c.args().ptr(xs.address()).ptr(ws.address()).ptr(y.address()).i64(rows).i32(cols).f32(eps);
        c.launch(kernel("rms_norm", y.dtype), (int) Math.min(rows, 1 << 20), 1, 1);
        done(xs, cs(x));
        done(ws, cs(weight));
    }

    // ------------------------------------------------------------------ windows and indices

    @Override
    public synchronized void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        CudaStorage z = cs(out);
        CudaContext c = ctx();
        CudaStorage x = as(cs(in), z.dtype);
        Tables t = tables(geometry);
        int ks = geometry.kernelSize();
        long total = (long) geometry.batch() * geometry.positionCount() * geometry.channels() * ks;
        c.args().ptr(x.address()).ptr(t.im2col.address()).ptr(z.address()).i32(geometry.channels())
                .i32(geometry.positionCount()).i32(ks).i64(geometry.imageSize()).f32(padValue).i64(total);
        c.launch(kernel("im2col", z.dtype), total);
        done(x, cs(in));
    }

    @Override
    public synchronized void col2im(Storage cols, ConvGeometry geometry, Storage out) {
        CudaStorage z = cs(out);
        CudaContext c = ctx();
        CudaStorage x = as(cs(cols), z.dtype);
        Tables t = tables(geometry);
        long total = (long) geometry.batch() * geometry.channels() * geometry.imageSize();
        c.args().ptr(x.address()).ptr(t.offsets.address()).ptr(t.entries.address()).ptr(z.address())
                .i32(geometry.channels()).i32(geometry.positionCount()).i32(geometry.kernelSize())
                .i64(geometry.imageSize()).i64(total);
        c.launch(kernel("col2im", z.dtype), total);
        done(x, cs(cols));
    }

    private Tables tables(ConvGeometry g) {
        String key = g.channels() + "|" + Arrays.toString(g.image()) + Arrays.toString(g.kernel())
                + Arrays.toString(g.stride()) + Arrays.toString(g.padBefore()) + Arrays.toString(g.dilation())
                + Arrays.toString(g.positions());
        Tables t = tables.get(key);
        if (t != null) {
            return t;
        }
        int[] table = table(g);
        int img = g.imageSize();
        int[] offsets = new int[img + 1];
        for (int cell : table) {
            if (cell >= 0) {
                offsets[cell + 1]++;
            }
        }
        for (int c = 0; c < img; c++) {
            offsets[c + 1] += offsets[c];
        }
        int[] entries = new int[offsets[img]];
        int[] next = Arrays.copyOf(offsets, img);
        for (int e = 0; e < table.length; e++) {
            if (table[e] >= 0) {
                entries[next[table[e]]++] = e;
            }
        }
        // evicted tables go back to the pool when collected; only later work can reuse them
        t = new Tables(uploadInts(table), uploadInts(offsets), uploadInts(entries));
        tables.put(key, t);
        return t;
    }

    /** {@code table[p * kernelSize + k]}: the image cell kernel cell {@code k} reads at position {@code p}, or -1. */
    static int[] table(ConvGeometry g) {
        int[] image = g.image();
        int[] kernel = g.kernel();
        int[] pos = g.positions();
        int rank = image.length;
        int ks = g.kernelSize();
        int positions = g.positionCount();
        int[] t = new int[positions * ks];
        int[] origin = new int[rank];
        for (int p = 0; p < positions; p++) {
            int rest = p;
            for (int d = rank - 1; d >= 0; d--) {
                origin[d] = (rest % pos[d]) * g.stride()[d] - g.padBefore()[d];
                rest /= pos[d];
            }
            for (int k = 0; k < ks; k++) {
                int kr = k;
                int cell = 0;
                int step = 1;
                boolean inside = true;
                for (int d = rank - 1; d >= 0; d--) {
                    int c = origin[d] + (kr % kernel[d]) * g.dilation()[d];
                    kr /= kernel[d];
                    if (c < 0 || c >= image[d]) {
                        inside = false;
                    }
                    cell += c * step;
                    step *= image[d];
                }
                t[p * ks + k] = inside ? cell : -1;
            }
        }
        return t;
    }

    /** The indices as whole numbers, checked; read from the upload when it was kept. */
    private int[] indices(CudaStorage storage, int count, long length) {
        double[] values;
        if (storage.host != null) {
            values = new double[count];
            for (int i = 0; i < count; i++) {
                values[i] = storage.host[i];
            }
        } else {
            values = downloadDouble(storage);
        }
        int[] idx = new int[count];
        for (int i = 0; i < count; i++) {
            double v = values[i];
            int j = (int) v;
            if (j != v || j < 0 || j >= length) {
                throw new IndexOutOfBoundsException("index " + v + " out of range [0, " + length + ")");
            }
            idx[i] = j;
        }
        return idx;
    }

    @Override
    public synchronized void gather(Storage in, long outer, long length, long inner, Storage indices, int count,
            Storage out) {
        CudaStorage z = cs(out);
        CudaContext c = ctx();
        indices(cs(indices), count, length);
        CudaStorage x = as(cs(in), z.dtype);
        CudaStorage ind = as(cs(indices), DType.FLOAT32);
        long total = outer * count * inner;
        c.args().ptr(x.address()).ptr(ind.address()).ptr(z.address()).i64(length).i64(inner).i64(count).i64(total);
        c.launch(kernel("gather", z.dtype), total);
        done(x, cs(in));
        done(ind, cs(indices));
    }

    @Override
    public synchronized void scatterAdd(Storage src, long outer, int count, long inner, Storage indices, long length,
            Storage out) {
        CudaStorage z = cs(out);
        CudaContext c = ctx();
        int[] idx = indices(cs(indices), count, length);
        CudaStorage s = as(cs(src), z.dtype);
        int len = Math.toIntExact(length);
        int[] offsets = new int[len + 1];
        for (int i : idx) {
            offsets[i + 1]++;
        }
        for (int l = 0; l < len; l++) {
            offsets[l + 1] += offsets[l];
        }
        int[] entries = new int[count];
        int[] next = Arrays.copyOf(offsets, len);
        for (int i = 0; i < count; i++) {
            entries[next[idx[i]]++] = i;
        }
        CudaStorage off = uploadInts(offsets);
        CudaStorage ent = uploadInts(entries);
        long total = outer * length * inner;
        c.args().ptr(s.address()).ptr(off.address()).ptr(ent.address()).ptr(z.address()).i64(length).i64(inner)
                .i64(count).i64(total);
        c.launch(kernel("scatter_add", z.dtype), total);
        done(s, cs(src));
        off.release();
        ent.release();
    }
}
