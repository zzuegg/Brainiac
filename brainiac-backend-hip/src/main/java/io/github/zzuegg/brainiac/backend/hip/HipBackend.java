package io.github.zzuegg.brainiac.backend.hip;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.PackedStorage;
import io.github.zzuegg.brainiac.tensor.spi.QuantizedRows;
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
 * A GPU backend for AMD GPUs on HIP (ROCm), called through the Foreign Function and Memory API:
 * no JNI and no libraries beyond AMD's own. It is a port of the CUDA backend, whose API HIP
 * mirrors. It needs
 * <ul>
 * <li>the HIP runtime ({@code amdhip64_7.dll}, which comes with AMD's display driver and the HIP
 * SDK, or {@code libamdhip64.so} from ROCm),</li>
 * <li>hipRTC, HIP's runtime compiler, which compiles the kernels for the GPU at first use (the
 * result is cached in {@code ~/.cache/brainiac/hip}),</li>
 * <li>optionally rocBLAS, for the matrix products; without it a simpler kernel of its own runs.</li>
 * </ul>
 * hipRTC and rocBLAS come with AMD's HIP SDK for Windows or ROCm on Linux; the libraries are
 * found through {@code HIP_PATH}, the usual install folders or the {@code PATH}. The kernels are
 * written for 32-lane wavefronts, so the backend runs on RDNA GPUs (Radeon RX 5000 and later).
 * Without a runtime or a device the backend is unavailable and Brainiac uses the other
 * devices; {@link #status()} says why.
 *
 * <p>Stores float32, float64, float16 and bfloat16 tensors; float16 and bfloat16 are computed
 * in float32 (their products in rocBLAS with float32 accumulation), float64 in double precision.
 * Every operation is queued on one stream, so the host only waits when it reads values back.
 *
 * <p>Run with {@code --enable-native-access=io.github.zzuegg.brainiac.backend.hip} (or
 * {@code ALL-UNNAMED} on the class path). System properties:
 * <ul>
 * <li>{@code brainiac.hip.device}: the device index, default the GPU with the most compute units
 * (a graphics card before the processor's integrated GPU)</li>
 * <li>{@code brainiac.hip.memory}: the memory budget in MB</li>
 * <li>{@code brainiac.hip.debug=true}: wait after every operation, so errors name the
 * operation that caused them</li>
 * <li>{@code brainiac.hip.critical=true}: link the per-operation runtime and rocBLAS calls as
 * critical downcalls, which makes each call cheaper</li>
 * <li>{@code brainiac.hip.runtime}, {@code brainiac.hip.hiprtc}, {@code brainiac.hip.rocblas}:
 * the library files ({@code brainiac.hip.rocblas=none}: no rocBLAS)</li>
 * <li>{@code brainiac.hip.arch}: the GPU's compiler target, such as {@code gfx1201}, when the
 * runtime's properties do not tell</li>
 * <li>{@code brainiac.hip.cache}: where compiled kernels are kept, {@code none} for nowhere</li>
 * <li>{@code brainiac.hip.statistics=true}: print {@link #statistics()} at exit</li>
 * <li>{@code brainiac.hip.devices}: {@code one} for only one device; {@code all} for one per GPU
 * ({@code hip}, {@code hip:1}, ...); by default one per GPU of the first one's architecture</li>
 * <li>{@code brainiac.hip.fusedAttention=false}: attention from separate operations</li>
 * <li>{@code brainiac.hip.fewRows=false}: products of up to 16 rows in float16 and bfloat16 by
 * rocBLAS instead of a kernel of its own</li>
 * </ul>
 *
 * <p>Attention runs fused (forward, backward, and with a bias) for heads up to 256; packed
 * weights stay packed on the device, read directly by products of up to 16 rows; Fourier
 * transforms of power-of-two lengths up to 2048 run on the device. Gradients of several GPUs
 * are summed on the host (there is no RCCL on Windows).
 */
public final class HipBackend implements Backend {

    private static final Cleaner CLEANER = Cleaner.create();

    private HipContext hip;
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
    private record Tables(HipStorage im2col, HipStorage offsets, HipStorage entries) {
    }

    private static HipBackend first;

    /**
     * The first instance that found a device, normally the one behind the {@code hip}
     * {@link Device}; for {@link #statistics()}.
     *
     * @return the instance, if any
     */
    public static synchronized Optional<HipBackend> loaded() {
        return Optional.ofNullable(first);
    }

    /** The device index, or -1 for the one {@code brainiac.hip.device} names (default the largest GPU). */
    private final int requested;
    private final boolean primary;

    /** Creates the backend; called by {@link java.util.ServiceLoader}. Does not touch the driver. */
    public HipBackend() {
        this(-1, true);
    }

    private HipBackend(int ordinal, boolean primary) {
        this.requested = ordinal;
        this.primary = primary;
    }

    /**
     * The first device keeps the name {@code hip (...)}; the others, found by {@link #siblings()},
     * are {@code hip:N (...)}.
     */
    @Override
    public String name() {
        if (hip == null) {
            return primary ? "hip" : "hip:" + requested;
        }
        return (primary ? "hip" : "hip:" + hip.ordinal) + " (" + hip.deviceName + ")";
    }

    /**
     * The machine's other AMD GPUs of the same architecture as the first, each a backend of its
     * own. Others, such as the processor's integrated GPU next to a graphics card, are left alone:
     * opening one made the process hang on Windows. {@code -Dbrainiac.hip.devices=one}: none;
     * {@code =all}: every GPU.
     */
    @Override
    public synchronized java.util.List<Backend> siblings() {
        String which = System.getProperty("brainiac.hip.devices", "same");
        if (!primary || !isAvailable() || "one".equals(which)) {
            return java.util.List.of();
        }
        java.util.List<Backend> others = new java.util.ArrayList<>();
        int count = hip.cu.deviceCount();
        for (int i = 0; i < count; i++) {
            if (i != hip.ordinal && ("all".equals(which) || sameArchitecture(i))) {
                others.add(new HipBackend(i, false));
            }
        }
        return others;
    }

    private boolean sameArchitecture(int ordinal) {
        try {
            int device = hip.cu.device(ordinal);
            String arch = hip.cu.properties(device, hip.cu.totalMemory(device)).architecture();
            return arch.split(":")[0].equals(hip.architecture.split(":")[0]);
        } catch (HipException e) {
            return false;
        }
    }

    /**
     * The device's index among the driver's devices.
     *
     * @return the index
     */
    public int ordinal() {
        return isAvailable() ? hip.ordinal : requested;
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
                hip = new HipContext(requested >= 0 ? requested : Integer.getInteger("brainiac.hip.device", HipContext.largestDevice()));
                initNanos = System.nanoTime() - start;
                synchronized (HipBackend.class) {
                    if (first == null && primary) {
                        first = this;
                    }
                }
                if (!"false".equals(System.getProperty("brainiac.hip.closeAtExit"))) {
                    Runtime.getRuntime().addShutdownHook(new Thread(this::closeAtExit, "hip-close"));
                }
                if (Boolean.getBoolean("brainiac.hip.statistics")) {
                    Runtime.getRuntime().addShutdownHook(new Thread(() -> System.err.println(name() + ": "
                            + statistics()), "hip-statistics"));
                }
            } catch (RuntimeException | LinkageError e) {
                unavailable = e.getMessage() == null ? e.toString() : e.getMessage();
                hip = null;
                if (HipContext.DEBUG) {
                    e.printStackTrace();
                }
            }
        }
        return hip != null;
    }

    /** Releases the device at exit ({@code -Dbrainiac.hip.closeAtExit=false}: leave it to the runtime). */
    private synchronized void closeAtExit() {
        if (hip != null) {
            try {
                hip.close();
            } catch (RuntimeException e) {
                System.err.println("brainiac hip: closing " + name() + ": " + e.getMessage());
            }
            hip = null;
            unavailable = "closed at exit";
        }
    }

    /**
     * Describes the device and libraries in use, or why the backend is unavailable.
     *
     * @return for example {@code "device 0: AMD Radeon RX 9070 XT, gfx1201, 64 CUs, ..."}
     */
    public synchronized String status() {
        if (!isAvailable()) {
            return "unavailable: " + unavailable;
        }
        HipContext c = hip;
        return "device " + c.ordinal + ": " + c.deviceName + ", " + c.architecture + ", " + c.multiprocessors
                + " CUs, " + (c.deviceMemory >> 20) + " MB, " + (c.budget >> 20) + " MB budget; HIP runtime "
                + HipRuntime.versionText(c.cu.runtimeVersion()) + ", driver " + c.cu.driverVersion() + " (" + c.cu.library
                + "); hipRTC "
                + c.hiprtc.version() + " (" + c.hiprtc.library + ", "
                + (c.cachedKernels ? "cached" : "compiled in " + c.compileNanos / 1_000_000 + " ms") + "); rocBLAS "
                + (c.rocblas != null ? c.rocblas.version() + " (" + c.rocblas.library + ")"
                        : "unavailable: " + c.rocblasUnavailable)
                + (HipContext.DEBUG ? "; debug" : "") + (Binder.CRITICAL ? "; critical downcalls" : "");
    }

    /**
     * Counts of work done so far, for tuning.
     *
     * @return launches, products, waits, copies and memory use
     */
    public synchronized String statistics() {
        if (hip == null) {
            return "not initialized";
        }
        HipContext c = hip;
        return "init " + initNanos / 1_000_000 + " ms, " + c.launches + " kernel launches, " + c.products
                + " rocBLAS products, " + c.waits + " waits, " + (c.uploadedBytes >> 20) + " MB uploaded, "
                + (c.downloadedBytes >> 20) + " MB downloaded, " + (c.inUse() >> 20) + " MB in use, "
                + (c.peak() >> 20) + " MB peak, " + (c.reserved() >> 20) + " MB reserved, " + c.collections
                + " collections";
    }

    /** Waits until the device has run everything queued, for timing. */
    public synchronized void finish() {
        if (hip != null) {
            hip.bind();
            hip.finish();
        }
    }

    @Override
    public synchronized Device.Memory memory() {
        if (hip == null) {
            return null;
        }
        return new Device.Memory(hip.inUse(), hip.peak(), hip.reserved(), hip.budget);
    }

    @Override
    public synchronized void resetPeakMemory() {
        if (hip != null) {
            hip.resetPeak();
        }
    }

    // ------------------------------------------------------------------ storage

    /** The context, current on this thread. */
    private HipContext ctx() {
        if (!isAvailable()) {
            throw new IllegalStateException("the hip backend is unavailable: " + unavailable);
        }
        hip.bind();
        return hip;
    }

    private HipStorage raw(long size, DType dtype, float[] host) {
        HipContext c = ctx();
        return new HipStorage(this, c, size, dtype, host, CLEANER);
    }

    private HipStorage cs(Storage s) {
        if (s instanceof HipStorage c && c.owner == this) {
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
            default -> throw new IllegalArgumentException("no HIP kernels for " + dtype);
        };
    }

    /** {@code s} in another format: itself if it already is, else a converted copy to {@link #done}. */
    private HipStorage as(HipStorage s, DType dtype) {
        if (s.dtype == dtype) {
            return s;
        }
        HipStorage c = raw(s.size, dtype, null);
        convert(s, c);
        return c;
    }

    /** Releases conversions made by {@link #as}: those that differ from the originals. */
    private static void done(HipStorage converted, HipStorage original) {
        if (converted != original) {
            converted.release();
        }
    }

    @Override
    public boolean supports(DType dtype) {
        return dtype.floating() && isAvailable();   // integer tensors live on the CPU
    }

    @Override
    public synchronized Storage allocate(long size) {
        return allocate(size, DType.FLOAT32);
    }

    @Override
    public synchronized Storage allocate(long size, DType dtype) {
        HipStorage s = raw(size, dtype, null);
        long bytes = s.bytes();
        if (bytes % 4 == 0) {
            hip.fill32(s.address(), 0, bytes / 4);
        } else {
            hip.fill8(s.address(), (byte) 0, bytes);
        }
        return s;
    }

    @Override
    public synchronized Storage allocateUninitialized(long size, DType dtype) {
        return raw(size, dtype, null);
    }

    @Override
    public synchronized void release(Storage storage) {
        if (storage instanceof Packed) {
            return;   // weights: freed with the tensor that holds them, by the collector
        }
        if (storage instanceof HipStorage s && s.owner == this) {
            s.release();
        }
    }

    @Override
    public synchronized Storage upload(float[] data) {
        HipStorage s = raw(data.length, DType.FLOAT32, data.length <= 4096 ? data.clone() : null);
        hip.upload(s.address(), MemorySegment.ofArray(data));
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
                HipStorage s = raw(data.length, dtype, null);
                hip.upload(s.address(), MemorySegment.ofArray(data));
                yield s;
            }
            case FLOAT16, BFLOAT16 -> {
                short[] h = new short[data.length];
                for (int i = 0; i < h.length; i++) {
                    float v = (float) data[i];
                    h[i] = dtype == DType.FLOAT16 ? Float.floatToFloat16(v) : DType.floatToBfloat16(v);
                }
                HipStorage s = raw(data.length, dtype, null);
                hip.upload(s.address(), MemorySegment.ofArray(h));
                yield s;
            }
            default -> throw new UnsupportedOperationException("HIP holds no " + dtype + " tensors");
        };
    }

    @Override
    public synchronized Storage upload(MemorySegment data, DType dtype) {
        if (!RawValues.nativeLittleEndian()) {
            return Backend.super.upload(data, dtype);
        }
        HipStorage s = raw(data.byteSize() / dtype.bytes(), dtype, null);
        hip.upload(s.address(), data.asSlice(0, s.bytes()));
        return s;
    }

    private HipStorage uploadInts(int[] data) {
        HipStorage s = raw(Math.max(1, data.length), DType.FLOAT32, null);
        hip.upload(s.address(), MemorySegment.ofArray(data));
        return s;
    }

    @Override
    public synchronized float[] download(Storage src) {
        HipStorage s = cs(src);
        ctx();
        float[] out = new float[Math.toIntExact(s.size)];
        if (s.dtype == DType.FLOAT32) {
            hip.download(MemorySegment.ofArray(out), s.address());
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
        HipStorage s = cs(src);
        ctx();
        int n = Math.toIntExact(s.size);
        double[] out = new double[n];
        switch (s.dtype) {
            case FLOAT64 -> hip.download(MemorySegment.ofArray(out), s.address());
            case FLOAT32 -> {
                float[] f = new float[n];
                hip.download(MemorySegment.ofArray(f), s.address());
                for (int i = 0; i < n; i++) {
                    out[i] = f[i];
                }
            }
            case FLOAT16, BFLOAT16 -> {
                short[] h = new short[n];
                hip.download(MemorySegment.ofArray(h), s.address());
                for (int i = 0; i < n; i++) {
                    out[i] = s.dtype == DType.FLOAT16 ? Float.float16ToFloat(h[i]) : DType.bfloat16ToFloat(h[i]);
                }
            }
        }
        return out;
    }

    @Override
    public synchronized void convert(Storage in, Storage out) {
        HipStorage x = cs(in);
        HipStorage y = cs(out);
        HipContext c = ctx();
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
            HipStorage f = raw(x.size, DType.FLOAT32, null);
            convert(x, f);
            convert(f, y);
            f.release();
        }
    }

    @Override
    public synchronized void fill(Storage dst, float value) {
        HipStorage s = cs(dst);
        HipContext c = ctx();
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
        HipStorage s = raw(size, DType.FLOAT32, null);
        hip.args().ptr(s.address()).i64(size).i64(seed);
        hip.launch("random_values", size);
        return s;
    }

    // ------------------------------------------------------------------ element-wise

    @Override
    public synchronized void unary(UnaryOp op, Storage in, Storage out) {
        HipStorage y = cs(out);
        HipContext c = ctx();
        HipStorage x = as(cs(in), y.dtype);
        c.args().ptr(x.address()).ptr(y.address()).i64(y.size).i32(op.ordinal());
        c.launch(kernel("unary", y.dtype), y.size);
        done(x, cs(in));
    }

    @Override
    public synchronized void activation(ActivationOp op, float param, Storage in, Storage out) {
        HipStorage y = cs(out);
        HipContext c = ctx();
        HipStorage x = as(cs(in), y.dtype);
        c.args().ptr(x.address()).ptr(y.address()).i64(y.size).i32(op.ordinal()).f32(param);
        c.launch(kernel("activation", y.dtype), y.size);
        done(x, cs(in));
    }

    @Override
    public synchronized void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        HipStorage y = cs(out);
        HipContext c = ctx();
        HipStorage x = as(cs(in), y.dtype);
        HipStorage g = as(cs(grad), y.dtype);
        c.args().ptr(x.address()).ptr(g.address()).ptr(y.address()).i64(y.size).i32(op.ordinal()).f32(param);
        c.launch(kernel("activation_backward", y.dtype), y.size);
        done(x, cs(in));
        done(g, cs(grad));
    }

    @Override
    public synchronized void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape,
            Storage out) {
        HipStorage z = cs(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        HipContext c = ctx();
        HipStorage x = as(cs(a), z.dtype);
        HipStorage y = as(cs(b), z.dtype);
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
        HipStorage z = cs(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        HipContext c = ctx();
        HipStorage x = as(cs(in), z.dtype);
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
        HipStorage y = cs(out);
        long rows = outer * inner;
        if (rows == 0) {
            return;
        }
        HipContext c = ctx();
        HipStorage x = as(cs(in), y.dtype);
        if (op == ReduceOp.ARGMAX) {
            c.args().ptr(x.address()).ptr(y.address()).i64(rows).i64(length).i64(inner);
            c.launch(kernel("argmax", y.dtype), rows);
        } else {
            reduce(c, op.ordinal(), x, y, rows, length, inner);
        }
        done(x, cs(in));
    }

    private void reduce(HipContext c, int op, HipStorage x, HipStorage y, long rows, long length, long inner) {
        long fill = 8L * c.multiprocessors * (HipContext.GROUP / 32);   // warps that keep the device busy
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
        HipStorage z = cs(out);
        if ((long) batch * m * n == 0) {
            return;
        }
        HipContext c = ctx();
        if (k == 0) {
            fill(z, 0);
            return;
        }
        HipStorage x = as(cs(a), z.dtype);
        HipStorage y = as(cs(b), z.dtype);
        DType t = z.dtype;
        boolean blas = c.rocblas != null;
        if (FEW_ROWS && batch == 1 && m >= 2 && m <= FEW && (t == DType.FLOAT16 || t == DType.BFLOAT16)) {
            // a handful of rows: a BLAS library runs these 16-bit products slower than float32 ones
            few(c, "few_dense", x, y.address(), 0, 0, 0, 0, m, k, n, z);
        } else if (blas) {
            HipContext.trace("rocBLAS " + t + " " + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n + "]");
            c.rocblas.gemm(t, false, false, m, n, k, x.address(), k, (long) m * k, y.address(), n, (long) k * n,
                    z.address(), (long) m * n, batch);
            c.products++;
            if (HipContext.DEBUG) {
                c.debugWait("rocBLAS " + t + " product " + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n + "]");
            }
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

    /** Whether products of a few 16-bit rows use a kernel of their own ({@code -Dbrainiac.hip.fewRows=false}: rocBLAS). */
    private static final boolean FEW_ROWS = !"false".equals(System.getProperty("brainiac.hip.fewRows"));
    /** The most rows {@code few_dense} and {@code few_packed} take (FEW_ROWS in kernels.hip). */
    static final int FEW = 16;
    /** The columns a block of the few-row kernels covers: a thread each. */
    private static final int FEW_COLUMNS = HipContext.GROUP;

    /**
     * {@code out[m, n] = a[m, k] x w[k, n]} for at most {@link #FEW} rows: a thread per column of
     * {@code w}, the {@code k} rows split into pieces until the device is busy, the pieces' float
     * sums then added by {@code few_sum}. {@code w} is dense ({@code few_dense}: its address) or
     * packed ({@code few_packed}: levels, scales, offsets, group, nibbles).
     */
    private void few(HipContext c, String kernel, HipStorage a, long w, long scales, long offsets, int group,
            int nibbles, int m, int k, int n, HipStorage out) {
        int columnBlocks = (n + FEW_COLUMNS - 1) / FEW_COLUMNS;
        int target = 4 * c.multiprocessors;
        int pieces = Math.max(1, Math.min((k + 127) / 128, target / columnBlocks));
        int chunk = ((k + pieces - 1) / pieces + 63) / 64 * 64;
        pieces = (k + chunk - 1) / chunk;
        HipStorage part = pieces > 1 ? raw((long) pieces * m * n, DType.FLOAT32, null) : null;
        long partAddress = part != null ? part.address() : 0;
        c.args().ptr(a.address()).ptr(w);
        if (kernel.equals("few_packed")) {
            c.ptr(scales).ptr(offsets).i32(group).i32(nibbles);
        }
        c.ptr(partAddress).ptr(out.address()).i32(m).i32(k).i32(n).i32(chunk);
        c.launch(kernel(kernel, out.dtype), columnBlocks, pieces, 1);
        if (part != null) {
            long mn = (long) m * n;
            c.args().ptr(part.address()).ptr(out.address()).i64(mn).i32(pieces);
            c.launch(kernel("few_sum", out.dtype), mn);
            part.release();
        }
    }

    // ------------------------------------------------------------------ packed (quantized) weights

    /**
     * A packed matrix on the device: the levels as PackedMatrix lays them out (a signed byte
     * each, or two per byte), the scales and offsets as float32 per group.
     */
    private record Packed(int rows, int cols, PackedMatrix host, HipStorage levels, HipStorage scales,
            HipStorage offsets) implements PackedStorage {

        @Override
        public long size() {
            return (long) rows * cols;
        }

        @Override
        public long bytes() {
            return levels.bytes() + scales.bytes() + (offsets == null ? 0 : offsets.bytes());
        }

        int group() {
            return host.group();
        }

        int nibbles() {
            return host.nibbles() ? 1 : 0;
        }
    }

    private HipStorage uploadBytes(byte[] data) {
        HipStorage s = raw(Math.max(1, (data.length + 3) / 4), DType.FLOAT32, null);
        hip.upload(s.address(), MemorySegment.ofArray(data));
        return s;
    }

    @Override
    public synchronized PackedStorage pack(PackedMatrix matrix) {
        ctx();
        HipStorage levels = uploadBytes(matrix.data());
        HipStorage scales = (HipStorage) upload(matrix.scales());
        HipStorage offsets = matrix.offsets() == null ? null : (HipStorage) upload(matrix.offsets());
        return new Packed(matrix.rows(), matrix.cols(), matrix, levels, scales, offsets);
    }

    @Override
    public PackedMatrix unpackToHost(PackedStorage packed) {
        if (packed instanceof Packed p) {
            return p.host();
        }
        return Backend.super.unpackToHost(packed);
    }

    /** {@code s} as this backend's packed form, moved here when it was packed for another backend. */
    private Packed packed(PackedStorage s) {
        if (s instanceof Packed p && p.levels().owner == this) {
            return p;
        }
        return (Packed) pack(unpackToHost(s));
    }

    @Override
    public synchronized void unpack(PackedStorage packed, Storage out) {
        if (packed instanceof QuantizedRows) {
            Backend.super.unpack(packed, out);
            return;
        }
        Packed p = packed(packed);
        HipStorage z = cs(out);
        HipContext c = ctx();
        c.args().ptr(p.levels().address()).ptr(p.scales().address())
                .ptr(p.offsets() == null ? 0 : p.offsets().address()).i32(p.group()).i32(p.nibbles())
                .ptr(z.address()).i32(p.rows()).i32(p.cols());
        c.launch(kernel("unpack_packed", z.dtype), p.size());
    }

    @Override
    public synchronized void matmulPacked(Storage a, PackedStorage b, int m, int k, int n, Storage out) {
        if (b instanceof QuantizedRows || m == 0 || k == 0) {
            Backend.super.matmulPacked(a, b, m, k, n, out);
            return;
        }
        HipStorage z = cs(out);
        HipContext c = ctx();
        if (m > FEW) {
            // many rows: unpacked into a temporary for rocBLAS, which is then free for the next
            HipStorage w = raw((long) k * n, z.dtype, null);
            unpack(b, w);
            matmulBatched(a, w, 1, m, k, n, z);
            w.release();
            return;
        }
        Packed p = packed(b);
        HipStorage x = as(cs(a), z.dtype);
        few(c, "few_packed", x, p.levels().address(), p.scales().address(),
                p.offsets() == null ? 0 : p.offsets().address(), p.group(), p.nibbles(), m, k, n, z);
        done(x, cs(a));
    }

    // ------------------------------------------------------------------ attention

    /** Whether attention runs fused ({@code -Dbrainiac.hip.fusedAttention=false}: never). */
    private static final boolean FUSED_ATTENTION = !"false".equals(System.getProperty("brainiac.hip.fusedAttention"));
    /** The largest head the attention kernels take: 32 lanes times ATT_LANES in kernels.hip. */
    static final int ATTENTION_HEAD = 256;

    @Override
    public boolean hasAttention(int[] shape, DType type, boolean backward) {
        return FUSED_ATTENTION && type != DType.FLOAT64 && shape[5] <= ATTENTION_HEAD && shape[6] <= ATTENTION_HEAD
                && shape[2] > 0 && shape[1] % shape[2] == 0 && isAvailable();
    }

    @Override
    public boolean hasAttentionWithBias(int[] shape, DType type) {
        return hasAttention(shape, type, false);
    }

    @Override
    public void attention(Storage q, Storage k, Storage v, int[] shape, float scale, boolean causal, Storage out,
            Storage lse) {
        attention(q, k, v, shape, scale, causal, null, new long[4], out, lse);
    }

    /** {@code batch, heads, kvHeads, tq, tk, dk, dv} and the scale as the kernels' AttShape struct. */
    private static int[] attShape(int[] shape, float scale, boolean causal) {
        return new int[] {shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], Float.floatToRawIntBits(scale),
            causal ? 1 : 0};
    }

    private void checkAttention(int[] shape, DType type) {
        if (type == DType.FLOAT64 || shape[5] > ATTENTION_HEAD || shape[6] > ATTENTION_HEAD) {
            throw new UnsupportedOperationException("fused attention on " + name() + ": " + type + ", heads of "
                    + shape[5] + " and " + shape[6]);
        }
    }

    /** Launches a kernel that runs a warp per item, with as many blocks as that takes. */
    private static void warps(HipContext c, String name, long count) {
        long blocks = (count * 32 + HipContext.GROUP - 1) / HipContext.GROUP;
        c.launch(name, (int) Math.max(1, blocks), 1, 1);
    }

    @Override
    public synchronized void attention(Storage q, Storage k, Storage v, int[] shape, float scale, boolean causal,
            Storage bias, long[] strides, Storage out, Storage lse) {
        HipStorage z = cs(out);
        DType t = z.dtype;
        checkAttention(shape, t);
        HipStorage l = cs(lse);
        long rows = (long) shape[0] * shape[1] * shape[3];
        if (rows == 0) {
            return;
        }
        HipContext c = ctx();
        HipStorage xq = as(cs(q), t);
        HipStorage xk = as(cs(k), t);
        HipStorage xv = as(cs(v), t);
        HipStorage xb = bias == null ? null : as(cs(bias), t);
        int tk = shape[4];
        // split each row's keys into pieces when the rows alone leave the device idle
        long target = 64L * c.multiprocessors;
        int parts = (int) Math.max(1, Math.min((tk + 63) / 64, (target + rows - 1) / rows));
        int chunk = Math.max(1, (tk + parts - 1) / parts);
        parts = Math.max(1, (tk + chunk - 1) / chunk);
        HipStorage part = parts > 1 ? raw(rows * parts * (shape[6] + 2), DType.FLOAT32, null) : null;
        c.args().ptr(xq.address()).ptr(xk.address()).ptr(xv.address()).ptr(z.address()).ptr(l.address())
                .ptr(xb == null ? 0 : xb.address()).ptr(part == null ? 0 : part.address())
                .ints(attShape(shape, scale, causal)).i64(rows).i32(xb == null ? 0 : 1).i64(strides[0])
                .i64(strides[1]).i64(strides[2]).i64(strides[3]).i32(parts).i32(chunk);
        warps(c, kernel("attention", t), rows * parts);
        if (part != null) {
            c.args().ptr(part.address()).ptr(z.address()).ptr(l.address()).i64(rows).i32(parts).i32(shape[6]);
            warps(c, kernel("attention_combine", t), rows);
            part.release();
        }
        done(xq, cs(q));
        done(xk, cs(k));
        done(xv, cs(v));
        if (xb != null) {
            done(xb, cs(bias));
        }
    }

    /** The largest latent the latent attention kernel takes: 32 lanes times LAT_LANES in kernels.hip. */
    static final int LATENT = 512;

    @Override
    public boolean hasLatentAttention(int[] shape, DType type) {
        return FUSED_ATTENTION && type != DType.FLOAT64 && shape[4] <= LATENT && shape[5] <= ATTENTION_HEAD
                && isAvailable();
    }

    @Override
    public synchronized void latentAttention(Storage qc, Storage qr, Storage c, Storage r, int[] shape, float scale,
            Storage bias, long[] strides, Storage out) {
        HipStorage z = cs(out);
        DType t = z.dtype;
        if (t == DType.FLOAT64 || shape[4] > LATENT || shape[5] > ATTENTION_HEAD) {
            throw new UnsupportedOperationException("fused latent attention on " + name() + ": " + t
                    + ", latents of " + shape[4] + " and rotary keys of " + shape[5]);
        }
        long rows = (long) shape[0] * shape[1] * shape[2];
        if (rows == 0) {
            return;
        }
        HipContext cx = ctx();
        HipStorage xqc = as(cs(qc), t);
        HipStorage xqr = as(cs(qr), t);
        HipStorage xc = as(cs(c), t);
        HipStorage xr = as(cs(r), t);
        HipStorage xb = bias == null ? null : as(cs(bias), t);
        cx.args().ptr(xqc.address()).ptr(xqr.address()).ptr(xc.address()).ptr(xr.address())
                .ptr(xb == null ? 0 : xb.address()).ptr(z.address()).i32(shape[1]).i32(shape[2]).i32(shape[3])
                .i32(shape[4]).i32(shape[5]).f32(scale).i64(rows).i32(xb == null ? 0 : 1).i64(strides[0])
                .i64(strides[1]).i64(strides[2]).i64(strides[3]);
        warps(cx, kernel("attention_latent", t), rows);
        done(xqc, cs(qc));
        done(xqr, cs(qr));
        done(xc, cs(c));
        done(xr, cs(r));
        if (xb != null) {
            done(xb, cs(bias));
        }
    }

    @Override
    public synchronized void attentionBackward(Storage q, Storage k, Storage v, Storage out, Storage lse,
            Storage grad, int[] shape, float scale, boolean causal, Storage dq, Storage dk, Storage dv) {
        HipStorage gq = cs(dq);
        HipStorage gk = cs(dk);
        HipStorage gv = cs(dv);
        DType t = gq.dtype;
        checkAttention(shape, t);
        HipContext c = ctx();
        long rows = (long) shape[0] * shape[1] * shape[3];
        long keyRows = (long) shape[0] * shape[2] * shape[4];
        if (rows == 0 || keyRows == 0) {
            fill(gq, 0);
            fill(gk, 0);
            fill(gv, 0);
            return;
        }
        HipStorage xq = as(cs(q), t);
        HipStorage xk = as(cs(k), t);
        HipStorage xv = as(cs(v), t);
        HipStorage xo = as(cs(out), t);
        HipStorage xl = as(cs(lse), t);
        HipStorage xg = as(cs(grad), t);
        HipStorage delta = raw(rows, DType.FLOAT32, null);
        c.args().ptr(xo.address()).ptr(xg.address()).ptr(delta.address()).i64(rows).i32(shape[6]);
        warps(c, kernel("attention_delta", t), rows);
        int[] s = attShape(shape, scale, causal);
        c.args().ptr(xq.address()).ptr(xk.address()).ptr(xv.address()).ptr(xl.address()).ptr(xg.address())
                .ptr(delta.address()).ptr(gq.address()).ints(s).i64(rows);
        warps(c, kernel("attention_dq", t), rows);
        c.args().ptr(xq.address()).ptr(xk.address()).ptr(xv.address()).ptr(xl.address()).ptr(xg.address())
                .ptr(delta.address()).ptr(gk.address()).ptr(gv.address()).ints(s).i64(keyRows);
        warps(c, kernel("attention_dkv", t), keyRows);
        delta.release();
        done(xq, cs(q));
        done(xk, cs(k));
        done(xv, cs(v));
        done(xo, cs(out));
        done(xl, cs(lse));
        done(xg, cs(grad));
    }

    // ------------------------------------------------------------------ Fourier transforms

    /** The longest rows {@code fft_rows} transforms, in shared memory. */
    static final int FFT_LENGTH = 2048;

    @Override
    public synchronized Storage fft(Storage in, long rows, int n, boolean inverse) {
        if (n < 2 || n > FFT_LENGTH || Integer.bitCount(n) != 1) {
            return Backend.super.fft(in, rows, n, inverse);
        }
        HipContext c = ctx();
        HipStorage x = as(cs(in), DType.FLOAT32);
        HipStorage y = raw(rows * n * 2, DType.FLOAT32, null);
        c.args().ptr(x.address()).ptr(y.address()).i64(rows).i32(n).i32(Integer.numberOfTrailingZeros(n))
                .i32(inverse ? 1 : 0);
        c.launch("fft_rows", (int) Math.max(1, Math.min(rows, 1 << 20)), 1, 1);
        done(x, cs(in));
        return y;
    }

    // ------------------------------------------------------------------ RMSNorm

    @Override
    public boolean hasRmsNorm(DType type) {
        return type != DType.FLOAT64 && isAvailable();
    }

    @Override
    public synchronized void rmsNorm(Storage x, Storage weight, long rows, int cols, float eps, Storage out) {
        HipStorage y = cs(out);
        HipContext c = ctx();
        HipStorage xs = as(cs(x), y.dtype);
        HipStorage ws = as(cs(weight), y.dtype);
        c.args().ptr(xs.address()).ptr(ws.address()).ptr(y.address()).i64(rows).i32(cols).f32(eps);
        c.launch(kernel("rms_norm", y.dtype), (int) Math.min(rows, 1 << 20), 1, 1);
        done(xs, cs(x));
        done(ws, cs(weight));
    }

    // ------------------------------------------------------------------ windows and indices

    @Override
    public synchronized void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        HipStorage z = cs(out);
        HipContext c = ctx();
        HipStorage x = as(cs(in), z.dtype);
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
        HipStorage z = cs(out);
        HipContext c = ctx();
        HipStorage x = as(cs(cols), z.dtype);
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
    private int[] indices(HipStorage storage, int count, long length) {
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
        HipStorage z = cs(out);
        HipContext c = ctx();
        indices(cs(indices), count, length);
        HipStorage x = as(cs(in), z.dtype);
        HipStorage ind = as(cs(indices), DType.FLOAT32);
        long total = outer * count * inner;
        c.args().ptr(x.address()).ptr(ind.address()).ptr(z.address()).i64(length).i64(inner).i64(count).i64(total);
        c.launch(kernel("gather", z.dtype), total);
        done(x, cs(in));
        done(ind, cs(indices));
    }

    @Override
    public synchronized void scatterAdd(Storage src, long outer, int count, long inner, Storage indices, long length,
            Storage out) {
        HipStorage z = cs(out);
        HipContext c = ctx();
        int[] idx = indices(cs(indices), count, length);
        HipStorage s = as(cs(src), z.dtype);
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
        HipStorage off = uploadInts(offsets);
        HipStorage ent = uploadInts(entries);
        long total = outer * length * inner;
        c.args().ptr(s.address()).ptr(off.address()).ptr(ent.address()).ptr(z.address()).i64(length).i64(inner)
                .i64(count).i64(total);
        c.launch(kernel("scatter_add", z.dtype), total);
        done(s, cs(src));
        off.release();
        ent.release();
    }
}
