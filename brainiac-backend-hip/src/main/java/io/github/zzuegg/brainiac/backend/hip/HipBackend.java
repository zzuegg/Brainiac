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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

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
 * <li>{@code brainiac.hip.slabs=false}: a hipMalloc for every small block instead of cutting
 * blocks up to 1 MB from slabs</li>
 * <li>{@code brainiac.hip.profile=true}: time every kernel, wait, copy and allocation, waiting
 * after each launch, and list the costliest in {@link #statistics()}; {@code =gpu}: time
 * kernels, copies and rocBLAS calls on the GPU with events, without waiting, and how long the
 * GPU sat idle before each</li>
 * <li>{@code brainiac.hip.devices}: {@code one} for only one device; {@code all} for one per GPU
 * ({@code hip}, {@code hip:1}, ...); by default one per GPU of the first one's architecture</li>
 * <li>{@code brainiac.hip.fusedAttention=false}: attention from separate operations</li>
 * <li>{@code brainiac.hip.fusion=false}: every element-wise step as a kernel of its own instead
 * of chains generated as one kernel ({@link HipFusion}), and transposed operands copied before
 * a product instead of read in place</li>
 * <li>{@code brainiac.hip.fewRows=false}: products of up to 16 rows in float16 and bfloat16 by
 * rocBLAS instead of a kernel of its own</li>
 * </ul>
 *
 * <p>Attention runs fused (forward, backward, and with a bias) for heads up to 256; packed
 * weights stay packed on the device, in the Vulkan backend's compact layout and read directly by products of
 * up to 8 rows; Fourier
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
                    synchronized (OPEN) {
                        if (OPEN.isEmpty()) {
                            Runtime.getRuntime().addShutdownHook(new Thread(HipBackend::closeAllAtExit, "hip-close"));
                        }
                        OPEN.add(this);
                    }
                }
                if (Boolean.getBoolean("brainiac.hip.statistics")) {
                    // printed by the closing hook when there is one: hooks run at the same time
                    if ("false".equals(System.getProperty("brainiac.hip.closeAtExit"))) {
                        Runtime.getRuntime().addShutdownHook(new Thread(() -> System.err.println(name() + ": "
                                + statistics()), "hip-statistics"));
                    }
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

    /** The backends with a context, for closing them all at exit. */
    private static final List<HipBackend> OPEN = new java.util.ArrayList<>();

    /**
     * Releases the devices at exit ({@code -Dbrainiac.hip.closeAtExit=false}: leave it to the
     * runtime), in one hook: every context first, then each device reset once. A reset clears
     * the device for the whole process, so resetting it while another context on it still held
     * memory and a rocBLAS handle made the process abort.
     */
    private static void closeAllAtExit() {
        List<HipBackend> all;
        synchronized (OPEN) {
            all = new java.util.ArrayList<>(OPEN);
        }
        List<HipContext> closed = new java.util.ArrayList<>();
        for (HipBackend b : all) {
            HipContext c = b.closeAtExit();
            if (c != null) {
                closed.add(c);
            }
        }
        Set<Integer> reset = new HashSet<>();
        for (HipContext c : closed) {
            if (reset.add(c.ordinal)) {
                try {
                    c.resetDevice();
                } catch (RuntimeException e) {
                    System.err.println("brainiac hip: resetting device " + c.ordinal + ": " + e.getMessage());
                }
            }
        }
    }

    /** Closes this backend's context, but leaves the device unreset; returns the context, or null. */
    private synchronized HipContext closeAtExit() {
        HipContext c = hip;
        if (c != null) {
            if (Boolean.getBoolean("brainiac.hip.statistics")) {
                System.err.println(name() + ": " + statistics());
            }
            try {
                c.close();
            } catch (RuntimeException e) {
                System.err.println("brainiac hip: closing " + name() + ": " + e.getMessage());
            }
            hip = null;
            unavailable = "closed at exit";
        }
        return c;
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
        return "init " + initNanos / 1_000_000 + " ms, " + c.launches + " kernel launches ("
                + c.callNanos / 1_000_000 + " ms in launch and copy calls), " + c.products
                + " rocBLAS products, " + c.waits + " waits, " + (c.uploadedBytes >> 20) + " MB uploaded, "
                + (c.downloadedBytes >> 20) + " MB downloaded, " + (c.inUse() >> 20) + " MB in use, "
                + (c.peak() >> 20) + " MB peak, " + (c.reserved() >> 20) + " MB reserved, " + c.collections
                + " collections, fusion " + (FUSION ? fusedLaunches + " fused launches of " + c.fusedKernels()
                        + " generated kernels (" + c.fusedCompileNanos / 1_000_000 + " ms compiling)" : "off")
                + c.profileReport() + c.sitesReport();
    }

    /**
     * For deciding whether to replay decoding as HIP graphs: the time per kernel of a chain of
     * {@code kernels} small element-wise kernels (896 float16 values each, as a decoded token's
     * are), launched one by one, replayed as an instantiated graph, and recorded again and
     * updated into that graph before each replay (what decoding would do every token).
     */
    synchronized String graphBenchmark(int kernels, int reps) {
        HipContext c = ctx();
        if (!c.cu.hasGraphs()) {
            return "no graphs in this HIP runtime";
        }
        int n = 896;
        HipStorage x = raw(n, DType.FLOAT16, null);
        HipStorage y = raw(n, DType.FLOAT16, null);
        fill(x, 1);
        fill(y, 0);
        ready(x);
        ready(y);
        String name = kernel("binary_contiguous", DType.FLOAT16);
        Runnable chain = () -> {
            for (int i = 0; i < kernels; i++) {
                c.args().ptr(x.address()).ptr(y.address()).ptr(y.address()).i64(n).i32(BinaryOp.ADD.ordinal());
                c.launch(name, n);
            }
        };
        chain.run();
        c.finish();
        long t0 = System.nanoTime();
        for (int r = 0; r < reps; r++) {
            chain.run();
        }
        c.finish();
        double plain = (System.nanoTime() - t0) / 1e3 / reps / kernels;
        c.cu.beginCapture(c.stream);
        chain.run();
        long graph = c.cu.endCapture(c.stream);
        long exec = c.cu.instantiate(graph);
        c.cu.launchGraph(exec, c.stream);
        c.finish();
        t0 = System.nanoTime();
        for (int r = 0; r < reps; r++) {
            c.cu.launchGraph(exec, c.stream);
        }
        c.finish();
        double replay = (System.nanoTime() - t0) / 1e3 / reps / kernels;
        boolean updated = true;
        long captureNanos = 0;
        t0 = System.nanoTime();
        for (int r = 0; r < reps; r++) {
            long s = System.nanoTime();
            c.cu.beginCapture(c.stream);
            chain.run();
            long g = c.cu.endCapture(c.stream);
            updated &= c.cu.updateGraph(exec, g);
            c.cu.destroyGraph(g);
            captureNanos += System.nanoTime() - s;
            c.cu.launchGraph(exec, c.stream);
        }
        c.finish();
        double recorded = (System.nanoTime() - t0) / 1e3 / reps / kernels;
        c.cu.destroyGraphExec(exec);
        c.cu.destroyGraph(graph);
        x.release();
        y.release();
        return String.format(java.util.Locale.ROOT,
                "%d kernels x %d: launched %.2f us each; graph replayed %.2f us each; recorded, updated%s and replayed"
                        + " %.2f us each (recording and updating %.2f us of it)",
                kernels, reps, plain, replay, updated ? "" : " (UPDATE FAILED)", recorded,
                captureNanos / 1e3 / reps / kernels);
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

    /** {@code s} with its values computed: an operand read now. */
    private HipStorage cs(Storage s) {
        HipStorage c = ls(s);
        ready(c);
        return c;
    }

    /** {@code s} as it is, possibly pending. */
    private HipStorage ls(Storage s) {
        if (s instanceof HipStorage c && c.owner == this) {
            return c;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to " + name());
    }

    /**
     * {@code s} about to be written by an operation that is not fused: pending results that
     * read it are computed first, then its own (a write may cover only part of it); starts a
     * new fusion generation.
     */
    private HipStorage wr(Storage s) {
        HipStorage c = ls(s);
        flushReaders(c);
        ready(c);
        generation++;
        c.indexBound = 0;   // overwritten: whatever bounded its values before no longer does
        return c;
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
        ready(c);
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
        if (storage instanceof Packed || storage instanceof Deferred) {
            return;   // weights: freed with the tensor that holds them, by the collector
        }
        if (storage instanceof HipStorage s && s.owner == this) {
            if (read(s)) {
                s.releaseWanted = true;   // freed once the pending results reading it are computed or dropped
                return;
            }
            // a pending result no other one reads: never computed (one that inlined it holds a
            // copy of its expression), and what it read may now be freed
            HipFusion.Expr e = drop(s);
            s.release();
            if (e != null) {
                releaseUnread(e);
            }
        }
    }

    @Override
    public synchronized void evaluate(List<Storage> storages) {
        for (Storage s : storages) {
            ready(ls(s));
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

    /**
     * Queues a small float32 read-back (decoding's chosen tokens) and returns at once, so the
     * next step can be queued while the device still works on this one; larger or other
     * reads happen now.
     */
    @Override
    public synchronized java.util.function.Supplier<float[]> downloadLater(Storage src) {
        HipStorage s = cs(src);
        HipContext c = ctx();
        if (s.dtype != DType.FLOAT32 || !c.readsLater(s.bytes())) {
            float[] now = download(s);
            return () -> now;
        }
        HipContext.Pending p = c.downloadLater(s.address(), s.bytes());
        int n = Math.toIntExact(s.size);
        return new java.util.function.Supplier<>() {
            private float[] got;

            @Override
            public float[] get() {
                synchronized (HipBackend.this) {
                    if (got == null) {
                        got = new float[n];
                        MemorySegment.copy(c.read(p), java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, got, 0, n);
                        c.release(p);
                    }
                    return got;
                }
            }
        };
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
        HipStorage x0 = ls(in);
        HipStorage y0 = ls(out);
        if (x0.dtype != y0.dtype && x0.dtype != DType.FLOAT64 && y0.dtype != DType.FLOAT64 && fusable(y0)
                && x0.size == y0.size) {
            // between float32, float16 and bfloat16: read in the other format by what uses it
            defer(y0, HipFusion.whole(x0));
            return;
        }
        HipStorage x = cs(in);
        HipStorage y = wr(out);
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
        HipStorage s0 = ls(dst);
        if (fusable(s0)) {
            defer(s0, new HipFusion.Const(value));
            return;
        }
        fillNow(wr(dst), value);
    }

    private void fillNow(HipStorage s, float value) {
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
        if (fusable(s)) {
            // generated inside the kernel that uses the values, never stored
            defer(s, new HipFusion.Random(seed));
            return s;
        }
        hip.args().ptr(s.address()).i64(size).i64(seed);
        hip.launch("random_values", size);
        return s;
    }

    // ------------------------------------------------------------------ fusion

    /** Whether element-wise chains are fused ({@code -Dbrainiac.hip.fusion=false} runs every step on its own). */
    static final boolean FUSION = !"false".equalsIgnoreCase(System.getProperty("brainiac.hip.fusion"));
    /**
     * Counts operations that are not fused. A pending result recorded before the latest of
     * them is computed before anything builds on it, so chains end at the same points in every
     * step and their generated kernels repeat instead of growing.
     */
    private long generation;
    private long fusedLaunches;

    /** Whether an element-wise result can be recorded instead of computed now. */
    private boolean fusable(HipStorage out, Storage... inputs) {
        if (!FUSION || out.size == 0 || out.size > Integer.MAX_VALUE || !isAvailable()) {
            return false;
        }
        for (Storage in : inputs) {
            if (ls(in).dtype != out.dtype) {
                return false;
            }
        }
        return true;
    }

    /** Records {@code e} as the pending value of {@code out}, computing inputs first where it pays. */
    private void defer(HipStorage out, HipFusion.Expr e) {
        long t0 = HipContext.PROFILE ? System.nanoTime() : 0;
        record(out, e);
        if (HipContext.PROFILE) {
            hip.record("host: recording element-wise steps", System.nanoTime() - t0);
        }
    }

    private void record(HipStorage out, HipFusion.Expr e) {
        // pending results reading out's current value compute it before it changes
        flushReaders(out);
        // in place: the output's value so far is computed first and read from memory, and the
        // result is computed now, as the output cannot both stay pending and be read by it
        boolean inPlace = HipFusion.reads(e, out);
        if (inPlace) {
            ready(out);
        }
        // a view adds no step of its own, so it cannot grow a chain
        boolean view = e instanceof HipFusion.Src;
        for (HipFusion.Src in : HipFusion.sources(e)) {
            HipStorage st = in.storage();
            if (st.pending == null || cheap(st.pending)) {
                continue;
            }
            boolean old = st.pendingGeneration != generation && !view;
            // read at more indices than it has (a broadcast): computed once, not at every read
            boolean repeated = in.count() > st.size;
            if (old || repeated) {
                ready(st);
            }
        }
        if (HipFusion.cost(e) > HipFusion.MAX_NODES) {
            for (HipFusion.Src in : HipFusion.sources(e)) {
                ready(in.storage());
            }
        }
        HipFusion.Expr replaced = drop(out);
        out.pending = e;
        out.pendingNodes = Math.min(HipFusion.nodes(e), 1 << 20);
        out.pendingGeneration = generation;
        for (HipFusion.Src in : HipFusion.sources(e)) {
            HipStorage st = in.storage();
            if (st.readers == null) {
                st.readers = new java.util.ArrayList<>(2);
            }
            if (st.readers.isEmpty() || st.readers.getLast().get() != out) {
                if (st.readers.size() >= 64) {
                    st.readers.removeIf(r -> r.get() == null || r.get().pending == null);
                }
                st.readers.add(new java.lang.ref.WeakReference<>(out));
            }
        }
        if (replaced != null) {
            releaseUnread(replaced);
        }
        if (inPlace) {
            ready(out);
        }
    }

    /** Pending values that cost nothing to compute again wherever they are read. */
    private static boolean cheap(HipFusion.Expr e) {
        return e instanceof HipFusion.Const || e instanceof HipFusion.Random
                || e instanceof HipFusion.Src s && s.storage().pending == null;
    }

    /** Forgets {@code s}'s pending value without computing it; returns it. */
    private static HipFusion.Expr drop(HipStorage s) {
        HipFusion.Expr e = s.pending;
        s.pending = null;
        s.pendingNodes = 0;
        return e;
    }

    /** Whether a pending result still reads {@code s} directly. */
    private static boolean read(HipStorage s) {
        if (s.readers == null) {
            return false;
        }
        for (java.lang.ref.WeakReference<HipStorage> r : s.readers) {
            HipStorage p = r.get();
            if (p != null && p.pending != null && HipFusion.sources(p.pending).stream().anyMatch(x -> x.storage() == s)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Computes the pending results that read {@code s}, before it is written. Those reading it
     * through another pending result are covered too: that one is computed, and they read it.
     */
    private void flushReaders(HipStorage s) {
        if (s.readers == null || s.readers.isEmpty()) {
            return;
        }
        List<java.lang.ref.WeakReference<HipStorage>> all = List.copyOf(s.readers);
        s.readers.clear();
        for (java.lang.ref.WeakReference<HipStorage> r : all) {
            HipStorage p = r.get();
            if (p != null && p != s && p.pending != null
                    && HipFusion.sources(p.pending).stream().anyMatch(x -> x.storage() == s)) {
                ready(p);
            }
        }
    }

    /** Computes a pending storage, so that its memory holds its values. */
    private void ready(HipStorage s) {
        if (s.pending == null) {
            return;
        }
        // no longer pending first: an expression computed in place reads the stored values
        HipFusion.Expr e = drop(s);
        compute(s, e);
        releaseUnread(e);
    }

    /**
     * Frees what {@code e} read whose release waited for pending results to stop reading it,
     * now that {@code e} is computed or dropped.
     */
    private void releaseUnread(HipFusion.Expr e) {
        for (HipFusion.Src in : HipFusion.sources(e)) {
            HipStorage st = in.storage();
            if (st.releaseWanted && !read(st)) {
                st.releaseWanted = false;
                release(st);
            }
        }
    }

    /** Computes expression {@code e} into {@code s}, which no longer holds it pending. */
    private void compute(HipStorage s, HipFusion.Expr e) {
        HipContext c = ctx();
        if (e instanceof HipFusion.Const k) {
            fillNow(s, k.value());
            return;
        }
        if (e instanceof HipFusion.Src v && v.storage().pending == null && v.dims().length == 1
                && v.strides()[0] == 1 && v.storage().dtype == s.dtype) {
            // a contiguous range of stored values
            int bytes = s.dtype.bytes();
            c.copy(s.address(), v.storage().address(v.offset()), s.size * bytes);
            return;
        }
        long t0 = HipContext.PROFILE ? System.nanoTime() : 0;
        HipFusion.Kernel k = HipFusion.generate(e, s.dtype);
        if (HipContext.PROFILE) {
            c.record("host: generating kernel sources", System.nanoTime() - t0);
        }
        launch(c, k, s.address(), s.size);
    }

    /** Runs a generated kernel over {@code n} items, storing at {@code out}. */
    private void launch(HipContext c, HipFusion.Kernel k, long out, long n) {
        long f = c.fusedFunction(k.source());
        for (HipStorage in : k.loads()) {
            in.address();   // memory taken before the arguments are written
        }
        c.args().ptr(out).i32((int) n).struct(k.metaBytes(), k::writeMeta);
        c.launchFused(f, n);
        fusedLaunches++;
    }

    /**
     * Where contiguous values of {@code shape} land in an array with {@code outStrides}, when
     * that is rows of {@code inner} values {@code outerStride} apart: those two, else null.
     */
    private static int[] rows(int[] shape, long[] outStrides) {
        int d = shape.length - 1;
        long inner = 1;
        while (d >= 0 && (shape[d] == 1 || outStrides[d] == inner)) {
            inner *= shape[d];
            d--;
        }
        long outer = -1;
        long count = 1;
        for (; d >= 0; d--) {
            if (shape[d] == 1) {
                continue;
            }
            if (outer < 0) {
                outer = outStrides[d];
            } else if (outStrides[d] != outer * count) {
                return null;
            }
            count *= shape[d];
        }
        if (inner > Integer.MAX_VALUE || outer > Integer.MAX_VALUE) {
            return null;
        }
        return new int[] {(int) inner, outer < 0 ? (int) inner : (int) outer};
    }

    /** Computes {@code in}'s pending value into rows of {@code z} (see {@link #rows}); {@code in} keeps it. */
    private void writePending(HipStorage in, HipStorage z, long outOffset, long n, int inner, int outerStride) {
        HipFusion.Kernel k = HipFusion.generate(in.pending, z.dtype, inner, outerStride);
        launch(ctx(), k, z.address(outOffset), n);
    }

    // ------------------------------------------------------------------ element-wise

    @Override
    public synchronized void unary(UnaryOp op, Storage in, Storage out) {
        if (fusable(ls(out), in)) {
            defer(ls(out), new HipFusion.Un(op.ordinal(), HipFusion.whole(ls(in))));
            return;
        }
        HipStorage y = wr(out);
        HipContext c = ctx();
        HipStorage x = as(cs(in), y.dtype);
        c.args().ptr(x.address()).ptr(y.address()).i64(y.size).i32(op.ordinal());
        c.launch(kernel("unary", y.dtype), y.size);
        done(x, cs(in));
    }

    @Override
    public synchronized void activation(ActivationOp op, float param, Storage in, Storage out) {
        if (fusable(ls(out), in)) {
            defer(ls(out), new HipFusion.Act(op.ordinal(), param, HipFusion.whole(ls(in))));
            return;
        }
        HipStorage y = wr(out);
        HipContext c = ctx();
        HipStorage x = as(cs(in), y.dtype);
        c.args().ptr(x.address()).ptr(y.address()).i64(y.size).i32(op.ordinal()).f32(param);
        c.launch(kernel("activation", y.dtype), y.size);
        done(x, cs(in));
    }

    @Override
    public synchronized void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        if (fusable(ls(out), in, grad)) {
            defer(ls(out), new HipFusion.ActBack(op.ordinal(), param, HipFusion.whole(ls(in)),
                    HipFusion.whole(ls(grad))));
            return;
        }
        HipStorage y = wr(out);
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
        HipStorage z0 = ls(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        if (fusable(z0, a, b)) {
            HipFusion.Src x = HipFusion.strided(ls(a), 0, shape, aStrides);
            HipFusion.Src y = HipFusion.strided(ls(b), 0, shape, bStrides);
            if (x != null && y != null) {
                defer(z0, new HipFusion.Bin(op.ordinal(), x, y));
                return;
            }
        }
        HipStorage z = wr(out);
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
        HipStorage z0 = ls(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        HipStorage in0 = ls(in);
        if (outOffset == 0 && n == z0.size && Arrays.equals(outStrides, Axes.contiguous(shape)) && fusable(z0, in)) {
            // a view: recorded, and read through its strides by whatever uses it
            HipFusion.Src x = HipFusion.strided(in0, inOffset, shape, inStrides);
            if (x != null) {
                defer(z0, x);
                return;
            }
        }
        if (FUSION && in0.pending != null && inOffset == 0 && in0.dtype == z0.dtype && n == in0.size
                && Arrays.equals(inStrides, Axes.contiguous(shape)) && n < Integer.MAX_VALUE) {
            int[] rows = rows(shape, outStrides);
            if (rows != null) {
                // a pending value written into part of an array (a key/value cache step): its
                // kernel stores there directly, without computing it into a temporary first
                HipStorage z = wr(out);
                if (in0.pending != null) {   // unless it read the array, and was computed for the write
                    writePending(in0, z, outOffset, n, rows[0], rows[1]);
                    return;
                }
            }
        }
        HipStorage z = wr(out);
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
        HipStorage y = wr(out);
        long rows = outer * inner;
        if (rows == 0) {
            return;
        }
        HipContext c = ctx();
        HipStorage x = as(cs(in), y.dtype);
        if (op == ReduceOp.ARGMAX) {
            if (y.dtype == DType.FLOAT32 || y.dtype == DType.FLOAT64) {
                y.indexBound = length;   // a gather by these indices need not download them to check
            }
            if (rows < 2L * c.multiprocessors && length > 4096 && rows <= FEW_COUNTERS) {
                // few long rows (a vocabulary's logits): each split over blocks, which the
                // row's last block to finish combines
                int pieces = (int) Math.max(1, Math.min(Math.min(256, 4L * c.multiprocessors / rows),
                        (length + 2047) / 2048));
                HipStorage part = raw(2L * rows * pieces, DType.FLOAT64, null);
                long counters = counters().address();
                c.args().ptr(x.address()).ptr(y.address()).ptr(part.address()).ptr(counters).i64(rows).i64(length)
                        .i64(inner).i32(pieces);
                c.launch(kernel("argmax_split", y.dtype), pieces, (int) rows, 1);
                part.release();
                done(x, cs(in));
                return;
            }
            c.args().ptr(x.address()).ptr(y.address()).i64(rows).i64(length).i64(inner);
            if (rows < 2L * c.multiprocessors && length > 1024) {
                // few long rows (a vocabulary's logits): a block each, not a thread
                c.launch(kernel("argmax_block", y.dtype), (int) rows, 1, 1);
            } else {
                c.launch(kernel("argmax", y.dtype), rows);
            }
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
        HipStorage z = wr(out);
        if ((long) batch * m * n == 0) {
            return;
        }
        HipContext c = ctx();
        if (k == 0) {
            fill(z, 0);
            ready(z);
            return;
        }
        if (c.rocblas != null) {
            // a transposed or otherwise strided operand still pending: rocBLAS reads it in place
            Operand va = view(ls(a), batch, m, k, z.dtype);
            Operand vb = view(ls(b), batch, k, n, z.dtype);
            if (vb != null && vb.transposed() && (m > FEW || va != null)) {
                // rocBLAS reads a transposed right operand very slowly (a tied head over a
                // prompt's rows: 260 ms where a copy and a plain product take a few)
                vb = null;
            }
            if (va != null || vb != null) {
                HipStorage x = va != null ? null : as(cs(a), z.dtype);
                HipStorage y = vb != null ? null : as(cs(b), z.dtype);
                if (va == null) {
                    va = Operand.of(x, m, k);
                }
                if (vb == null) {
                    vb = Operand.of(y, k, n);
                }
                strided(c, va, vb, batch, m, k, n, z);
                if (x != null) {
                    done(x, cs(a));
                }
                if (y != null) {
                    done(y, cs(b));
                }
                return;
            }
        }
        HipStorage x = as(cs(a), z.dtype);
        HipStorage y = as(cs(b), z.dtype);
        DType t = z.dtype;
        boolean blas = c.rocblas != null;
        if (DENSE_MV && batch == 1 && m <= PACKED_ROWS && n % 4 == 0 && t != DType.FLOAT64) {
            // decoding's products: four columns a thread from 8-byte loads (rocBLAS's gemv
            // reached a third of the bandwidth on a feed-forward network's shapes)
            matvec(c, "dense_mv", x, y.address(), null, m, k, n, z);
        } else if (batch == 1 && m == 1 && !MATVEC.equals("few") && t == DType.FLOAT16 && blas && c.rocblas.hasHalfGemv()
                && !fewFaster(c, x, y, k, n, z)) {
            long t0 = HipContext.PROFILE ? System.nanoTime() : 0;
            long g = c.gpuStart();
            c.rocblas.halfGemv(false, n, k, n, y.address(), x.address(), z.address());
            c.gpuEnd(g, "rocBLAS gemv " + k + "x" + n);
            c.products++;
            if (HipContext.PROFILE) {
                c.profileSync();
                c.record("rocBLAS FLOAT16 gemv", System.nanoTime() - t0);
            }
        } else if (FEW_ROWS && batch == 1 && m >= 1 && m <= FEW && (t == DType.FLOAT16 || t == DType.BFLOAT16)) {
            // a handful of rows: a BLAS library runs these 16-bit products slower than float32 ones
            few(c, "few_dense", x, y.address(), m, k, n, z);
        } else if (blas) {
            HipContext.trace("rocBLAS " + t + " " + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n + "]");
            long t0 = HipContext.PROFILE ? System.nanoTime() : 0;
            long g = c.gpuStart();
            c.rocblas.gemm(t, false, false, m, n, k, x.address(), k, (long) m * k, y.address(), n, (long) k * n,
                    z.address(), (long) m * n, batch);
            c.gpuEnd(g, "rocBLAS gemm " + t);
            c.products++;
            if (HipContext.PROFILE) {
                c.profileSync();
                c.record("rocBLAS " + t + (m <= FEW ? " few rows" : ""), System.nanoTime() - t0);
            }
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

    @Override
    public boolean readsTransposedInPlace() {
        // a pending transpose is read by rocBLAS in place
        return FUSION && isAvailable() && hip.rocblas != null;
    }

    /**
     * How rocBLAS reads a matrix operand: from {@code values} at {@code offset}, rows {@code ld}
     * apart (columns, when {@code transposed}), {@code batch} apart from one matrix to the next.
     */
    private record Operand(HipStorage values, long offset, long batch, int ld, boolean transposed) {

        /** Stored {@code rows x cols} matrices one after another. */
        static Operand of(HipStorage values, int rows, int cols) {
            return new Operand(values, 0, (long) rows * cols, cols, false);
        }
    }

    /** A pending strided read of stored values ({@code [batch,] rows, cols}), as an operand; null if none. */
    private static Operand view(HipStorage s, int batch, int rows, int cols, DType type) {
        if (!(s.pending instanceof HipFusion.Src src) || src.direct() || src.storage().pending != null
                || src.storage().dtype != type) {
            return null;
        }
        int[] d = src.dims();
        int[] st = src.strides();
        for (int x : st) {
            if (x < 0) {
                return null;
            }
        }
        long b;
        int row;
        int col;
        if (batch == 1 && d.length == 2 && d[0] == rows && d[1] == cols) {
            b = 0;
            row = st[0];
            col = st[1];
        } else if (d.length == 3 && d[0] == batch && d[1] == rows && d[2] == cols) {
            b = st[0];
            row = st[1];
            col = st[2];
        } else {
            return null;
        }
        if (col == 1 && row >= cols) {
            return new Operand(src.storage(), src.offset(), b, row, false);
        }
        if (row == 1 && col >= rows) {
            return new Operand(src.storage(), src.offset(), b, col, true);
        }
        return null;
    }

    /** {@code z = a x b} by rocBLAS, the operands read as they lie. */
    private void strided(HipContext c, Operand a, Operand b, int batch, int m, int k, int n, HipStorage z) {
        DType t = z.dtype;
        long pa = a.values().address(a.offset());
        long pb = b.values().address(b.offset());
        long t0 = HipContext.PROFILE ? System.nanoTime() : 0;
        if (batch == 1 && m <= FEW && b.transposed() && !a.transposed() && (m == 1 || a.ld() == k)) {
            // few rows times a transposed matrix (a tied output head): a warp per row of the
            // stored matrix, read contiguously; rocBLAS's transposed gemv is several times slower
            int vec = t.bytes() == 2 && b.ld() % 4 == 0 && pb % 8 == 0 ? 1 : 0;
            c.args().ptr(pa).ptr(pb).ptr(z.address()).i32(m).i32(k).i32(n).i64(b.ld()).i32(vec);
            c.launch(kernel("few_t", t), (n + HipContext.GROUP / 32 - 1) / (HipContext.GROUP / 32), 1, 1);
            return;
        } else if (batch == 1 && m == 1 && t == DType.FLOAT16 && MATVEC.equals("rocblas") && c.rocblas.hasHalfGemv()
                && !a.transposed()) {
            long g = c.gpuStart();
            c.rocblas.halfGemv(b.transposed(), b.ld(), k, n, pb, pa, z.address());
            c.gpuEnd(g, "rocBLAS gemv in place");
        } else {
            HipContext.trace("rocBLAS " + t + " " + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n
                    + "] in place" + (a.transposed() ? ", a transposed" : "") + (b.transposed() ? ", b transposed" : ""));
            long g = c.gpuStart();
            c.rocblas.gemm(t, a.transposed(), b.transposed(), m, n, k, pa, a.ld(), batch == 1 ? 0 : a.batch(), pb,
                    b.ld(), batch == 1 ? 0 : b.batch(), z.address(), (long) m * n, batch);
            c.gpuEnd(g, "rocBLAS gemm in place " + t);
        }
        c.products++;
        if (HipContext.PROFILE) {
            c.profileSync();
            c.record("rocBLAS " + t + " in place", System.nanoTime() - t0);
        }
        if (HipContext.DEBUG) {
            c.debugWait("rocBLAS " + t + " product in place " + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n
                    + "]");
        }
    }

    /** Whether products of a few 16-bit rows use a kernel of their own ({@code -Dbrainiac.hip.fewRows=false}: rocBLAS). */
    private static final boolean FEW_ROWS = !"false".equals(System.getProperty("brainiac.hip.fewRows"));
    /** The most rows {@code few_dense} takes (FEW_ROWS in kernels.hip). */
    static final int FEW = 16;
    /** Column blocks the few-row kernels can count finished pieces for: an int each in {@link #fewCounters}. */
    private static final int FEW_COUNTERS = 1 << 16;
    /** Counters of finished pieces per column block, all 0 between launches. */
    private HipStorage fewCounters;
    /**
     * Counters of finished pieces per row, for attention and argmax: all 0 between launches,
     * as the last piece of each row sets its counter back.
     */
    private HipStorage rowCounters;

    private HipStorage counters() {
        if (rowCounters == null) {
            rowCounters = raw(FEW_COUNTERS, DType.FLOAT32, null);
            fillNow(rowCounters, 0);
        }
        return rowCounters;
    }
    /** The columns a block of the few-row kernels covers: a thread each. */
    private static final int FEW_COLUMNS = HipContext.GROUP;

    /**
     * {@code out[m, n] = a[m, k] x w[k, n]} for at most {@link #FEW} rows: a thread per column of
     * {@code w}, the {@code k} rows split into pieces until the device is busy, the pieces' float
     * sums then added by {@code few_sum}. {@code w} is the dense matrix's address.
     */
    private void few(HipContext c, String kernel, HipStorage a, long w, int m, int k, int n, HipStorage out) {
        int columnBlocks = (n + FEW_COLUMNS - 1) / FEW_COLUMNS;
        int target = 4 * c.multiprocessors;
        int pieces = Math.max(1, Math.min((k + 127) / 128, target / columnBlocks));
        int chunk = ((k + pieces - 1) / pieces + 63) / 64 * 64;
        pieces = (k + chunk - 1) / chunk;
        HipStorage part = pieces > 1 ? raw((long) pieces * m * n, DType.FLOAT32, null) : null;
        long partAddress = part != null ? part.address() : 0;
        // the pieces are added by the last block of each column block to finish, which counts them
        boolean counted = part != null && columnBlocks <= FEW_COUNTERS;
        if (counted && fewCounters == null) {
            fewCounters = raw(FEW_COUNTERS, DType.FLOAT32, null);
            fill(fewCounters, 0);
            ready(fewCounters);
        }
        c.args().ptr(a.address()).ptr(w);
        c.ptr(partAddress).ptr(out.address()).i32(m).i32(k).i32(n).i32(chunk).ptr(counted ? fewCounters.address() : 0);
        c.launch(kernel(kernel, out.dtype), columnBlocks, pieces, 1);
        if (counted) {
            part.release();
        } else if (part != null) {
            long mn = (long) m * n;
            c.args().ptr(part.address()).ptr(out.address()).i64(mn).i32(pieces);
            c.launch(kernel("few_sum", out.dtype), mn);
            part.release();
        }
    }

    /**
     * What multiplies a single float16 row by a dense matrix, decoding's products
     * ({@code -Dbrainiac.hip.matvec}): {@code auto} (the default) whichever of rocBLAS's float16
     * gemv and the few-row kernel is faster for the shape, timed once; {@code rocblas} or
     * {@code few} always that one. Without rocBLAS, and for bfloat16 and several rows, the
     * few-row kernel.
     */
    private static final String MATVEC = System.getProperty("brainiac.hip.matvec", "auto");

    /** Per shape ({@code k << 32 | n}): whether the few-row kernel beat rocBLAS's gemv. */
    private final Map<Long, Boolean> fewFaster = new java.util.HashMap<>();

    /**
     * With {@code -Dbrainiac.hip.matvec=auto} (the default): whether the few-row kernel
     * multiplies a float16 row by this shape of matrix faster than rocBLAS's gemv, timed on the
     * GPU the first time the shape comes up (rocBLAS is fast for some shapes and several times
     * slower for others, such as a feed-forward network's down projection).
     */
    private boolean fewFaster(HipContext c, HipStorage x, HipStorage y, int k, int n, HipStorage z) {
        if (!MATVEC.equals("auto") || !FEW_ROWS || !c.cu.hasEvents()) {
            return false;
        }
        long key = (long) k << 32 | n;
        Boolean f = fewFaster.get(key);
        if (f == null) {
            Runnable blas = () -> c.rocblas.halfGemv(false, n, k, n, y.address(), x.address(), z.address());
            Runnable few = () -> few(c, "few_dense", x, y.address(), 1, k, n, z);
            float tb = c.timeOnGpu(blas, 5);
            float tf = c.timeOnGpu(few, 5);
            f = tf < 0.95f * tb;
            fewFaster.put(key, f);
            if (HipContext.DEBUG || HipContext.PROFILE || HipContext.GPU_PROFILE) {
                System.err.printf(java.util.Locale.ROOT, "brainiac hip: [1, %d] x [%d, %d] float16: rocBLAS %.1f us,"
                        + " few-row kernel %.1f us: %s%n", k, k, n, tb * 1e3, tf * 1e3, f ? "few" : "rocBLAS");
            }
        }
        return f;
    }

    // ------------------------------------------------------------------ packed (quantized) weights

    /**
     * A packed matrix on the device, in the layout {@link PackedLayout} prepares (the Vulkan
     * backend's): the levels as bytes, nibbles, nibble and high-bit planes or table indices, the
     * scales and offsets factored and float16 where the matrix allows.
     */
    private record Packed(int rows, int cols, PackedLayout layout, HipStorage levels, HipStorage hi,
            HipStorage scaleCodes, HipStorage offsetCodes, HipStorage scales, HipStorage offsets)
            implements PackedStorage {

        @Override
        public long size() {
            return (long) rows * cols;
        }

        @Override
        public long bytes() {
            long b = 0;
            for (HipStorage s : new HipStorage[] {levels, hi, scaleCodes, offsetCodes, scales, offsets}) {
                b += s == null ? 0 : s.bytes();
            }
            return b;
        }

        PackedMatrix host() {
            return layout.matrix();
        }

        /** Adds the arguments of a packed matrix in kernels.hip: six addresses, n, group, flags, perSuper, shift. */
        void push(HipContext c) {
            long any = levels.address();
            c.ptr(any).ptr(hi == null ? any : hi.address()).ptr(scaleCodes == null ? any : scaleCodes.address())
                    .ptr(offsetCodes == null ? any : offsetCodes.address()).ptr(scales.address())
                    .ptr(offsets == null ? any : offsets.address()).i32(cols).i32(host().group())
                    .i32(layout.layout()).i32(layout.perSuper()).f32(layout.shift());
        }
    }

    private HipStorage uploadBytes(byte[] data) {
        HipStorage s = raw(Math.max(1, (data.length + 3) / 4), DType.FLOAT32, null);
        hip.upload(s.address(), MemorySegment.ofArray(data));
        return s;
    }

    private HipStorage uploadHalves(short[] data) {
        HipStorage s = raw(Math.max(1, (data.length + 1) / 2), DType.FLOAT32, null);
        hip.upload(s.address(), MemorySegment.ofArray(data));
        return s;
    }

    /**
     * A packed matrix not uploaded until a product or unpacking first needs it: a model's
     * loader joins some matrices on the host ({@code Llama.fuseProjections} joins q, k and v,
     * and gate and up), and the parts it joins then never take device memory.
     */
    private static final class Deferred implements PackedStorage {
        final HipBackend owner;
        final PackedMatrix host;
        /** The uploaded matrix, once uploaded. */
        volatile Packed done;
        /** Whether the background upload should leave it (it was unpacked to the host first). */
        volatile boolean skip;

        Deferred(HipBackend owner, PackedMatrix host) {
            this.owner = owner;
            this.host = host;
        }

        @Override
        public int rows() {
            return host.rows();
        }

        @Override
        public int cols() {
            return host.cols();
        }

        @Override
        public long size() {
            return (long) host.rows() * host.cols();
        }

        @Override
        public long bytes() {
            Packed p = done;
            return p != null ? p.bytes() : host.bytes();
        }
    }

    /**
     * Packs a matrix for this backend. It uploads in the background while the caller goes on
     * (reading a model's next weights), or on its first use if that comes first. A matrix
     * unpacked to the host before it uploaded, as {@code Llama.fuseProjections} joins q, k and
     * v and gate and up, is left for its first use instead: joined parts then never upload.
     */
    @Override
    public PackedStorage pack(PackedMatrix matrix) {
        Deferred d = new Deferred(this, matrix);
        if (PREFETCH) {
            fetch(d, false);
        }
        return d;
    }

    /** Packs a matrix that uploads on its first use or {@link #prefetch}, not in the background at once. */
    @Override
    public PackedStorage packLater(PackedMatrix matrix) {
        return new Deferred(this, matrix);
    }

    @Override
    public void prefetch(PackedStorage packed) {
        if (packed instanceof Deferred d && d.owner == this) {
            fetch(d, true);
        }
    }

    @Override
    public void prefetchNow(PackedStorage packed) {
        if (packed instanceof Deferred d && d.owner == this) {
            packed(d);
        }
    }

    /** Uploads in the background, unless it was unpacked to the host first and {@code always} is false. */
    private void fetch(Deferred d, boolean always) {
        prefetcher.execute(() -> {
            if (always || !d.skip) {
                try {
                    packed(d);
                } catch (RuntimeException | OutOfMemoryError e) {
                    // out of memory or such: the first use tries again, and fails there
                }
            }
        });
    }

    /** Whether packed matrices upload in the background ({@code -Dbrainiac.hip.prefetch=false}: on first use). */
    private static final boolean PREFETCH = !"false".equals(System.getProperty("brainiac.hip.prefetch"));
    /** Uploads packed matrices in the background, one at a time; its thread starts with the first. */
    private final java.util.concurrent.ExecutorService prefetcher = java.util.concurrent.Executors
            .newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "brainiac-hip-upload");
                t.setDaemon(true);
                return t;
            });

    /** Uploads a packed matrix, preparing its layout without the lock. */
    private Packed upload(PackedMatrix matrix) {
        return upload(PackedLayout.of(matrix));
    }

    private Packed upload(PackedLayout st) {
        PackedMatrix matrix = st.matrix();
        synchronized (this) {
            ctx();
            HipStorage levels = uploadBytes(st.levels());
            HipStorage hi = st.hi() == null ? null : uploadBytes(st.hi());
            HipStorage scaleCodes = st.scaleCodes() == null ? null : uploadBytes(st.scaleCodes());
            HipStorage offsetCodes = st.offsetCodes() == null ? null : uploadBytes(st.offsetCodes());
            HipStorage scales = st.scaleHalves() != null ? uploadHalves(st.scaleHalves())
                    : (HipStorage) upload(st.scales());
            HipStorage offsets = st.offsetHalves() != null ? uploadHalves(st.offsetHalves())
                    : st.offsets() != null ? (HipStorage) upload(st.offsets()) : null;
            return new Packed(matrix.rows(), matrix.cols(), st, levels, hi, scaleCodes, offsetCodes, scales, offsets);
        }
    }

    @Override
    public PackedMatrix unpackToHost(PackedStorage packed) {
        if (packed instanceof Deferred d) {
            if (d.done == null) {
                d.skip = true;
            }
            return d.host;
        }
        if (packed instanceof Packed p) {
            return p.host();
        }
        return Backend.super.unpackToHost(packed);
    }

    /** {@code s} as this backend's packed form on the device: uploaded on first use, or moved here from another backend. */
    private Packed packed(PackedStorage s) {
        if (s instanceof Deferred d && d.owner == this) {
            Packed p = d.done;
            if (p != null) {
                return p;
            }
            // prepared without the lock (by both threads, if the background upload is at it too)
            PackedLayout st = PackedLayout.of(d.host);
            synchronized (this) {
                if (d.done == null) {
                    d.done = upload(st);
                }
                return d.done;
            }
        }
        if (s instanceof Packed p && p.levels().owner == this) {
            return p;
        }
        return upload(unpackToHost(s));
    }

    @Override
    public synchronized void unpack(PackedStorage packed, Storage out) {
        if (packed instanceof QuantizedRows) {
            Backend.super.unpack(packed, out);
            return;
        }
        Packed p = packed(packed);
        HipStorage z = wr(out);
        HipContext c = ctx();
        c.args();
        p.push(c);
        c.ptr(z.address()).i32(p.rows());
        if (p.cols() % 4 == 0) {
            c.launch(kernel("unpack_words", z.dtype), p.size() / 4);   // a word of levels per thread
        } else {
            c.launch(kernel("unpack_packed", z.dtype), p.size());
        }
    }

    /** Rows up to which a product reads packed levels directly ({@code packed_mv}); more unpack first. */
    private static final int PACKED_ROWS = 8;
    /** Blocks a product by packed weights aims for ({@code -Dbrainiac.hip.packedBlocks}; 0: four per compute unit). */
    private static final int PACKED_BLOCKS = Integer.getInteger("brainiac.hip.packedBlocks", 0);
    /** The fewest rows of k a piece of such a product takes. */
    private static final int PACKED_PIECE = Integer.getInteger("brainiac.hip.packedPiece", 64);

    @Override
    public synchronized void matmulPacked(Storage a, PackedStorage b, int m, int k, int n, Storage out) {
        if (b instanceof QuantizedRows || m == 0 || k == 0) {
            Backend.super.matmulPacked(a, b, m, k, n, out);
            return;
        }
        HipStorage z = wr(out);
        HipContext c = ctx();
        if (m > PACKED_ROWS) {
            // many rows: unpacked into a temporary for rocBLAS, which is then free for the next
            HipStorage w = raw((long) k * n, z.dtype, null);
            unpack(b, w);
            matmulBatched(a, w, 1, m, k, n, z);
            w.release();
            return;
        }
        Packed p = packed(b);
        HipStorage x = as(cs(a), z.dtype);
        matvec(c, "packed_mv", x, 0, p, m, k, n, z);
        done(x, cs(a));
    }

    /** Whether products of up to 8 rows by a dense matrix use {@code dense_mv} ({@code -Dbrainiac.hip.denseMv=false}: rocBLAS or few_dense). */
    private static final boolean DENSE_MV = !"false".equals(System.getProperty("brainiac.hip.denseMv"));

    /**
     * {@code z[m, n] = x[m, k] x w[k, n]} for {@code m <= PACKED_ROWS} by {@code packed_mv} (w
     * packed) or {@code dense_mv} (w dense, at address {@code w}).
     */
    private void matvec(HipContext c, String kernel, HipStorage x, long w, Packed p, int m, int k, int n,
            HipStorage z) {
        // 128 columns per block; enough pieces of k to fill the device, of at least PACKED_PIECE rows
        int columnBlocks = (n + 127) / 128;
        int target = PACKED_BLOCKS > 0 ? PACKED_BLOCKS : 4 * c.multiprocessors;
        int pieces = Math.max(1, Math.min((k + PACKED_PIECE - 1) / PACKED_PIECE, target / columnBlocks));
        if (columnBlocks > FEW_COUNTERS) {
            pieces = 1;
        }
        int chunk = ((k + pieces - 1) / pieces + 1) / 2 * 2;
        pieces = (k + chunk - 1) / chunk;
        HipStorage part = pieces > 1 ? raw((long) pieces * m * n, DType.FLOAT32, null) : null;
        if (part != null && fewCounters == null) {
            fewCounters = raw(FEW_COUNTERS, DType.FLOAT32, null);
            fillNow(fewCounters, 0);
        }
        int rows = m == 1 ? 1 : m == 2 ? 2 : m <= 4 ? 4 : 8;
        c.args().ptr(x.address());
        if (p != null) {
            p.push(c);
            c.ptr(part != null ? part.address() : 0).ptr(z.address()).i32(m).i32(k).i32(chunk);
        } else {
            c.ptr(w).ptr(part != null ? part.address() : 0).ptr(z.address()).i32(m).i32(k).i32(n).i32(chunk);
        }
        c.ptr(part != null ? fewCounters.address() : 0);
        c.launch(kernel(kernel + rows, z.dtype), columnBlocks, pieces, 1);
        if (part != null) {
            part.release();
        }
    }

    // ------------------------------------------------------------------ attention

    /** Whether attention runs fused ({@code -Dbrainiac.hip.fusedAttention=false}: never). */
    private static final boolean FUSED_ATTENTION = !"false".equals(System.getProperty("brainiac.hip.fusedAttention"));
    /** The fewest keys a piece of a split attention row takes ({@code -Dbrainiac.hip.attentionPiece}). */
    private static final int ATTENTION_PIECE = Math.max(1, Integer.getInteger("brainiac.hip.attentionPiece", 16));
    /** Warps per block of the attention kernel ({@code -Dbrainiac.hip.attentionBlock}). */
    private static final int ATTENTION_BLOCK_WARPS = Math.clamp(Integer.getInteger("brainiac.hip.attentionBlock", 2), 1, 8);
    /** Warps per compute unit a split attention aims for ({@code -Dbrainiac.hip.attentionWarps}). */
    private static final int ATTENTION_WARPS = Math.max(1, Integer.getInteger("brainiac.hip.attentionWarps", 64));
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
        HipStorage z = wr(out);
        DType t = z.dtype;
        checkAttention(shape, t);
        HipStorage l = wr(lse);
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
        long target = (long) ATTENTION_WARPS * c.multiprocessors;
        int parts = (int) Math.max(1, Math.min((tk + ATTENTION_PIECE - 1) / ATTENTION_PIECE, (target + rows - 1) / rows));
        int chunk = Math.max(1, (tk + parts - 1) / parts);
        parts = Math.max(1, (tk + chunk - 1) / chunk);
        HipStorage part = parts > 1 ? raw(rows * parts * (shape[6] + 2), DType.FLOAT32, null) : null;
        // the pieces are combined by each row's last warp to finish, which counts them
        boolean counted = part != null && rows <= FEW_COUNTERS;
        long counters = counted ? counters().address() : 0;
        c.args().ptr(xq.address()).ptr(xk.address()).ptr(xv.address()).ptr(z.address()).ptr(l.address())
                .ptr(xb == null ? 0 : xb.address()).ptr(part == null ? 0 : part.address())
                .ints(attShape(shape, scale, causal)).i64(rows).i32(xb == null ? 0 : 1).i64(strides[0])
                .i64(strides[1]).i64(strides[2]).i64(strides[3]).i32(parts).i32(chunk)
                .ptr(counters);
        // blocks of few warps: decoding has only a hundred or so warps, which in blocks of 8
        // ran on a few compute units of the GPU's dozens
        long warps = rows * parts;
        c.launch(kernel("attention", t), (int) Math.max(1, (warps + ATTENTION_BLOCK_WARPS - 1) / ATTENTION_BLOCK_WARPS),
                1, 1, 32 * ATTENTION_BLOCK_WARPS);
        if (counted) {
            part.release();
        } else if (part != null) {
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
        HipStorage z = wr(out);
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
        HipStorage gq = wr(dq);
        HipStorage gk = wr(dk);
        HipStorage gv = wr(dv);
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
        HipStorage y = wr(out);
        HipContext c = ctx();
        HipStorage xs = as(cs(x), y.dtype);
        HipStorage ws = as(cs(weight), y.dtype);
        c.args().ptr(xs.address()).ptr(ws.address()).ptr(y.address()).i64(rows).i32(cols).f32(eps);
        c.launch(kernel("rms_norm", y.dtype), (int) Math.min(rows, 1 << 20), 1, 1);
        done(xs, cs(x));
        done(ws, cs(weight));
    }

    @Override
    public boolean hasAddRmsNorm(DType type) {
        return hasRmsNorm(type);
    }

    @Override
    public synchronized void addRmsNorm(Storage x, Storage r, Storage weight, long rows, int cols, float eps,
            Storage sum, Storage out) {
        HipStorage y = wr(out);
        HipStorage s = wr(sum);
        if (!hasAddRmsNorm(y.dtype) || s.dtype != y.dtype) {
            throw new UnsupportedOperationException("fused add and RMSNorm: " + y.dtype);
        }
        HipContext c = ctx();
        HipStorage xs = as(cs(x), y.dtype);
        HipStorage rs = as(cs(r), y.dtype);
        HipStorage ws = as(cs(weight), y.dtype);
        c.args().ptr(xs.address()).ptr(rs.address()).ptr(ws.address()).ptr(s.address()).ptr(y.address()).i64(rows)
                .i32(cols).f32(eps);
        c.launch(kernel("add_rms_norm", y.dtype), (int) Math.min(rows, 1 << 20), 1, 1);
        done(xs, cs(x));
        done(rs, cs(r));
        done(ws, cs(weight));
    }

    // ------------------------------------------------------------------ windows and indices

    @Override
    public synchronized void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        HipStorage z = wr(out);
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
        HipStorage z = wr(out);
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
        HipStorage z = wr(out);
        HipContext c = ctx();
        long bound = cs(indices).indexBound;
        if (bound <= 0 || bound > length) {
            indices(cs(indices), count, length);   // else made on the device in range: no download
        }
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
        HipStorage z = wr(out);
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
