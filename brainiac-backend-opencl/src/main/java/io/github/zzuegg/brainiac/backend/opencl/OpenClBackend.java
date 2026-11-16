package io.github.zzuegg.brainiac.backend.opencl;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Runs tensor operations on a GPU through OpenCL, called directly with the Foreign
 * Function and Memory API: no native code to build and no libraries beyond the OpenCL
 * driver that comes with the graphics card.
 *
 * <p>Tensors live in device memory; only {@code toArray()}, {@code item()} and similar
 * calls copy values back. All four number formats work: float64 on devices with
 * {@code cl_khr_fp64}, float16 and bfloat16 stored in 16 bits and computed in float32.
 *
 * <p>The backend is available when an OpenCL GPU (or accelerator) is found; otherwise
 * {@link #isAvailable()} returns false and {@link Device#best()} picks the CPU. System
 * properties:
 * <ul>
 *   <li>{@code brainiac.opencl.devices}: {@code gpu} (default), {@code cpu} or {@code all}
 *       device types to consider, so OpenCL CPU drivers such as PoCL can be used too</li>
 *   <li>{@code brainiac.opencl.device}: the index among those devices; by default the one
 *       with dedicated memory and the most compute units (a graphics card before integrated
 *       graphics)</li>
 *   <li>{@code brainiac.opencl.memory}: the device memory to use, in MB; by default 80%
 *       of the device's memory. Unreachable tensors are freed by the garbage collector,
 *       which is triggered when this budget runs out.</li>
 *   <li>{@code brainiac.opencl.library}: the path of the OpenCL library, if not found</li>
 * </ul>
 *
 * <p>Java needs permission to call native code: run with
 * {@code --enable-native-access=io.github.zzuegg.brainiac.backend.opencl} (or
 * {@code ALL-UNNAMED} on the class path) to avoid a warning.
 */
public final class OpenClBackend implements Backend {

    private static final Cleaner CLEANER = Cleaner.create();
    private static final int MAX_RANK = 16;

    private CL cl;
    private MemorySegment device;
    private MemorySegment context;
    private MemorySegment queue;
    private String deviceName = "";
    private String deviceList = "";
    private String unavailable;
    private boolean initialized;
    private boolean fp64;
    /** Identifies the device and driver in the program cache. */
    private String fingerprint = "";
    private int tile;
    /** Work items sharing one output of a long reduction: a power of two the device allows. */
    private int reduceGroup = 64;
    private long budget;
    private final AtomicLong allocated = new AtomicLong();
    private final Pool pool = new Pool();
    private final Map<MetaKey, MemorySegment> metas = new LinkedHashMap<>(64, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<MetaKey, MemorySegment> eldest) {
            if (size() > 1024) {
                cl.release(eldest.getValue());   // freed by OpenCL once queued kernels are done
                return true;
            }
            return false;
        }
    };

    /** Shape and stride tables, cached on the device because training repeats them. */
    private record MetaKey(long[] values) {
        @Override
        public boolean equals(Object o) {
            return o instanceof MetaKey k && Arrays.equals(values, k.values);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(values);
        }
    }

    /**
     * Buffers of unreachable tensors, kept for reuse by size: creating device buffers is slow
     * on discrete GPUs. Reuse is safe because the command queue runs in order, so kernels
     * still reading a recycled buffer finish before any later kernel writes it.
     */
    static final class Pool {
        private final Map<Long, java.util.concurrent.ConcurrentLinkedQueue<MemorySegment>> free =
                new java.util.concurrent.ConcurrentHashMap<>();
        private final AtomicLong bytes = new AtomicLong();

        void put(long size, MemorySegment mem) {
            free.computeIfAbsent(size, k -> new java.util.concurrent.ConcurrentLinkedQueue<>()).add(mem);
            bytes.addAndGet(size);
        }

        MemorySegment take(long size) {
            java.util.concurrent.ConcurrentLinkedQueue<MemorySegment> q = free.get(size);
            MemorySegment m = q == null ? null : q.poll();
            if (m != null) {
                bytes.addAndGet(-size);
            }
            return m;
        }

        long bytes() {
            return bytes.get();
        }

        /** Releases every pooled buffer; returns the bytes freed. */
        long drain(CL cl) {
            long freed = 0;
            for (Map.Entry<Long, java.util.concurrent.ConcurrentLinkedQueue<MemorySegment>> e : free.entrySet()) {
                MemorySegment m;
                while ((m = e.getValue().poll()) != null) {
                    cl.release(m);
                    freed += e.getKey();
                    bytes.addAndGet(-e.getKey());
                }
            }
            return freed;
        }
    }
    private final Map<DType, Program> programs = new EnumMap<>(DType.class);
    private final Map<String, Tables> tables = new LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, Tables> eldest) {
            if (size() > 32) {
                return true;   // freed once no pending unrolled image uses it
            }
            return false;
        }
    };

    private record Program(MemorySegment program, Map<String, MemorySegment> kernels) {
    }

    /** Index tables for one convolution geometry, kept on the device. */
    private record Tables(MemorySegment im2col, MemorySegment col2imOffsets, MemorySegment col2imEntries) {
    }

    /** Frees a geometry's tables once neither the cache nor a pending expression holds them. */
    private record TablesRelease(CL cl, MemorySegment im2col, MemorySegment col2imOffsets,
            MemorySegment col2imEntries) implements Runnable {
        @Override
        public void run() {
            cl.release(im2col);
            cl.release(col2imOffsets);
            cl.release(col2imEntries);
        }
    }

    /** Creates the backend; called by {@link java.util.ServiceLoader}. */
    public OpenClBackend() {
    }

    @Override
    public String name() {
        return deviceName.isEmpty() ? "opencl" : "opencl (" + deviceName + ")";
    }

    @Override
    public Device.Kind kind() {
        return Device.Kind.GPU;
    }

    @Override
    public synchronized boolean isAvailable() {
        if (!initialized) {
            initialized = true;
            try {
                init();
            } catch (RuntimeException | LinkageError e) {
                unavailable = e.getMessage() == null ? e.toString() : e.getMessage();
                cl = null;
            }
        }
        return cl != null;
    }

    /**
     * Describes the device in use, or why none is.
     *
     * @return for example {@code "NVIDIA GeForce RTX 4070, 12 GB, float64"} or the reason
     *     OpenCL is unavailable
     */
    public synchronized String status() {
        if (!isAvailable()) {
            return "unavailable: " + unavailable;
        }
        return deviceName + ", " + (budget >> 20) + " MB budget" + (fp64 ? ", float64" : ", no float64")
                + "; devices " + deviceList;
    }

    private void init() {
        CL lib = CL.load();
        String types = System.getProperty("brainiac.opencl.devices", "gpu").toLowerCase(Locale.ROOT);
        long mask = switch (types) {
            case "gpu" -> CL.DEVICE_TYPE_GPU | CL.DEVICE_TYPE_ACCELERATOR;
            case "cpu" -> CL.DEVICE_TYPE_CPU;
            case "all" -> CL.DEVICE_TYPE_ALL;
            default -> throw new IllegalStateException("brainiac.opencl.devices must be gpu, cpu or all, not " + types);
        };
        List<MemorySegment> found = new ArrayList<>();
        for (MemorySegment platform : lib.platforms()) {
            found.addAll(lib.devices(platform, mask));
        }
        if (found.isEmpty()) {
            throw new IllegalStateException("no OpenCL " + types + " device found"
                    + (types.equals("gpu") ? " (set -Dbrainiac.opencl.devices=all to use OpenCL CPU drivers)" : ""));
        }
        Integer chosen = Integer.getInteger("brainiac.opencl.device");
        int index = chosen != null ? chosen : best(lib, found);
        if (index < 0 || index >= found.size()) {
            throw new IllegalStateException("brainiac.opencl.device=" + index + " but only " + found.size()
                    + " devices were found");
        }
        StringBuilder names = new StringBuilder();
        for (int i = 0; i < found.size(); i++) {
            names.append(i == 0 ? "" : ", ").append(i).append(": ").append(lib.deviceString(found.get(i),
                    CL.DEVICE_NAME)).append(i == index ? " (in use)" : "");
        }
        deviceList = names.toString();
        device = found.get(index);
        deviceName = lib.deviceString(device, CL.DEVICE_NAME);
        fp64 = (" " + lib.deviceString(device, CL.DEVICE_EXTENSIONS) + " ").contains(" cl_khr_fp64 ");
        long maxGroup = lib.deviceLong(device, CL.DEVICE_MAX_WORK_GROUP_SIZE);
        tile = maxGroup >= 256 ? 16 : 8;
        reduceGroup = (int) Math.min(128, Long.highestOneBit(Math.max(1, maxGroup)));
        long memory = lib.deviceLong(device, CL.DEVICE_GLOBAL_MEM_SIZE);
        Integer mb = Integer.getInteger("brainiac.opencl.memory");
        budget = mb != null ? (long) mb << 20 : memory / 10 * 8;
        fingerprint = lib.deviceString(device, CL.DEVICE_NAME) + "|" + lib.deviceString(device, CL.DEVICE_VERSION)
                + "|" + lib.deviceString(device, CL.DRIVER_VERSION) + "|" + Runtime.version().feature();
        context = lib.createContext(device);
        queue = lib.createQueue(context, device, profiling);
        cl = lib;
        program(DType.FLOAT32);
        if (profiling) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                if (!kernelTimes.isEmpty() || !launched.isEmpty()) {
                    System.err.print(profile());
                }
            }, "opencl-profile"));
        }
    }

    /**
     * Picks the fastest-looking device: dedicated memory before memory shared with the host
     * (integrated graphics), then the most compute units times clock speed, then the most
     * memory.
     */
    private static int best(CL lib, List<MemorySegment> devices) {
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < devices.size(); i++) {
            MemorySegment d = devices.get(i);
            double score;
            try {
                boolean shared = (lib.deviceLong(d, CL.DEVICE_HOST_UNIFIED_MEMORY) & 0xFFFFFFFFL) != 0;
                boolean gpu = (lib.deviceLong(d, CL.DEVICE_TYPE) & CL.DEVICE_TYPE_GPU) != 0;
                double compute = (double) (lib.deviceLong(d, CL.DEVICE_MAX_COMPUTE_UNITS) & 0xFFFFFFFFL)
                        * Math.max(1, lib.deviceLong(d, CL.DEVICE_MAX_CLOCK_FREQUENCY) & 0xFFFFFFFFL);
                double memory = lib.deviceLong(d, CL.DEVICE_GLOBAL_MEM_SIZE);
                score = (shared ? 0 : 4e18) + (gpu ? 2e18 : 0) + compute * 1e6 + memory / 1e12;
            } catch (RuntimeException e) {
                score = -i;
            }
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------- programs

    private Program program(DType type) {
        Program p = programs.get(type);
        if (p != null) {
            return p;
        }
        if (!supports(type)) {
            throw new UnsupportedOperationException(deviceName + " does not support " + type
                    + " (the device lacks cl_khr_fp64)");
        }
        String source = prelude(type) + kernelSource();
        MemorySegment program = build(source);
        Map<String, MemorySegment> kernels = new HashMap<>();
        for (String name : List.of("fill", "unary", "activation", "activation_backward", "binary_contiguous",
                "binary_strided", "copy_strided", "reduce_rows", "reduce_group", "argmax", "matmul", "im2col",
                "col2im", "gather", "scatter_add", "to_float", "from_float", "random")) {
            kernels.put(name, cl.createKernel(program, name));
        }
        p = new Program(program, kernels);
        programs.put(type, p);
        return p;
    }

    private String prelude(DType type) {
        // No contraction into fused multiply-adds anywhere, so generated kernels and the
        // precompiled ones round every step alike.
        String common = "#pragma OPENCL FP_CONTRACT OFF\n#define TS " + tile + "\n#define RG " + reduceGroup + "\n";
        return common + switch (type) {
            case FLOAT32 -> """
                    #define T float
                    #define RAW float
                    #define ACC float
                    #define LOAD(p, i) ((p)[i])
                    #define STORE(p, i, v) ((p)[i] = (v))
                    """;
            case FLOAT64 -> """
                    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
                    #define T double
                    #define RAW double
                    #define ACC double
                    #define LOAD(p, i) ((p)[i])
                    #define STORE(p, i, v) ((p)[i] = (v))
                    """;
            case FLOAT16 -> """
                    #define T half
                    #define RAW ushort
                    #define ACC float
                    #define LOAD(p, i) vload_half((i), (p))
                    #define STORE(p, i, v) vstore_half_rte((v), (i), (p))
                    """;
            case BFLOAT16 -> """
                    #define T ushort
                    #define RAW ushort
                    #define ACC float
                    ushort to_bf16(float v) {
                        uint b = as_uint(v);
                        if (isnan(v)) return (ushort) ((b >> 16) | 0x40);
                        b += 0x7FFF + ((b >> 16) & 1);
                        return (ushort) (b >> 16);
                    }
                    #define LOAD(p, i) as_float(((uint) (p)[i]) << 16)
                    #define STORE(p, i, v) ((p)[i] = to_bf16(v))
                    """;
        };
    }

    private static String kernelSource() {
        try (InputStream in = OpenClBackend.class.getResourceAsStream("kernels.cl")) {
            if (in == null) {
                throw new IllegalStateException("kernels.cl is missing from the OpenCL backend");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** A temporary device buffer argument, released by the caller. */
    private record Temp(MemorySegment mem) {
    }

    private void run(DType type, String kernel, long[] global, long[] local, Object... args) {
        for (long g : global) {
            if (g == 0) {
                return;
            }
        }
        launch(kernel, program(type).kernels().get(kernel), global, local, args);
        generation++;
    }

    /** The arguments each kernel was last launched with: buffer addresses and values. */
    private final Map<Long, Object[]> kernelArgs = new HashMap<>();

    private void launch(String name, MemorySegment k, long[] global, long[] local, Object... args) {
        for (long g : global) {
            if (g == 0) {
                return;
            }
        }
        for (Object a : args) {
            if (a instanceof ClStorage s) {
                ready(s);
                s.mem();   // created (or taken from the pool) outside the timed enqueue
            }
        }
        long hostStart = profiling ? System.nanoTime() : 0;
        // a kernel keeps its arguments between launches: set only those that changed
        Object[] last = kernelArgs.computeIfAbsent(k.address(), a -> new Object[args.length]);
        if (last.length < args.length) {
            last = Arrays.copyOf(last, args.length);
            kernelArgs.put(k.address(), last);
        }
        Arena arena = null;
        try {
            for (int i = 0; i < args.length; i++) {
                Object key = switch (args[i]) {
                    case ClStorage s -> s.mem().address();
                    case Temp t -> t.mem().address();
                    case MemorySegment m -> m.address();
                    default -> args[i];
                };
                if (key.equals(last[i])) {
                    continue;
                }
                if (arena == null) {
                    arena = Arena.ofConfined();
                }
                MemorySegment value = switch (args[i]) {
                    case ClStorage s -> handle(arena, s.mem());
                    case Temp t -> handle(arena, t.mem());
                    case MemorySegment m -> handle(arena, m);
                    case Integer v -> arena.allocateFrom(JAVA_INT, v);
                    case Long v -> arena.allocateFrom(JAVA_LONG, v);
                    case Float v -> arena.allocateFrom(JAVA_FLOAT, v);
                    default -> throw new IllegalArgumentException("unsupported kernel argument " + args[i]);
                };
                last[i] = null;   // unknown if setting it fails
                cl.setArg(k, i, value);
                last[i] = key;
            }
        } finally {
            if (arena != null) {
                arena.close();
            }
        }
        MemorySegment event = cl.run(queue, k, global, local, profiling);
        if (flushEvery != 0 && ++sinceFlush >= flushInterval) {
            // some drivers hold commands back until a flush, leaving the device idle meanwhile
            cl.flush(queue);
            sinceFlush = 0;
            if (flushEvery < 0) {
                flushInterval = Math.min(flushInterval * 2, 64);
            }
        }
        if (profiling) {
            hostLaunchNanos += System.nanoTime() - hostStart;
            launched.add(new Launch(name, event));
            if (launched.size() >= 4_096) {
                cl.finish(queue);
                settled();
            }
        }
    }

    // ---------------------------------------------------------------- fusion

    /** Whether element-wise chains are fused ({@code -Dbrainiac.opencl.fusion=false} turns it off). */
    private final boolean fusion = !"false".equalsIgnoreCase(System.getProperty("brainiac.opencl.fusion"));
    private final Map<String, Future<MemorySegment>> fusedKernels = new HashMap<>();
    /**
     * Whether fused kernels also store the pending inputs they compute, so chains can span
     * other kernels ({@code -Dbrainiac.opencl.fusion.store=true}). Off by default: on AMD's
     * driver, kernels writing several buffers cost more than the launches they save.
     */
    private final boolean storeIntermediates = Boolean.getBoolean("brainiac.opencl.fusion.store");
    /** Compiles fused kernels in the background ({@code -Dbrainiac.opencl.fusion.async=false} compiles inline). */
    private final ExecutorService compiler = "false".equalsIgnoreCase(
            System.getProperty("brainiac.opencl.fusion.async")) ? null
            : Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "brainiac-opencl-compiler");
                t.setDaemon(true);
                return t;
            });
    private long eagerFallbacks;
    private String helpers;
    /**
     * Counts launches of other kernels. A pending result recorded before the latest of them
     * is computed before anything builds on it, so chains end at the same points in every
     * training step and their generated kernels repeat instead of growing.
     */
    private long generation;   // also advanced when the host waits for the device
    private int fusedCompiled;
    private int fusedFailed;
    private long fusedCompileNanos;

    /** Whether an element-wise result can be recorded instead of computed now. */
    private boolean fusable(ClStorage out, Storage... inputs) {
        if (!fusion || (out.dtype != DType.FLOAT32 && out.dtype != DType.FLOAT64) || out.size == 0) {
            return false;
        }
        for (Storage in : inputs) {
            if (cl(in).dtype != out.dtype) {
                return false;
            }
        }
        return true;
    }

    /** Records {@code e} as the pending value of {@code out}, computing large inputs first. */
    private void defer(ClStorage out, Fused.Expr e) {
        program(out.dtype);
        for (Fused.Src in : Fused.sources(e)) {
            // a recorded product is a single step, so carrying it over cannot grow a chain
            if ((!in.direct() || !storeIntermediates) && in.storage().pending != null
                    && !(in.direct() && in.storage().pending instanceof Fused.MatMul)
                    && !(in.storage().pending instanceof Fused.Const)   // nothing to grow
                    && !(in.storage().pending instanceof Fused.Random)
                    // an unrolled image is computed where a matmul loads it, never stored
                    && !(in.storage().pending instanceof Fused.Im2col && in.storage().pendingNodes <= 8)
                    && in.storage().pendingGeneration != generation) {
                ready(in.storage());
            } else if (!in.direct() && in.storage().pending != null && Fused.scatters(in.storage().pending)) {
                // read at other indices (a broadcast, a permutation): summed once, not at every read
                ready(in.storage());
            }
        }
        int n = Fused.nodes(e);
        if (n > Fused.MAX_NODES) {
            for (ClStorage in : Fused.inputs(e)) {
                ready(in);
            }
            n = Fused.nodes(e);
        }
        out.pending = e;
        out.pendingNodes = n;
        out.pendingGeneration = generation;
    }

    /** Computes a pending storage, so that its buffer holds its values. */
    private void ready(ClStorage s) {
        if (s.pending instanceof Fused.MatMul) {
            runMatMul(s);
        } else if (s.pending instanceof Fused.Reduce) {
            runReduces(s);
        } else if (s.pending != null) {
            materialize(s);
        }
    }

    /** Runs a precompiled kernel with explicit work sizes, without starting a new generation. */
    private void quietRun(DType type, String kernel, long[] global, long[] local, Object... args) {
        launch(kernel, program(type).kernels().get(kernel), global, local, args);
    }

    /**
     * Turns {@code s} (a chain on exactly one matrix product, read element for element) into
     * one matmul kernel with the chain as epilogue; false if the chain does not fit or its
     * kernel is still compiling, leaving everything as it was.
     */
    private boolean epilogue(ClStorage s, ClStorage base) {
        Fused.MatMul mm = (Fused.MatMul) base.pending;
        if (base.size != s.size) {
            return false;
        }
        Fused.Expr pa = prologue(mm.a());
        Fused.Expr pb = prologue(mm.b());
        Fused.Kernel k = Fused.generateMatMul(s.pending, base, storeIntermediates, pa, layout(mm, pa, true), pb,
                layout(mm, pb, false));
        if (k.argBytes() > Fused.MAX_ARG_BYTES) {
            return false;
        }
        MemorySegment kernel = fusedKernel(s.dtype, k.source());
        if (kernel == null) {
            eagerFallbacks++;
            return false;
        }
        s.pending = null;
        s.pendingNodes = 0;
        base.pending = null;
        base.pendingNodes = 0;
        for (ClStorage w : k.outputs()) {
            w.pending = null;
            w.pendingNodes = 0;
        }
        launch("fused_matmul", kernel, mm.global(), mm.local(), matmulArgs(s, mm, k, pa != null, pb != null));
        return true;
    }

    /** The arguments of a kernel from {@link Fused#generateMatMul}. */
    private Object[] matmulArgs(ClStorage out, Fused.MatMul mm, Fused.Kernel k, boolean computeA, boolean computeB) {
        Object[] args = new Object[2 + mm.args().length + k.args().size()];
        args[0] = out;
        args[1] = out.size;
        System.arraycopy(mm.args(), 0, args, 2, mm.args().length);
        if (computeA) {
            args[2] = new Temp(unused());   // computed by the kernel: no buffer needed
        }
        if (computeB) {
            args[3] = new Temp(unused());
        }
        for (int i = 0; i < k.args().size(); i++) {
            args[2 + mm.args().length + i] = k.args().get(i);
        }
        return args;
    }

    private MemorySegment unused;

    /** A small buffer passed for an operand a generated matmul computes itself. */
    private MemorySegment unused() {
        if (unused == null) {
            unused = createBuffer(CL.MEM_READ_WRITE, 256, MemorySegment.NULL);
        }
        return unused;
    }

    /**
     * How a matmul reads an operand that is an unrolled image: 1 when its rows and columns are
     * the image's, 2 when transposed, 0 otherwise (or when it is not one).
     */
    private static int layout(Fused.MatMul mm, Fused.Expr e, boolean first) {
        if (!(e instanceof Fused.Im2col c) || mm.global()[2] != 1) {
            return 0;
        }
        Object[] a = mm.args();
        int base = first ? 9 : 12;
        long width = (long) c.channels() * c.ks();
        long offset = (Long) a[base];
        long row = (Long) a[base + 1];
        long col = (Long) a[base + 2];
        if (offset != 0) {
            return 0;
        }
        // rows of the operand: M for the first, K for the second
        int rows = first ? (Integer) a[3] : (Integer) a[4];
        int cols = first ? (Integer) a[4] : (Integer) a[5];
        if ((row == width || rows == 1) && (col == 1 || cols == 1) && cols == width) {
            return 1;
        }
        if ((row == 1 || rows == 1) && (col == width || cols == 1) && rows == width) {
            return 2;
        }
        return 0;
    }

    /** Whether a matmul operand is left pending for the matmul kernel to compute as it loads it. */
    private boolean computedInMatMul(ClStorage s) {
        return fusion && s.pending != null && !(s.pending instanceof Fused.MatMul)
                && !(s.pending instanceof Fused.Reduce) && s.pendingNodes <= 16 && Fused.unrolls(s.pending);
    }

    /** The value a matmul computes for an operand left pending, with what it reads ready; null if computed. */
    private Fused.Expr prologue(ClStorage s) {
        if (s == null || s.pending == null || s.pending instanceof Fused.MatMul || s.pending instanceof Fused.Reduce) {
            return null;
        }
        readyReduces(s.pending);
        for (ClStorage p : Fused.matmuls(s.pending).keySet()) {
            ready(p);
        }
        return s.pending;
    }

    private void materialize(ClStorage s) {
        readyReduces(s.pending);
        Map<ClStorage, Boolean> products = Fused.matmuls(s.pending);
        if (products.size() == 1) {
            Map.Entry<ClStorage, Boolean> only = products.entrySet().iterator().next();
            if (only.getValue() && epilogue(s, only.getKey())) {
                return;
            }
        }
        for (ClStorage p : products.keySet()) {
            ready(p);
        }
        long genStart = profiling ? System.nanoTime() : 0;
        Fused.Kernel k = Fused.generate(s.pending, storeIntermediates);
        if (profiling) {
            hostGenerateNanos += System.nanoTime() - genStart;
        }
        if (k.argBytes() > Fused.MAX_ARG_BYTES) {
            for (ClStorage in : Fused.inputs(s.pending)) {
                ready(in);
            }
            k = Fused.generate(s.pending, storeIntermediates);
        }
        Fused.Expr e = s.pending;
        s.pending = null;
        s.pendingNodes = 0;
        MemorySegment kernel = fusedKernel(s.dtype, k.source());
        if (kernel == null) {
            // the kernel is still compiling in the background; run the steps one by one meanwhile
            eager(e, s, storeIntermediates);
            eagerFallbacks++;
            return;
        }
        for (ClStorage w : k.outputs()) {
            w.pending = null;
            w.pendingNodes = 0;
        }
        Object[] args = new Object[k.args().size() + 2];
        args[0] = s;
        args[1] = s.size;
        for (int i = 0; i < k.args().size(); i++) {
            args[i + 2] = k.args().get(i);
        }
        launch("fused", kernel, new long[] {s.size}, null, args);
    }

    @Override
    public synchronized void evaluate(List<Storage> storages) {
        if (!fusion) {
            return;
        }
        Set<ClStorage> keep = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Storage st : storages) {
            ClStorage s = cl(st);
            if (s.pending instanceof Fused.MatMul) {
                continue;   // one launch whenever it runs; later, a chain on it can be its epilogue
            } else if (s.pending instanceof Fused.Reduce) {
                runReduces(s);   // with the other requested reductions of its shape
            } else if (s.pending != null) {
                readyReduces(s.pending);
                Map<ClStorage, Boolean> products = Fused.matmuls(s.pending);
                if (products.size() == 1) {
                    Map.Entry<ClStorage, Boolean> only = products.entrySet().iterator().next();
                    if (only.getValue() && epilogue(s, only.getKey())) {
                        continue;   // a chain on one product: it becomes that matmul's epilogue
                    }
                }
                for (ClStorage p : products.keySet()) {
                    ready(p);
                }
                keep.add(s);
            }
        }
        Map<DType, List<ClStorage>> byType = new EnumMap<>(DType.class);
        for (ClStorage s : keep) {
            byType.computeIfAbsent(s.dtype, t -> new ArrayList<>()).add(s);
        }
        for (Map.Entry<DType, List<ClStorage>> e : byType.entrySet()) {
            List<ClStorage> todo = e.getValue();
            // the largest chains first: the values they compute on the way need no range of their own
            todo.sort(Comparator.comparingInt((ClStorage s) -> -s.pendingNodes));
            while (!todo.isEmpty()) {
                Fused.Many k = Fused.generateMany(todo, keep, Fused.MAX_ARG_BYTES);
                List<ClStorage> group = List.copyOf(todo.subList(0, k.used()));
                if (k.kernel().argBytes() > Fused.MAX_ARG_BYTES || !evaluate(e.getKey(), k)) {
                    for (ClStorage s : group) {
                        ready(s);
                    }
                }
                todo.removeIf(s -> s.pending == null);
            }
        }
    }

    /** Runs a kernel from {@link Fused#generateMany}; false while it is still compiling. */
    private boolean evaluate(DType type, Fused.Many many) {
        Fused.Kernel k = many.kernel();
        MemorySegment kernel = fusedKernel(type, k.source());
        if (kernel == null) {
            // still compiling: compute what the kernel would write step by step, without
            // generating kernels for the parts, so the same chains are pending afterwards
            eagerFallbacks++;
            List<ClStorage> writes = new ArrayList<>(k.outputs());
            writes.add(many.first());
            for (ClStorage w : writes) {
                if (w.pending != null) {
                    Fused.Expr e = w.pending;
                    eager(e, w, false);
                    w.pending = null;
                    w.pendingNodes = 0;
                }
            }
            return true;
        }
        many.first().pending = null;
        many.first().pendingNodes = 0;
        for (ClStorage w : k.outputs()) {
            w.pending = null;
            w.pendingNodes = 0;
        }
        Object[] args = new Object[k.args().size() + 2];
        args[0] = many.first();
        args[1] = many.total();
        for (int i = 0; i < k.args().size(); i++) {
            args[i + 2] = k.args().get(i);
        }
        launch("fused_many", kernel, new long[] {many.total()}, null, args);
        return true;
    }

    /**
     * Where compiled programs are kept between runs ({@code -Dbrainiac.opencl.cache=<dir>}, or
     * {@code none} to always compile). The key covers the device, driver and source, so a
     * driver update or a changed kernel compiles anew.
     */
    private Path cacheDir() {
        String dir = System.getProperty("brainiac.opencl.cache");
        if (dir == null) {
            return Path.of(System.getProperty("user.home"), ".cache", "brainiac", "opencl");
        }
        return dir.isBlank() || dir.equalsIgnoreCase("none") ? null : Path.of(dir);
    }

    /** Builds a program, from the on-disk cache when an earlier run compiled the same source. */
    private MemorySegment build(String source) {
        Path dir = cacheDir();
        Path file = cacheFile(source);
        if (file != null) {
            try {
                if (Files.isRegularFile(file)) {
                    MemorySegment p = cl.buildProgramFromBinary(context, device, Files.readAllBytes(file), "");
                    if (p != null) {
                        return p;
                    }
                }
            } catch (IOException | RuntimeException e) {
                file = null;   // the cache is only an optimization
            }
        }
        MemorySegment program = cl.buildProgram(context, device, source, "");
        if (file != null) {
            byte[] binary = cl.programBinary(program);
            if (binary != null) {
                try {
                    Files.createDirectories(dir);
                    Path tmp = Files.createTempFile(dir, "kernel", ".tmp");
                    Files.write(tmp, binary);
                    Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
                } catch (IOException | RuntimeException e) {
                    // not cached; compiled again next run
                }
            }
        }
        return program;
    }

    /** The cache file for a program source, or null when caching is off. */
    private Path cacheFile(String source) {
        Path dir = cacheDir();
        if (dir == null) {
            return null;
        }
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            sha.update(fingerprint.getBytes(StandardCharsets.UTF_8));
            sha.update((byte) 0);
            sha.update(source.getBytes(StandardCharsets.UTF_8));
            return dir.resolve(HexFormat.of().formatHex(sha.digest()) + ".bin");
        } catch (NoSuchAlgorithmException e) {
            return null;
        }
    }

    /**
     * The compiled kernel for a generated source, or null while it is compiling in the
     * background or when the driver could not compile it.
     */
    private MemorySegment fusedKernel(DType type, String kernelSource) {
        String key = type + "\n" + kernelSource;
        Future<MemorySegment> compiled = fusedKernels.get(key);
        if (compiled == null) {
            if (helpers == null) {
                String all = kernelSource();
                helpers = all.substring(0, all.indexOf("__kernel"));
            }
            String source = prelude(type) + helpers + kernelSource;
            Callable<MemorySegment> build = () -> {
                long start = System.nanoTime();
                MemorySegment kernel;
                try {
                    kernel = cl.createKernel(build(source), "fused");
                } catch (RuntimeException first) {
                    try {
                        kernel = cl.createKernel(build(source), "fused");   // some drivers fail transiently
                    } catch (RuntimeException again) {
                        // the steps keep running one by one, so a driver bug costs speed, not the run
                        synchronized (fusedKernels) {
                            if (fusedFailed++ == 0) {
                                System.err.println("brainiac: the OpenCL driver failed to compile a generated "
                                        + "kernel; running its steps separately: " + again.getMessage());
                            }
                        }
                        return null;
                    }
                }
                synchronized (fusedKernels) {
                    fusedCompiled++;
                    fusedCompileNanos += System.nanoTime() - start;
                }
                return kernel;
            };
            Path cached = cacheFile(source);
            if (compiler != null && (cached == null || !Files.isRegularFile(cached))) {
                compiled = compiler.submit(build);
            } else {
                FutureTask<MemorySegment> now = new FutureTask<>(build);
                now.run();
                compiled = now;
            }
            fusedKernels.put(key, compiled);
        }
        if (!compiled.isDone()) {
            return null;
        }
        try {
            return compiled.get();
        } catch (ExecutionException ex) {
            throw ex.getCause() instanceof RuntimeException r ? r : new IllegalStateException(ex.getCause());
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while compiling a fused kernel", ex);
        }
    }

    /**
     * Computes {@code e} into {@code out} with the precompiled kernels, one per step, keeping
     * exactly the inputs the generated kernel would keep ({@code atRoot}: {@code e} is
     * evaluated at the result's own index).
     */
    private void eager(Fused.Expr e, ClStorage out, boolean atRoot) {
        DType t = out.dtype;
        long n = out.size;
        switch (e) {
            case Fused.MatMul mm -> throw new IllegalStateException("a matrix product inside a chain");
            case Fused.Reduce r -> throw new IllegalStateException("a reduction inside a chain");
            case Fused.Random x -> quiet(t, "random", n, out, x.seed(), n);
            case Fused.Col2im c -> quiet(t, "col2im", n, computed(c.cols().storage(), false), new Temp(c.offsets()),
                    new Temp(c.entries()), out, c.channels(), c.positions(), c.ks(), c.img(), n);
            case Fused.Im2col c -> quiet(t, "im2col", n, computed(c.x().storage(), false), new Temp(c.table()), out,
                    c.channels(), c.positions(), c.ks(), c.img(), c.pad(), n);
            case Fused.Const c -> quiet(t, "fill", n, out, c.value(), n);
            case Fused.Src src -> {
                ClStorage from = computed(src.storage(), atRoot && src.direct());
                quietCopy(from, src.offset(), src.strides(), src.dims(), out);
            }
            case Fused.Un u -> quiet(t, "unary", n, u.op(), value(u.x(), out, atRoot), out, n);
            case Fused.Act a -> quiet(t, "activation", n, a.op(), a.param(), value(a.x(), out, atRoot), out, n);
            case Fused.ActBack a -> quiet(t, "activation_backward", n, a.op(), a.param(),
                    value(a.x(), out, atRoot), value(a.grad(), out, atRoot), out, n);
            case Fused.Bin b -> {
                int[] shape = b.a() instanceof Fused.Src sa ? sa.dims()
                        : b.b() instanceof Fused.Src sb ? sb.dims() : new int[] {Math.toIntExact(n)};
                long[] dense = contiguous(shape);
                ClStorage x = operand(b.a(), out, atRoot);
                ClStorage y = operand(b.b(), out, atRoot);
                long[] xs = b.a() instanceof Fused.Src sa ? sa.strides() : dense;
                long[] ys = b.b() instanceof Fused.Src sb ? sb.strides() : dense;
                if (Arrays.equals(xs, dense) && Arrays.equals(ys, dense)) {
                    quiet(t, "binary_contiguous", n, b.op(), x, y, out, n);
                } else {
                    quiet(t, "binary_strided", n, b.op(), x, y, out, n, shape.length, meta(meta(shape, xs, ys)));
                }
            }
        }
    }

    /** A step's input as a buffer of the result's size, computing it if it is itself a step. */
    private ClStorage value(Fused.Expr e, ClStorage like, boolean atRoot) {
        if (e instanceof Fused.Src s && s.direct()) {
            return computed(s.storage(), atRoot);
        }
        ClStorage v = raw(like.size, like.dtype);
        eager(e, v, atRoot);
        return v;
    }

    /** A binary step's operand: a source as it is (read through its strides), anything else computed. */
    private ClStorage operand(Fused.Expr e, ClStorage like, boolean atRoot) {
        if (e instanceof Fused.Src s) {
            return computed(s.storage(), atRoot && s.direct());
        }
        return value(e, like, atRoot);
    }

    /**
     * The values of a storage for a step-by-step run. With {@code keep} it is computed into
     * its own buffer, as the generated kernel would; otherwise into a temporary buffer and it
     * stays pending, so later chains are recorded exactly as with compiled kernels.
     */
    private ClStorage computed(ClStorage s, boolean keep) {
        if (s.pending instanceof Fused.MatMul || s.pending instanceof Fused.Reduce) {
            ready(s);
        }
        if (s.pending == null) {
            return s;
        }
        if (keep) {
            Fused.Expr e = s.pending;
            s.pending = null;
            s.pendingNodes = 0;
            eager(e, s, true);
            return s;
        }
        ClStorage v = raw(s.size, s.dtype);
        eager(s.pending, v, false);
        return v;
    }

    private void quietCopy(ClStorage in, long offset, long[] strides, int[] dims, ClStorage out) {
        int[] shape = dims.length == 0 ? new int[] {1} : dims;
        long[] inStrides = dims.length == 0 ? new long[] {1} : strides;
        Temp meta = meta(meta(shape, inStrides, contiguous(shape)));
        quiet(out.dtype, "copy_strided", out.size, in, offset, out, 0L, out.size, shape.length, meta);
    }

    /** Runs a precompiled kernel without starting a new fusion generation. */
    private void quiet(DType type, String kernel, long n, Object... args) {
        launch(kernel, program(type).kernels().get(kernel), new long[] {n}, null, args);
    }

    private void run(DType type, String kernel, long n, Object... args) {
        run(type, kernel, new long[] {n}, null, args);
    }

    private static MemorySegment handle(Arena arena, MemorySegment mem) {
        MemorySegment h = arena.allocate(ADDRESS);
        h.set(ADDRESS, 0, mem);
        return h;
    }

    // ---------------------------------------------------------------- memory

    @Override
    public boolean supports(DType dtype) {
        return dtype != DType.FLOAT64 || fp64;
    }

    @Override
    public Storage allocate(long size) {
        return allocate(size, DType.FLOAT32);
    }

    @Override
    public synchronized Storage allocate(long size, DType dtype) {
        ClStorage s = raw(size, dtype);
        fill(s, 0f);
        return s;
    }

    @Override
    public synchronized Storage allocateUninitialized(long size, DType dtype) {
        return raw(size, dtype);
    }

    /** Allocates without clearing. */
    private long buffersCreated;
    private long buffersReused;
    private long bytesCreated;
    private long createdSinceCollect;
    private int collections;
    private long collectNanos;

    /**
     * The size a buffer of {@code bytes} is created with: rounded up to one of eight steps per
     * power of two, so tensors of nearby sizes (another batch size, another layer) share
     * pooled buffers, wasting at most an eighth.
     */
    static long bucket(long bytes) {
        long b = Math.max(256, bytes);
        long step = Long.highestOneBit(b) >> 3;
        return (b + step - 1) / step * step;
    }

    private ClStorage raw(long size, DType dtype) {
        program(dtype);
        ClStorage s = new ClStorage(this, new ClStorage.Buffer(), size, dtype);
        CLEANER.register(s, new Release(cl, s.buffer, bucket(size * dtype.bytes()), allocated, pool,
                budget / 4));
        return s;
    }

    @Override
    public synchronized void release(Storage storage) {
        ClStorage s = cl(storage);
        MemorySegment mem = s.buffer.mem;
        if (s.pending != null || mem == null || s.released) {
            return;   // pending results may still be inlined elsewhere: left to the collector
        }
        s.released = true;
        s.buffer.mem = null;   // the cleaner then has nothing to do
        long bytes = s.buffer.bytes;
        if (bytes == bucket(bytes) && pool.bytes() + bytes <= budget / 4) {
            pool.put(bytes, mem);   // reuse is safe: the queue runs in order
        } else {
            allocated.addAndGet(-bytes);
            cl.release(mem);
        }
    }

    /** A device buffer for a storage being written for the first time, from the pool if one fits. */
    MemorySegment buffer(long bytes) {
        MemorySegment mem = pool.take(bytes);
        if (mem == null) {
            mem = createBuffer(CL.MEM_READ_WRITE, bytes, MemorySegment.NULL);
            allocated.addAndGet(bytes);
            buffersCreated++;
            bytesCreated += bytes;
            createdSinceCollect += bytes;
            if (createdSinceCollect > Math.min(128L << 20, budget / 16)) {
                // Buffers return to the pool only once the JVM collects their tensors, which
                // can take long when little Java memory is allocated. Ask for it, rather
                // than filling the device with new buffers.
                createdSinceCollect = 0;
                collections++;
                long gcStart = System.nanoTime();
                System.gc();
                collectNanos += System.nanoTime() - gcStart;
            }
        } else {
            buffersReused++;
        }
        return mem;
    }

    /**
     * Returns a buffer to the pool once its storage is unreachable, or frees it when the
     * pool is full. Holds no reference to the storage.
     */
    private record Release(CL cl, ClStorage.Buffer buffer, long bytes, AtomicLong allocated, Pool pool,
            long poolLimit) implements Runnable {
        @Override
        public void run() {
            MemorySegment mem = buffer.mem;
            if (mem == null) {
                return;   // never written: no buffer was created
            }
            if (pool.bytes() + bytes <= poolLimit) {
                pool.put(bytes, mem);
            } else {
                allocated.addAndGet(-bytes);
                cl.release(mem);
            }
        }
    }

    private MemorySegment createBuffer(long flags, long bytes, MemorySegment host) {
        for (int attempt = 0; ; attempt++) {
            if (allocated.get() + bytes <= budget || attempt >= 4) {
                MemorySegment mem = cl.createBuffer(context, flags, bytes, host);
                if (mem != null) {
                    return mem;
                }
                if (attempt >= 4) {
                    throw new OutOfMemoryError("OpenCL device " + deviceName + " is out of memory allocating "
                            + bytes + " bytes (" + (allocated.get() >> 20) + " MB in use)");
                }
            }
            // Pooled buffers and unreachable tensors still hold device memory.
            allocated.addAndGet(-pool.drain(cl));
            if (allocated.get() + bytes <= budget) {
                continue;
            }
            System.gc();
            try {
                Thread.sleep(5L << attempt);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for device memory", e);
            }
        }
    }

    private Temp temp(MemorySegment hostData) {
        return new Temp(createBuffer(CL.MEM_READ_ONLY | CL.MEM_COPY_HOST_PTR, hostData.byteSize(), hostData));
    }

    private Temp ints(int[] values) {
        try (Arena arena = Arena.ofConfined()) {
            return temp(arena.allocateFrom(JAVA_INT, values.length == 0 ? new int[] {0} : values));
        }
    }

    /** A cached device copy of a shape and stride table; never released by the caller. */
    private Temp meta(long[] values) {
        MetaKey key = new MetaKey(values);
        MemorySegment m = metas.get(key);
        if (m == null) {
            m = longs(values).mem();
            metas.put(key, m);
        }
        return new Temp(m);
    }

    private Temp longs(long... values) {
        try (Arena arena = Arena.ofConfined()) {
            return temp(arena.allocateFrom(JAVA_LONG, values));
        }
    }

    private void release(Temp... temps) {
        for (Temp t : temps) {
            cl.release(t.mem());   // freed by OpenCL once the queued kernels are done
        }
    }

    /** Whether kernels are timed on the device ({@code -Dbrainiac.opencl.profile=true}). */
    private final boolean profiling = Boolean.getBoolean("brainiac.opencl.profile");
    private final List<Launch> launched = new ArrayList<>();
    private final Map<String, long[]> kernelTimes = new HashMap<>();
    private long queuedToStart;
    private long deviceIdle;
    private long hostIdle;
    private long firstStart;
    private long lastEnd;
    private long hostLaunchNanos;
    /**
     * Launches between explicit flushes ({@code -Dbrainiac.opencl.flushEvery}; 0 leaves
     * submission to the driver; -1 adapts: every launch right after the host waited for the
     * device, then doubling up to 64 while the host keeps queueing). The default of 4 trained
     * fastest on AMD's driver.
     */
    private final int flushEvery = Integer.getInteger("brainiac.opencl.flushEvery", 4);
    private int flushInterval = Math.max(1, flushEvery);
    private int sinceFlush;

    /** Called after the host waited for the device: the next launches start it again quickly. */
    private void synced() {
        if (flushEvery < 0) {
            flushInterval = 1;
            sinceFlush = 0;
        }
    }
    private long hostGenerateNanos;

    private record Launch(String kernel, MemorySegment event) {
    }

    /** Collects the timings of finished kernels; call only when the queue is idle. */
    private void settled() {
        for (Launch l : launched) {
            long[] t = kernelTimes.computeIfAbsent(l.kernel(), k -> new long[2]);
            long[] ts = cl.timestamps(l.event());
            t[0]++;
            t[1] += ts[3] - ts[2];
            queuedToStart += ts[2] - ts[0];
            if (lastEnd != 0 && ts[2] > lastEnd) {
                // idle until the kernel was queued is the host's time; after, the driver's
                long queued = Math.max(lastEnd, Math.min(ts[0], ts[2]));
                hostIdle += queued - lastEnd;
                deviceIdle += ts[2] - queued;
            }
            if (firstStart == 0) {
                firstStart = ts[2];
            }
            lastEnd = Math.max(lastEnd, ts[3]);
            cl.releaseEvent(l.event());
        }
        launched.clear();
    }

    /**
     * Device time per kernel since start, slowest first. Kernels are only timed when the JVM
     * runs with {@code -Dbrainiac.opencl.profile=true}; the report is then also printed at exit.
     *
     * @return one line per kernel: total milliseconds, launches and average microseconds
     */
    public synchronized String profile() {
        if (!profiling) {
            return "profiling is off (run with -Dbrainiac.opencl.profile=true)";
        }
        if (cl != null) {
            cl.finish(queue);
            settled();
        }
        long total = 0;
        for (long[] t : kernelTimes.values()) {
            total += t[1];
        }
        StringBuilder out = new StringBuilder(String.format(Locale.ROOT, "OpenCL kernel time %.1f ms%n", total / 1e6));
        if (fusedCompiled > 0) {
            out.append(String.format(Locale.ROOT,
                    "  %d fused kernels generated, compiled in %.1f ms; %d chains run unfused while compiling%n",
                    fusedCompiled, fusedCompileNanos / 1e6, eagerFallbacks));
        }
        if (fusedFailed > 0) {
            out.append(String.format(Locale.ROOT, "  %d generated kernels failed to compile%n", fusedFailed));
        }
        long launches = 0;
        for (long[] t : kernelTimes.values()) {
            launches += t[0];
        }
        if (launches > 0) {
            out.append(String.format(Locale.ROOT,
                    "  %d launches; device span %.1f ms, idle between kernels %.1f ms with work queued (driver) and"
                            + " %.1f ms waiting for the host; queued-to-start %.1f us avg%n",
                    launches, (lastEnd - firstStart) / 1e6, deviceIdle / 1e6, hostIdle / 1e6,
                    queuedToStart / 1e3 / launches));
            out.append(String.format(Locale.ROOT, "  host: %.1f ms enqueuing (%.1f us each), %.1f ms generating code%n",
                    hostLaunchNanos / 1e6, hostLaunchNanos / 1e3 / launches, hostGenerateNanos / 1e6));
        }
        out.append(String.format(Locale.ROOT, "  buffers: %d created (%d MB), %d reused from the pool (%d MB pooled, %d MB allocated; %d collections, %.1f ms)%n",
                buffersCreated, bytesCreated >> 20, buffersReused, pool.bytes() >> 20, allocated.get() >> 20,
                collections, collectNanos / 1e6));
        List<Map.Entry<String, long[]>> rows = new ArrayList<>(kernelTimes.entrySet());
        rows.sort((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1]));
        for (Map.Entry<String, long[]> e : rows) {
            long[] t = e.getValue();
            out.append(String.format(Locale.ROOT, "  %-28s %9.1f ms %8d x %8.1f us%n", e.getKey(), t[1] / 1e6, t[0],
                    t[1] / 1e3 / t[0]));
        }
        return out.toString();
    }

    /**
     * A new device buffer holding a copy of {@code host}. The copy is made while the buffer is
     * created, so it neither waits for queued kernels nor keeps the host memory alive.
     */
    private ClStorage uploaded(MemorySegment host, long size, DType dtype) {
        program(dtype);
        long bytes = host.byteSize();
        MemorySegment mem = createBuffer(CL.MEM_READ_WRITE | CL.MEM_COPY_HOST_PTR, bytes, host);
        allocated.addAndGet(bytes);
        ClStorage.Buffer buffer = new ClStorage.Buffer();
        buffer.bytes = bytes;
        buffer.mem = mem;
        ClStorage s = new ClStorage(this, buffer, size, dtype);
        // pooled only when the pool's sizes can use it
        CLEANER.register(s, new Release(cl, buffer, bytes, allocated, pool, bytes == bucket(bytes) ? budget / 4 : 0));
        return s;
    }

    @Override
    public synchronized Storage upload(float[] data) {
        if (data.length == 0) {
            return raw(0, DType.FLOAT32);
        }
        try (Arena arena = Arena.ofConfined()) {
            return uploaded(arena.allocateFrom(JAVA_FLOAT, data), data.length, DType.FLOAT32);
        }
    }

    @Override
    public synchronized Storage upload(double[] data, DType dtype) {
        if (data.length == 0) {
            return raw(0, dtype);
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = switch (dtype) {
                case FLOAT64 -> arena.allocateFrom(JAVA_DOUBLE, data);
                case FLOAT32 -> {
                    float[] f = new float[data.length];
                    for (int i = 0; i < f.length; i++) {
                        f[i] = (float) data[i];
                    }
                    yield arena.allocateFrom(JAVA_FLOAT, f);
                }
                case FLOAT16, BFLOAT16 -> {
                    short[] h = new short[data.length];
                    for (int i = 0; i < h.length; i++) {
                        h[i] = dtype == DType.FLOAT16 ? Float.floatToFloat16((float) data[i])
                                : DType.floatToBfloat16((float) data[i]);
                    }
                    yield arena.allocateFrom(JAVA_SHORT, h);
                }
            };
            return uploaded(host, data.length, dtype);
        }
    }

    @Override
    public synchronized float[] download(Storage src) {
        ClStorage s = cl(src);
        ready(s);
        if (s.dtype == DType.FLOAT32) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment host = arena.allocate(JAVA_FLOAT, Math.max(1, s.size));
                if (s.size > 0) {
                    cl.read(queue, s.mem(), 0, host.asSlice(0, s.bytes()));
                    // a blocking read on the in-order queue means all earlier work is done
                    settled();
                    synced();
                    generation++;
                }
                return host.asSlice(0, s.bytes()).toArray(JAVA_FLOAT);
            }
        }
        double[] d = downloadDouble(src);
        float[] f = new float[d.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }

    @Override
    public synchronized double[] downloadDouble(Storage src) {
        ClStorage s = cl(src);
        ready(s);
        double[] out = new double[Math.toIntExact(s.size)];
        if (s.size == 0) {
            return out;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(s.bytes(), 8);
            cl.read(queue, s.mem(), 0, host);
            settled();
            synced();
            generation++;
            switch (s.dtype) {
                case FLOAT64 -> MemorySegment.copy(host, JAVA_DOUBLE, 0, out, 0, out.length);
                case FLOAT32 -> {
                    for (int i = 0; i < out.length; i++) {
                        out[i] = host.getAtIndex(JAVA_FLOAT, i);
                    }
                }
                case FLOAT16 -> {
                    for (int i = 0; i < out.length; i++) {
                        out[i] = Float.float16ToFloat(host.getAtIndex(JAVA_SHORT, i));
                    }
                }
                case BFLOAT16 -> {
                    for (int i = 0; i < out.length; i++) {
                        out[i] = DType.bfloat16ToFloat(host.getAtIndex(JAVA_SHORT, i));
                    }
                }
            }
        }
        return out;
    }

    @Override
    public synchronized void convert(Storage in, Storage out) {
        ClStorage a = cl(in);
        ClStorage b = cl(out);
        ready(a);
        if (a.dtype == b.dtype) {
            if (a.size > 0) {
                cl.copy(queue, a.mem(), b.mem(), a.bytes());
            }
        } else if (b.dtype == DType.FLOAT32) {
            run(a.dtype, "to_float", a.size, a, b, a.size);
        } else if (a.dtype == DType.FLOAT32) {
            run(b.dtype, "from_float", a.size, a, b, a.size);
        } else {
            ClStorage mid = raw(a.size, DType.FLOAT32);
            run(a.dtype, "to_float", a.size, a, mid, a.size);
            run(b.dtype, "from_float", a.size, mid, b, a.size);
        }
    }

    /** Returns the storage in the given type, converting into a new buffer if needed. */
    private ClStorage as(Storage s, DType type) {
        ClStorage c = cl(s);
        if (c.dtype == type) {
            return c;
        }
        ClStorage converted = raw(c.size, type);
        convert(c, converted);
        return converted;
    }

    @Override
    public synchronized void fill(Storage dst, float value) {
        ClStorage s = cl(dst);
        if (fusable(s)) {
            program(s.dtype);
            s.pending = new Fused.Const(value);
            s.pendingNodes = 1;
            s.pendingGeneration = generation;
            return;
        }
        run(s.dtype, "fill", s.size, s, value, s.size);
    }

    // ---------------------------------------------------------------- element-wise

    @Override
    public synchronized Storage random(long size, long seed) {
        ClStorage o = raw(size, DType.FLOAT32);
        if (fusable(o)) {
            // recorded: generated inside the kernel that uses the values, never stored
            defer(o, new Fused.Random(seed));
        } else if (size > 0) {
            run(DType.FLOAT32, "random", size, o, seed, size);
        }
        return o;
    }

    @Override
    public synchronized void unary(UnaryOp op, Storage in, Storage out) {
        ClStorage o = cl(out);
        if (fusable(o, in)) {
            defer(o, new Fused.Un(op.ordinal(), Fused.whole(cl(in))));
            return;
        }
        run(o.dtype, "unary", o.size, op.ordinal(), as(in, o.dtype), o, o.size);
    }

    @Override
    public synchronized void activation(ActivationOp op, float param, Storage in, Storage out) {
        ClStorage o = cl(out);
        if (fusable(o, in)) {
            defer(o, new Fused.Act(op.ordinal(), param, Fused.whole(cl(in))));
            return;
        }
        run(o.dtype, "activation", o.size, op.ordinal(), param, as(in, o.dtype), o, o.size);
    }

    @Override
    public synchronized void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        ClStorage o = cl(out);
        if (fusable(o, in, grad)) {
            defer(o, new Fused.ActBack(op.ordinal(), param, Fused.whole(cl(in)), Fused.whole(cl(grad))));
            return;
        }
        run(o.dtype, "activation_backward", o.size, op.ordinal(), param, as(in, o.dtype), as(grad, o.dtype), o,
                o.size);
    }

    @Override
    public synchronized void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape,
            Storage out) {
        ClStorage o = cl(out);
        if (fusable(o, a, b) && shape.length <= MAX_RANK) {
            defer(o, new Fused.Bin(op.ordinal(), Fused.strided(cl(a), 0, shape, aStrides),
                    Fused.strided(cl(b), 0, shape, bStrides)));
            return;
        }
        ClStorage x = as(a, o.dtype);
        ClStorage y = as(b, o.dtype);
        long[] contiguous = contiguous(shape);
        if (Arrays.equals(aStrides, contiguous) && Arrays.equals(bStrides, contiguous)) {
            run(o.dtype, "binary_contiguous", o.size, op.ordinal(), x, y, o, o.size);
            return;
        }
        Temp meta = meta(meta(shape, aStrides, bStrides));
        run(o.dtype, "binary_strided", o.size, op.ordinal(), x, y, o, o.size, shape.length, meta);
    }

    @Override
    public synchronized void copy(Storage in, long inOffset, long[] inStrides, int[] shape, Storage out,
            long outOffset, long[] outStrides) {
        ClStorage o = cl(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        if (outOffset == 0 && n == o.size && Arrays.equals(outStrides, contiguous(shape)) && fusable(o, in)
                && shape.length <= MAX_RANK) {
            // a view: recorded, and read through its strides by whatever uses it
            defer(o, Fused.strided(cl(in), inOffset, shape, inStrides));
            return;
        }
        if (shape.length == 0) {
            shape = new int[] {1};
            inStrides = new long[] {1};
            outStrides = new long[] {1};
        }
        Temp meta = meta(meta(shape, inStrides, outStrides));
        run(o.dtype, "copy_strided", n, as(in, o.dtype), inOffset, o, outOffset, n, shape.length, meta);
    }

    private static long[] meta(int[] shape, long[] first, long[] second) {
        if (shape.length > MAX_RANK) {
            throw new IllegalArgumentException("at most " + MAX_RANK + " dimensions, got " + shape.length);
        }
        int r = shape.length;
        long[] m = new long[3 * r];
        for (int d = 0; d < r; d++) {
            m[d] = shape[d];
            m[r + d] = first[d];
            m[2 * r + d] = second[d];
        }
        return m;
    }

    private static long[] contiguous(int[] shape) {
        long[] s = new long[shape.length];
        long step = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = step;
            step *= shape[i];
        }
        return s;
    }

    // ---------------------------------------------------------------- reductions and matmul

    @Override
    public synchronized void reduce(ReduceOp op, Storage in, long outer, long length, long inner, Storage out) {
        ClStorage x = cl(in);
        ClStorage o = cl(out);
        long rows = outer * inner;
        if (op == ReduceOp.ARGMAX && !(o.dtype == DType.FLOAT32 && !grouped(rows, length) && fusable(o, x))) {
            ClStorage target = o.dtype == DType.FLOAT32 ? o : raw(o.size, DType.FLOAT32);
            run(x.dtype, "argmax", rows, x, target, outer, length, inner);
            if (target != o) {
                convert(target, o);
            }
            return;
        }
        if (fusable(o, x)) {
            // recorded: reductions of the same shape needed at the same time run as one kernel
            program(o.dtype);
            o.pending = new Fused.Reduce(op.ordinal(), x, outer, length, inner);
            o.pendingNodes = 1;
            o.pendingGeneration = generation;
            pendingReduces.add(new java.lang.ref.WeakReference<>(o));
            return;
        }
        ClStorage src = as(x, o.dtype);
        if (grouped(rows, length)) {
            run(o.dtype, "reduce_group", new long[] {rows * reduceGroup}, new long[] {reduceGroup}, op.ordinal(), src,
                    o, rows, length, inner);
        } else {
            run(o.dtype, "reduce_rows", rows, op.ordinal(), src, o, rows, length, inner);
        }
    }

    /** Whether a reduction shares each output across a work-group (long rows, few outputs). */
    private boolean grouped(long rows, long length) {
        return rows < 1024 && length > 256 && reduceGroup > 1;
    }

    /** Reductions recorded and not yet run. */
    private final List<java.lang.ref.WeakReference<ClStorage>> pendingReduces = new ArrayList<>();

    /** Runs the pending reductions an expression reads. */
    private void readyReduces(Fused.Expr e) {
        for (ClStorage r : Fused.reduces(e)) {
            ready(r);
        }
    }

    /**
     * Runs the pending reduction of {@code s}, together with the other pending reductions of
     * the same shape and type whose inputs need nothing else computed first, in one generated
     * kernel that computes their inputs on the fly. Each sums in the order of the precompiled
     * kernels, so results do not depend on what ran together.
     */
    private void runReduces(ClStorage s) {
        Fused.Reduce r = (Fused.Reduce) s.pending;
        readyReduces(Fused.whole(r.x()));
        if (s.pending != r) {
            return;   // ran with another group meanwhile
        }
        List<ClStorage> group = new ArrayList<>();
        group.add(s);
        for (java.util.Iterator<java.lang.ref.WeakReference<ClStorage>> it = pendingReduces.iterator(); it.hasNext();) {
            ClStorage o = it.next().get();
            if (o == null || !(o.pending instanceof Fused.Reduce q)) {
                it.remove();
                continue;
            }
            if (o != s && o.dtype == s.dtype && q.outer() == r.outer() && q.length() == r.length()
                    && q.inner() == r.inner() && Fused.reduces(Fused.whole(q.x())).isEmpty()) {
                group.add(o);
            }
        }
        for (ClStorage o : group) {
            for (ClStorage p : Fused.matmuls(Fused.whole(((Fused.Reduce) o.pending).x())).keySet()) {
                ready(p);
            }
        }
        long rows = r.outer() * r.inner();
        int size = grouped(rows, r.length()) ? reduceGroup : 1;
        Fused.Kernel k = Fused.generateReduce(group, size);
        while (k.argBytes() > Fused.MAX_ARG_BYTES && group.size() > 1) {
            group = new ArrayList<>(group.subList(0, (group.size() + 1) / 2));
            k = Fused.generateReduce(group, size);
        }
        MemorySegment kernel = k.argBytes() > Fused.MAX_ARG_BYTES ? null : fusedKernel(s.dtype, k.source());
        if (kernel == null) {
            // compiling (or too large): the precompiled kernels sum in the same order
            eagerFallbacks++;
            Fused.Reduce mine = r;
            s.pending = null;
            s.pendingNodes = 0;
            ClStorage src = as(mine.x(), s.dtype);
            if (mine.op() == ReduceOp.ARGMAX.ordinal()) {
                run(s.dtype, "argmax", rows, src, s, mine.outer(), mine.length(), mine.inner());
            } else if (grouped(rows, mine.length())) {
                run(s.dtype, "reduce_group", new long[] {rows * reduceGroup}, new long[] {reduceGroup}, mine.op(),
                        src, s, rows, mine.length(), mine.inner());
            } else {
                run(s.dtype, "reduce_rows", rows, mine.op(), src, s, rows, mine.length(), mine.inner());
            }
            return;
        }
        for (ClStorage o : group) {
            o.pending = null;
            o.pendingNodes = 0;
        }
        Object[] args = new Object[k.args().size() + 4];
        args[0] = s;
        args[1] = rows;
        args[2] = r.length();
        args[3] = r.inner();
        for (int i = 0; i < k.args().size(); i++) {
            args[i + 4] = k.args().get(i);
        }
        String name = group.size() > 1 ? "fused_reduce_many" : "fused_reduce";
        if (size > 1) {
            launch(name, kernel, new long[] {rows * size}, new long[] {size}, args);
        } else {
            launch(name, kernel, new long[] {rows}, null, args);
        }
        generation++;
    }

    @Override
    public void matmul(Storage a, Storage b, int m, int k, int n, Storage out) {
        matmulBatched(a, b, 1, m, k, n, out);
    }

    @Override
    public synchronized void matmulBatched(Storage a, Storage b, int batch, int m, int k, int n, Storage out) {
        ClStorage o = cl(out);
        long[] global = {roundUp(n, tile), roundUp(m, tile), batch};
        long[] local = {tile, tile, 1};
        Operand x = operand(a, o.dtype, batch, m, k);
        Operand y = operand(b, o.dtype, batch, k, n);
        Object[] args = {x.storage(), y.storage(), o, m, k, n, x.batch(), y.batch(), (long) m * n, x.offset(),
                x.row(), x.col(), y.offset(), y.row(), y.col()};
        if (fusable(o, x.storage(), y.storage()) && (long) m * n * batch > 0) {
            // recorded: an element-wise chain on the product can become the kernel's epilogue,
            // and an unrolled image operand is computed as the kernel loads it
            ClStorage pa = computedInMatMul(x.storage()) ? x.storage() : null;
            ClStorage pb = computedInMatMul(y.storage()) ? y.storage() : null;
            if (pa == null) {
                ready(x.storage());
            }
            if (pb == null) {
                ready(y.storage());
            }
            // not a chain boundary: the chain that uses the product becomes its epilogue
            o.pending = new Fused.MatMul(args, global, local, pa, pb);
            o.pendingNodes = 1;
            o.pendingGeneration = generation;
            return;
        }
        run(o.dtype, "matmul", global, local, args);
    }

    /** Runs a recorded matrix product with the precompiled kernel. */
    private void runMatMul(ClStorage s) {
        Fused.MatMul mm = (Fused.MatMul) s.pending;
        Fused.Expr pa = prologue(mm.a());
        Fused.Expr pb = prologue(mm.b());
        if (pa != null || pb != null) {
            Fused.Kernel k = Fused.generateMatMul(null, s, false, pa, layout(mm, pa, true), pb, layout(mm, pb, false));
            MemorySegment kernel = k.argBytes() > Fused.MAX_ARG_BYTES ? null : fusedKernel(s.dtype, k.source());
            if (kernel != null) {
                s.pending = null;
                s.pendingNodes = 0;
                launch("fused_matmul", kernel, mm.global(), mm.local(), matmulArgs(s, mm, k, pa != null, pb != null));
                return;
            }
            eagerFallbacks++;   // compiling: the operands are computed into their buffers first
        }
        s.pending = null;
        s.pendingNodes = 0;
        quietRun(s.dtype, "matmul", mm.global(), mm.local(), mm.args());
    }

    /** A matrix operand read in place: element (i, j) of batch b at offset + b*batch + i*row + j*col. */
    private record Operand(ClStorage storage, long offset, long batch, long row, long col) {
    }

    /**
     * A matmul operand. A pending view (a transpose, slice or permutation recorded by
     * {@link #copy}) is read through its strides instead of being copied first.
     */
    private Operand operand(Storage s, DType type, int batch, int rows, int cols) {
        ClStorage c = cl(s);
        if (c.pending instanceof Fused.Src v && v.storage().dtype == type && v.storage() != c) {
            int[] dims = v.dims();
            long[] st = v.strides();
            int r = dims.length;
            if (r >= 2 && dims[r - 2] == rows && dims[r - 1] == cols) {
                // leading dimensions must form one evenly strided batch axis
                long count = 1;
                long batchStride = (long) rows * cols;
                boolean even = true;
                for (int d = r - 3; d >= 0; d--) {
                    if (dims[d] == 1) {
                        continue;
                    }
                    if (count == 1) {
                        batchStride = st[d];
                    } else if (st[d] != batchStride * count) {
                        even = false;
                    }
                    count *= dims[d];
                }
                if (even && count == batch) {
                    if (!computedInMatMul(v.storage())) {
                        ready(v.storage());
                    }
                    return new Operand(v.storage(), v.offset(), batch == 1 ? 0 : batchStride,
                            rows == 1 ? 0 : st[r - 2], cols == 1 ? 0 : st[r - 1]);
                }
            }
        }
        ClStorage dense = as(c, type);
        return new Operand(dense, 0, (long) rows * cols, cols, 1);
    }

    private static long roundUp(long v, int multiple) {
        return Math.max(multiple, Math.ceilDiv(v, multiple) * multiple);
    }

    // ---------------------------------------------------------------- windows and indexing

    @Override
    public synchronized void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        ClStorage o = cl(out);
        Tables t = tables(geometry);
        int ks = geometry.kernelSize();
        long total = (long) geometry.batch() * geometry.positionCount() * geometry.channels() * ks;
        ClStorage x = cl(in);
        if (fusable(o, x) && x.size <= Integer.MAX_VALUE) {
            // recorded: a matmul computes the windows as it loads them, a pooling as it reduces them
            defer(o, new Fused.Im2col(new Fused.Src(x, 0, new int[] {(int) x.size}, new long[] {1}, false),
                    t.im2col(), t, geometry.channels(), geometry.positionCount(), ks, geometry.imageSize(), padValue));
            return;
        }
        run(o.dtype, "im2col", total, as(in, o.dtype), new Temp(t.im2col()), o, geometry.channels(),
                geometry.positionCount(), ks, geometry.imageSize(), padValue, total);
    }

    @Override
    public synchronized void col2im(Storage cols, ConvGeometry geometry, Storage out) {
        ClStorage o = cl(out);
        Tables t = tables(geometry);
        long total = (long) geometry.batch() * geometry.channels() * geometry.imageSize();
        ClStorage c = cl(cols);
        if (fusable(o, c) && c.size <= Integer.MAX_VALUE) {
            // recorded: computed with the chain that produces the columns and the one that uses the image
            defer(o, new Fused.Col2im(new Fused.Src(c, 0, new int[] {(int) c.size}, new long[] {1}, false),
                    t.col2imOffsets(), t.col2imEntries(), t, geometry.channels(), geometry.positionCount(),
                    geometry.kernelSize(), geometry.imageSize()));
            return;
        }
        run(o.dtype, "col2im", total, as(cols, o.dtype), new Temp(t.col2imOffsets()), new Temp(t.col2imEntries()),
                o, geometry.channels(), geometry.positionCount(), geometry.kernelSize(), geometry.imageSize(), total);
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
        t = new Tables(ints(table).mem(), ints(offsets).mem(), ints(entries).mem());
        CLEANER.register(t, new TablesRelease(cl, t.im2col(), t.col2imOffsets(), t.col2imEntries()));
        tables.put(key, t);
        return t;
    }

    /** {@code table[p * kernelSize + k]}: the image cell kernel cell {@code k} reads at position {@code p}, or -1. */
    private static int[] table(ConvGeometry g) {
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

    private int[] indices(Storage storage, int count, long length) {
        double[] values = downloadDouble(storage);
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
        ClStorage o = cl(out);
        Temp idx = ints(indices(indices, count, length));
        long total = outer * count * inner;
        run(o.dtype, "gather", total, as(in, o.dtype), idx, o, length, inner, count, total);
        release(idx);
    }

    @Override
    public synchronized void scatterAdd(Storage src, long outer, int count, long inner, Storage indices, long length,
            Storage out) {
        ClStorage o = cl(out);
        int[] idx = indices(indices, count, length);
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
        Temp off = ints(offsets);
        Temp ent = ints(entries);
        long total = outer * length * inner;
        run(o.dtype, "scatter_add", total, as(src, o.dtype), off, ent, o, length, inner, count, total);
        release(off, ent);
    }

    private ClStorage cl(Storage s) {
        if (s instanceof ClStorage c && c.owner == this) {
            return c;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to " + name());
    }

    /** Waits until all queued work is done; for timing. */
    public synchronized void finish() {
        if (cl != null) {
            cl.finish(queue);
            settled();
        }
    }
}
