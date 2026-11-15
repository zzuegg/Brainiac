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
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
    private int tile;
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
                eldest.getValue().release(cl);
                return true;
            }
            return false;
        }
    };

    private record Program(MemorySegment program, Map<String, MemorySegment> kernels) {
    }

    /** Index tables for one convolution geometry, kept on the device. */
    private record Tables(MemorySegment im2col, MemorySegment col2imOffsets, MemorySegment col2imEntries) {
        void release(CL cl) {
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
        tile = lib.deviceLong(device, CL.DEVICE_MAX_WORK_GROUP_SIZE) >= 256 ? 16 : 8;
        long memory = lib.deviceLong(device, CL.DEVICE_GLOBAL_MEM_SIZE);
        Integer mb = Integer.getInteger("brainiac.opencl.memory");
        budget = mb != null ? (long) mb << 20 : memory / 10 * 8;
        context = lib.createContext(device);
        queue = lib.createQueue(context, device);
        cl = lib;
        program(DType.FLOAT32);
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
        MemorySegment program = cl.buildProgram(context, device, source, "");
        Map<String, MemorySegment> kernels = new HashMap<>();
        for (String name : List.of("fill", "unary", "activation", "activation_backward", "binary_contiguous",
                "binary_strided", "copy_strided", "reduce_partial", "reduce_final", "argmax", "matmul", "im2col",
                "col2im", "gather", "scatter_add", "to_float", "from_float")) {
            kernels.put(name, cl.createKernel(program, name));
        }
        p = new Program(program, kernels);
        programs.put(type, p);
        return p;
    }

    private String prelude(DType type) {
        String common = "#define TS " + tile + "\n";
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
        MemorySegment k = program(type).kernels().get(kernel);
        try (Arena arena = Arena.ofConfined()) {
            for (int i = 0; i < args.length; i++) {
                MemorySegment value = switch (args[i]) {
                    case ClStorage s -> handle(arena, s.mem);
                    case Temp t -> handle(arena, t.mem());
                    case Integer v -> arena.allocateFrom(JAVA_INT, v);
                    case Long v -> arena.allocateFrom(JAVA_LONG, v);
                    case Float v -> arena.allocateFrom(JAVA_FLOAT, v);
                    default -> throw new IllegalArgumentException("unsupported kernel argument " + args[i]);
                };
                cl.setArg(k, i, value);
            }
        }
        cl.run(queue, k, global, local);
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
        run(dtype, "fill", size, s, 0f, size);
        return s;
    }

    /** Allocates without clearing. */
    private ClStorage raw(long size, DType dtype) {
        program(dtype);
        long bytes = Math.max(1, size) * dtype.bytes();
        MemorySegment mem = pool.take(bytes);
        if (mem == null) {
            mem = createBuffer(CL.MEM_READ_WRITE, bytes, MemorySegment.NULL);
            allocated.addAndGet(bytes);
        }
        ClStorage s = new ClStorage(this, mem, size, dtype);
        CLEANER.register(s, new Release(cl, mem, bytes, allocated, pool, budget / 4));
        return s;
    }

    /**
     * Returns a buffer to the pool once its storage is unreachable, or frees it when the
     * pool is full. Holds no reference to the storage.
     */
    private record Release(CL cl, MemorySegment mem, long bytes, AtomicLong allocated, Pool pool, long poolLimit)
            implements Runnable {
        @Override
        public void run() {
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

    @Override
    public synchronized Storage upload(float[] data) {
        ClStorage s = raw(data.length, DType.FLOAT32);
        if (data.length > 0) {
            try (Arena arena = Arena.ofConfined()) {
                cl.write(queue, s.mem, 0, arena.allocateFrom(JAVA_FLOAT, data));
            }
        }
        return s;
    }

    @Override
    public synchronized Storage upload(double[] data, DType dtype) {
        ClStorage s = raw(data.length, dtype);
        if (data.length == 0) {
            return s;
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
            cl.write(queue, s.mem, 0, host);
        }
        return s;
    }

    @Override
    public synchronized float[] download(Storage src) {
        ClStorage s = cl(src);
        if (s.dtype == DType.FLOAT32) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment host = arena.allocate(JAVA_FLOAT, Math.max(1, s.size));
                if (s.size > 0) {
                    cl.read(queue, s.mem, 0, host.asSlice(0, s.bytes()));
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
        double[] out = new double[Math.toIntExact(s.size)];
        if (s.size == 0) {
            return out;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(s.bytes(), 8);
            cl.read(queue, s.mem, 0, host);
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
        if (a.dtype == b.dtype) {
            if (a.size > 0) {
                cl.copy(queue, a.mem, b.mem, a.bytes());
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
        run(s.dtype, "fill", s.size, s, value, s.size);
    }

    // ---------------------------------------------------------------- element-wise

    @Override
    public synchronized void unary(UnaryOp op, Storage in, Storage out) {
        ClStorage o = cl(out);
        run(o.dtype, "unary", o.size, op.ordinal(), as(in, o.dtype), o, o.size);
    }

    @Override
    public synchronized void activation(ActivationOp op, float param, Storage in, Storage out) {
        ClStorage o = cl(out);
        run(o.dtype, "activation", o.size, op.ordinal(), param, as(in, o.dtype), o, o.size);
    }

    @Override
    public synchronized void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        ClStorage o = cl(out);
        run(o.dtype, "activation_backward", o.size, op.ordinal(), param, as(in, o.dtype), as(grad, o.dtype), o,
                o.size);
    }

    @Override
    public synchronized void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape,
            Storage out) {
        ClStorage o = cl(out);
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
        if (op == ReduceOp.ARGMAX) {
            ClStorage target = o.dtype == DType.FLOAT32 ? o : raw(o.size, DType.FLOAT32);
            run(x.dtype, "argmax", rows, x, target, outer, length, inner);
            if (target != o) {
                convert(target, o);
            }
            return;
        }
        ClStorage src = as(x, o.dtype);
        long chunks = rows >= 1024 || length <= 256 ? 1 : Math.min(Math.ceilDiv(length, 256), Math.max(1, 65536 / rows));
        long chunkLen = Math.ceilDiv(length, chunks);
        // partial results in the accumulator type, from the buffer pool
        ClStorage part = raw(rows * chunks, o.dtype == DType.FLOAT64 ? DType.FLOAT64 : DType.FLOAT32);
        run(o.dtype, "reduce_partial", rows * chunks, op.ordinal(), src, part, outer, length, inner, chunks, chunkLen);
        run(o.dtype, "reduce_final", rows, op.ordinal(), part, o, outer, length, inner, chunks);
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
        run(o.dtype, "matmul", global, local, as(a, o.dtype), as(b, o.dtype), o, m, k, n, (long) m * k,
                (long) k * n, (long) m * n);
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
        run(o.dtype, "im2col", total, as(in, o.dtype), new Temp(t.im2col()), o, geometry.channels(),
                geometry.positionCount(), ks, geometry.imageSize(), padValue, total);
    }

    @Override
    public synchronized void col2im(Storage cols, ConvGeometry geometry, Storage out) {
        ClStorage o = cl(out);
        Tables t = tables(geometry);
        long total = (long) geometry.batch() * geometry.channels() * geometry.imageSize();
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
        }
    }
}
