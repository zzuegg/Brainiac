package io.github.zzuegg.brainiac.backend.hip;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * One HIP device in use: the stream every operation is queued on, the compiled
 * kernels and a pool of device memory.
 *
 * <p>Everything runs in order on one stream, so the host only waits when it reads values back.
 * Memory returned to the pool is handed out again only to work queued later, which the stream
 * runs after everything that used it before.
 *
 * <p>With {@code -Dbrainiac.hip.debug=true} the host waits for the device after every kernel,
 * copy and product, so an error is reported by the operation that caused it (with the kernel's
 * name and launch size) instead of by a later one.
 */
final class HipContext {

    static final boolean DEBUG = Boolean.getBoolean("brainiac.hip.debug");
    /** Prints every call into HIP before it is made ({@code -Dbrainiac.hip.trace=true}), to find a hang. */
    static final boolean TRACE = Boolean.getBoolean("brainiac.hip.trace");

    static void trace(String what) {
        if (TRACE) {
            System.err.println("brainiac hip: " + what);
            System.err.flush();
        }
    }
    /**
     * Times each kernel, wait, copy and allocation ({@code -Dbrainiac.hip.profile=true}), waiting
     * after every launch, so that {@link HipBackend#statistics()} can list where the time goes.
     */
    static final boolean PROFILE = Boolean.getBoolean("brainiac.hip.profile");
    private final java.util.Map<String, long[]> profile = new java.util.HashMap<>();
    private long profileStart;

    /** Waits for the device without counting it as a wait, to time what was just queued. */
    void profileSync() {
        cu.synchronize(stream);
    }

    /** With profiling on: counts {@code nanos} spent in {@code what}. */
    void record(String what, long nanos) {
        if (PROFILE) {
            if (profileStart == 0) {
                profileStart = System.nanoTime() - nanos;
            }
            long[] e = profile.computeIfAbsent(what, w -> new long[2]);
            e[0]++;
            e[1] += nanos;
        }
    }

    /**
     * Times each kernel, copy and rocBLAS call on the GPU itself ({@code -Dbrainiac.hip.profile=gpu}),
     * with events recorded before and after it on the stream, without waiting: what each takes
     * on the device, and how long the device sat idle before it (waiting for the host, or for
     * the launch to arrive).
     */
    static final boolean GPU_PROFILE = "gpu".equals(System.getProperty("brainiac.hip.profile"));
    private final ArrayDeque<Long> events = new ArrayDeque<>();
    private final List<Object[]> marks = new ArrayList<>();
    /** Per name: calls, busy ms, idle ms before. */
    private final Map<String, double[]> gpuProfile = new HashMap<>();
    private long lastEnd;
    private double gpuBusy;
    private double gpuIdle;

    private long event() {
        Long e = events.poll();
        return e != null ? e : cu.createEvent();
    }

    /** Milliseconds {@code work} takes on the GPU, the mean of {@code reps} runs after one to warm up. */
    float timeOnGpu(Runnable work, int reps) {
        work.run();
        work.run();
        cu.synchronize(stream);   // a library's first call may set itself up out of order
        long a = event();
        long b = event();
        cu.recordEvent(a, stream);
        for (int i = 0; i < reps; i++) {
            work.run();
        }
        cu.recordEvent(b, stream);
        cu.synchronizeEvent(b);
        float ms = cu.elapsed(a, b) / reps;
        events.push(a);
        events.push(b);
        return ms;
    }

    /** With the GPU profile on: records an event before work that {@link #gpuEnd} names; else 0. */
    long gpuStart() {
        if (!GPU_PROFILE) {
            return 0;
        }
        long e = event();
        cu.recordEvent(e, stream);
        return e;
    }

    /** With the GPU profile on: records the end of work started at {@code start}, as {@code what}. */
    void gpuEnd(long start, String what) {
        if (!GPU_PROFILE || start == 0) {
            return;
        }
        long e = event();
        cu.recordEvent(e, stream);
        marks.add(new Object[] {what, start, e});
        if (marks.size() >= 4096) {
            harvest();
        }
    }

    /** Waits for the recorded events and adds up their times. */
    private void harvest() {
        if (marks.isEmpty()) {
            return;
        }
        cu.synchronizeEvent((Long) marks.getLast()[2]);
        for (Object[] m : marks) {
            long start = (Long) m[1];
            long end = (Long) m[2];
            double busy = cu.elapsed(start, end);
            double idle = lastEnd != 0 ? Math.max(0, cu.elapsed(lastEnd, start)) : 0;
            double[] t = gpuProfile.computeIfAbsent((String) m[0], k -> new double[3]);
            t[0]++;
            t[1] += busy;
            t[2] += idle;
            gpuBusy += busy;
            gpuIdle += idle;
            events.push(start);
            if (lastEnd != 0) {
                events.push(lastEnd);
            }
            lastEnd = end;
        }
        marks.clear();
    }

    private String gpuReport() {
        harvest();
        StringBuilder b = new StringBuilder(String.format(java.util.Locale.ROOT,
                "%ngpu profile: %.1f ms busy, %.1f ms idle between timed work", gpuBusy, gpuIdle));
        gpuProfile.entrySet().stream().sorted((x, y) -> Double.compare(y.getValue()[1] + y.getValue()[2],
                x.getValue()[1] + x.getValue()[2])).limit(30)
                .forEach(e -> b.append(String.format(java.util.Locale.ROOT,
                        "%n  %-34s %8d calls %9.1f ms busy %7.1f us each %9.1f ms idle before",
                        e.getKey(), (long) e.getValue()[0], e.getValue()[1], e.getValue()[1] * 1e3 / e.getValue()[0],
                        e.getValue()[2])));
        return b.toString();
    }

    /** The profile so far: the costliest entries, and how much of the time they cover. */
    String profileReport() {
        if (GPU_PROFILE && cu.hasEvents()) {
            return gpuReport();
        }
        if (!PROFILE || profile.isEmpty()) {
            return "";
        }
        long total = 0;
        for (long[] e : profile.values()) {
            total += e[1];
        }
        long wall = System.nanoTime() - profileStart;
        StringBuilder b = new StringBuilder("\nprofile: " + total / 1_000_000 + " ms timed of " + wall / 1_000_000
                + " ms since the first call (the rest is the host between calls)");
        profile.entrySet().stream().sorted((x, y) -> Long.compare(y.getValue()[1], x.getValue()[1])).limit(25)
                .forEach(e -> b.append(String.format(java.util.Locale.ROOT, "%n  %-34s %8d calls %10.1f ms %8.1f us each",
                        e.getKey(), e.getValue()[0], e.getValue()[1] / 1e6, e.getValue()[1] / 1e3 / e.getValue()[0])));
        return b.toString();
    }

    static final int GROUP = 256;
    /** Most blocks a one-dimensional launch uses; the kernels loop over the rest. */
    private static final int MAX_BLOCKS = 1 << 20;

    final HipRuntime cu;
    final Hiprtc hiprtc;
    final Rocblas rocblas;
    final String rocblasUnavailable;
    final int device;
    final String deviceName;
    final int multiprocessors;
    final int warpSize;
    final long deviceMemory;
    final long budget;
    /** The GPU's compiler target, such as {@code gfx1201}. */
    final String architecture;
    final long compileNanos;
    final boolean cachedKernels;
    final long stream;
    /** Modules by part of kernels.hip (see {@link #part}), loaded at first use. */
    private final Map<String, Long> modules = new HashMap<>();
    private final Map<String, Long> functions = new HashMap<>();
    /** The source of kernels.hip, read once. */
    private static String source;

    // kernel arguments: values, and the void*[] pointing at each, reused for every launch
    private final MemorySegment argValues;
    private final MemorySegment argPointers;
    private int argOffset;
    private int argCount;

    // page-locked memory for reading values back
    private MemorySegment staging;

    // statistics
    long launches;
    /** Time the host spent in launch and copy calls, which return before the device runs them. */
    long callNanos;
    long products;
    long waits;
    long uploadedBytes;
    long downloadedBytes;

    /** The runtime and hipRTC, loaded once for every device. */
    private static HipRuntime sharedRuntime;
    private static Hiprtc sharedHiprtc;

    static synchronized HipRuntime runtime() {
        if (sharedRuntime == null) {
            trace("loading the HIP runtime");
            HipRuntime d = new HipRuntime();
            trace("hipInit (" + d.library + ")");
            d.init();
            sharedRuntime = d;
        }
        return sharedRuntime;
    }

    private static synchronized Hiprtc hiprtc() {
        if (sharedHiprtc == null) {
            trace("loading hipRTC");
            sharedHiprtc = new Hiprtc();
        }
        return sharedHiprtc;
    }

    /** The device current on each thread, as far as this backend knows. */
    private static final ThreadLocal<HipContext> CURRENT = new ThreadLocal<>();

    final int ordinal;

    /**
     * The device with the most compute units (then the most memory): the graphics card rather
     * than an integrated GPU, which HIP may number first.
     */
    static int largestDevice() {
        HipRuntime rt = runtime();
        int best = 0;
        long bestScore = -1;
        for (int i = 0; i < rt.deviceCount(); i++) {
            long score;
            try {
                int device = rt.device(i);
                long memory = rt.totalMemory(device);
                score = (long) rt.properties(device, memory).multiprocessors() << 40 | Math.min(memory >> 20, (1L << 40) - 1);
            } catch (HipException e) {
                score = 0;
            }
            if (score > bestScore) {
                best = i;
                bestScore = score;
            }
        }
        return best;
    }

    /**
     * Opens a device.
     *
     * @param ordinal its index among the driver's devices
     */
    HipContext(int ordinal) {
        cu = runtime();
        int count = cu.deviceCount();
        if (count == 0) {
            throw new HipException("the HIP runtime found no AMD GPU", 0);
        }
        if (ordinal < 0 || ordinal >= count) {
            throw new HipException("device " + ordinal + " asked for (-Dbrainiac.hip.device) but there are " + count
                    + " devices", 0);
        }
        this.ordinal = ordinal;
        device = cu.device(ordinal);
        deviceName = cu.deviceName(device);
        deviceMemory = cu.totalMemory(device);
        HipRuntime.Properties props = cu.properties(device, deviceMemory);
        architecture = props.architecture();
        warpSize = props.warpSize();
        multiprocessors = props.multiprocessors() > 0 ? props.multiprocessors() : 32;
        if (warpSize == 64 || architecture.startsWith("gfx9") || architecture.startsWith("gfx8")) {
            // the kernels' reductions and attention are written for 32 lanes
            throw new HipException(deviceName + " (" + architecture + ") runs 64-lane wavefronts; the hip backend "
                    + "needs an RDNA GPU (gfx10, gfx11, gfx12)", 0);
        }
        trace("device " + ordinal + ": " + deviceName + ", " + architecture + ", warp " + warpSize + ", "
                + multiprocessors + " CUs");
        cu.setDevice(ordinal);
        CURRENT.set(this);
        trace("hipStreamCreateWithFlags");
        stream = cu.createStream(HipRuntime.HIP_STREAM_NON_BLOCKING);

        long mb = Long.getLong("brainiac.hip.memory", 0);
        long free = cu.memoryInfo()[0];
        budget = mb > 0 ? mb << 20 : Math.max(256L << 20, free - Math.max(512L << 20, deviceMemory / 16));

        hiprtc = hiprtc();
        long start = System.nanoTime();
        boolean[] cached = new boolean[1];
        module("CORE", cached);
        cachedKernels = cached[0];
        compileNanos = System.nanoTime() - start;

        Arena global = Arena.global();
        argValues = global.allocate(4096, 16);
        argPointers = global.allocate(ADDRESS, 64);

        Rocblas blas = null;
        String why = null;
        if ("none".equals(System.getProperty("brainiac.hip.rocblas"))) {
            why = "turned off (-Dbrainiac.hip.rocblas=none)";
        } else {
            try {
                trace("loading rocBLAS");
                blas = new Rocblas(stream, architecture);
                trace("rocBLAS " + blas.version());
            } catch (HipException | LinkageError | IllegalArgumentException e) {
                why = e.getMessage();
            }
        }
        rocblas = blas;
        rocblasUnavailable = why;
    }

    /**
     * Makes the device current on the calling thread, if it is not already. HIP keeps the
     * current device per operating-system thread, and a virtual thread (the studio serves
     * requests on them) runs on whichever carrier thread is free, whose current device may be
     * another GPU's: there the device is set every time.
     */
    void bind() {
        if (Thread.currentThread().isVirtual()) {
            cu.setDevice(ordinal);
        } else if (CURRENT.get() != this) {
            cu.setDevice(ordinal);
            CURRENT.set(this);
        }
    }

    // ------------------------------------------------------------------ kernels

    /**
     * The code object of a part of kernels.hip for this GPU: from the cache folder when it was
     * compiled before by the same hipRTC for the same architecture, otherwise compiled now (a
     * few seconds) and stored there. {@code -Dbrainiac.hip.cache} names the folder (default
     * {@code ~/.cache/brainiac/hip}, {@code none} for no cache).
     */
    private byte[] kernels(String part, boolean[] cached) {
        return compile(source(), "kernels.hip", "kernels", List.of("-DBRAINIAC_" + part), cached);
    }

    /** Compiles {@code source} for this GPU, or reads it from the cache folder (see {@link #kernels}). */
    private byte[] compile(String source, String name, String prefix, List<String> extra, boolean[] cached) {
        List<String> options = new ArrayList<>(List.of("--gpu-architecture=" + architecture, "-std=c++17", "-O3"));
        options.addAll(extra);
        String hiprtcVersion = hiprtc.version();
        Path file = cacheFile(prefix, source + "\n" + options + "\n" + hiprtcVersion + "\n" + hiprtc.library);
        if (file != null && Files.isRegularFile(file)) {
            try {
                byte[] code = Files.readAllBytes(file);
                cached[0] = true;
                return code;
            } catch (IOException e) {
                // compiled again below
            }
        }
        Hiprtc.Code result = hiprtc.compile(source, name, options);
        if (DEBUG && !result.log().isBlank()) {
            System.err.println("brainiac hip: hipRTC messages:\n" + result.log());
        }
        byte[] code = result.code();
        if (file != null) {
            try {
                Files.createDirectories(file.getParent());
                Path tmp = Files.createTempFile(file.getParent(), prefix, ".tmp");
                Files.write(tmp, code);
                Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException | UnsupportedOperationException e) {
                // no cache then
            }
        }
        return code;
    }

    private static Path cacheFile(String prefix, String key) {
        String dir = System.getProperty("brainiac.hip.cache");
        if ("none".equals(dir)) {
            return null;
        }
        Path folder = dir != null ? Path.of(dir)
                : Path.of(System.getProperty("user.home"), ".cache", "brainiac", "hip");
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.UTF_8));
            return folder.resolve(prefix + "-" + HexFormat.of().formatHex(hash, 0, 12) + ".hsaco");
        } catch (NoSuchAlgorithmException e) {
            return null;
        }
    }

    private static synchronized String source() {
        if (source == null) {
            try (InputStream in = HipContext.class.getResourceAsStream("kernels.hip")) {
                if (in == null) {
                    throw new HipException("kernels.hip is missing from the brainiac-backend-hip module", 0);
                }
                source = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            } catch (IOException e) {
                throw new HipException("reading kernels.hip: " + e.getMessage(), e);
            }
        }
        return source;
    }

    /**
     * The part of kernels.hip that defines a kernel: {@code CORE}, {@code FEW_<format>},
     * {@code ATTENTION_<format>} or {@code FFT}. Compiled separately, so a program only waits
     * for hipRTC on the kernels it uses.
     */
    static String part(String kernel) {
        String format = kernel.substring(kernel.lastIndexOf('_') + 1).toUpperCase(java.util.Locale.ROOT);
        if (kernel.startsWith("few_") || kernel.startsWith("unpack_packed_") || kernel.startsWith("packed_mv") || kernel.startsWith("dense_mv") || kernel.startsWith("unpack_words_")) {
            return "FEW_" + format;
        }
        if (kernel.startsWith("attention")) {
            return "ATTENTION_" + format;
        }
        if (kernel.equals("fft_rows")) {
            return "FFT";
        }
        return "CORE";
    }

    /** The module of a part of kernels.hip, compiled (or read from the cache) and loaded at first use. */
    private long module(String part, boolean[] cached) {
        Long m = modules.get(part);
        if (m == null) {
            byte[] code = kernels(part, cached);
            try (Arena a = Arena.ofConfined()) {
                MemorySegment image = a.allocate(code.length + 1L, 16);
                MemorySegment.copy(code, 0, image, JAVA_BYTE, 0, code.length);
                trace("hipModuleLoadData " + part + " (" + code.length + " bytes)");
                m = cu.loadModule(image);
            }
            modules.put(part, m);
        }
        return m;
    }

    /** Generated kernels ({@link HipFusion}): their functions by source. */
    private final Map<String, Long> fused = new HashMap<>();
    /** Their names in profiles: fused#0, fused#1, ... as the sources are numbered in the dump folder. */
    private final Map<Long, String> fusedNames = new HashMap<>();
    /** Time spent compiling (or reading from the cache) and loading generated kernels. */
    long fusedCompileNanos;

    /**
     * The function {@code fused} of a generated kernel, compiled with the shared part of
     * kernels.hip (its formats and math) at first use, or read from the cache folder.
     */
    long fusedFunction(String body) {
        Long f = fused.get(body);
        if (f == null) {
            long start = System.nanoTime();
            String all = prelude() + "\n" + body;
            byte[] code = compile(all, "fused.hip", "fused", List.of(), new boolean[1]);
            long m;
            try (Arena a = Arena.ofConfined()) {
                MemorySegment image = a.allocate(code.length + 1L, 16);
                MemorySegment.copy(code, 0, image, JAVA_BYTE, 0, code.length);
                trace("hipModuleLoadData fused#" + fused.size() + " (" + code.length + " bytes)");
                m = cu.loadModule(image);
            }
            modules.put("fused#" + fused.size(), m);
            f = cu.function(m, "fused");
            fusedNames.put(f, "fused#" + fused.size());
            fused.put(body, f);
            fusedCompileNanos += System.nanoTime() - start;
            if (PROFILE || DEBUG) {
                try {
                    Path dir = Path.of(System.getProperty("java.io.tmpdir"), "brainiac-hip-fused");
                    Files.createDirectories(dir);
                    Files.writeString(dir.resolve("fused_" + (fused.size() - 1) + ".hip"), body);
                } catch (IOException e) {
                    // only a help for reading the profile
                }
            }
        }
        return f;
    }

    /** Distinct generated kernels so far. */
    int fusedKernels() {
        return fused.size();
    }

    /** kernels.hip up to its indexing section: the formats and the math generated kernels call. */
    private static String prelude() {
        String s = source();
        int end = s.indexOf("// ---------------------------------------------------------------- indexing");
        return end < 0 ? s : s.substring(0, end);
    }

    /** Launches a generated kernel's function over {@code n} items; the arguments are set. */
    void launchFused(long f, long n) {
        int blocks = (int) Math.max(1, Math.min(MAX_BLOCKS, (n + GROUP - 1) / GROUP));
        long g = gpuStart();
        long t0 = System.nanoTime();
        int r = cu.launch(f, blocks, 1, 1, GROUP, 0, stream, argPointers);
        callNanos += System.nanoTime() - t0;
        if (g != 0) {
            gpuEnd(g, fusedNames.getOrDefault(f, "fused") + " (" + n + " values)");
        }
        launches++;
        if (r != HipRuntime.HIP_SUCCESS) {
            throw cu.error(r, "launching a fused kernel (" + blocks + " blocks)");
        }
        if (PROFILE) {
            cu.synchronize(stream);
            record(fusedNames.getOrDefault(f, "fused") + " (" + n + " values)", System.nanoTime() - t0);
        }
        if (DEBUG) {
            debugWait("a fused kernel (" + blocks + " blocks)");
        }
    }

    long function(String name) {
        Long f = functions.get(name);
        if (f == null) {
            f = cu.function(module(part(name), new boolean[1]), name);
            functions.put(name, f);
        }
        return f;
    }

    /** Starts the arguments of the next launch. */
    HipContext args() {
        argOffset = 0;
        argCount = 0;
        return this;
    }

    private long slot(int bytes) {
        if (argCount == 64 || argOffset + bytes > argValues.byteSize()) {
            throw new IllegalStateException("too many kernel arguments");
        }
        long at = argOffset;
        argPointers.setAtIndex(ADDRESS, argCount++, argValues.asSlice(at, bytes));
        argOffset += (bytes + 7) & ~7;
        return at;
    }

    /** A device pointer (or any 64-bit integer). */
    HipContext ptr(long v) {
        argValues.set(JAVA_LONG, slot(8), v);
        return this;
    }

    HipContext i64(long v) {
        return ptr(v);
    }

    HipContext i32(int v) {
        argValues.set(JAVA_INT, slot(4), v);
        return this;
    }

    HipContext f32(float v) {
        argValues.set(JAVA_FLOAT, slot(4), v);
        return this;
    }

    HipContext f64(double v) {
        argValues.set(JAVA_DOUBLE, slot(8), v);
        return this;
    }

    /** A struct of {@code bytes} bytes passed by value, written by {@code writer} at the offset it is given. */
    HipContext struct(int bytes, java.util.function.ObjLongConsumer<MemorySegment> writer) {
        long at = slot(bytes);
        writer.accept(argValues, at);
        return this;
    }

    /** A struct of 4-byte fields passed by value; a float field goes as its raw bits. */
    HipContext ints(int... fields) {
        long at = slot(4 * fields.length);
        for (int i = 0; i < fields.length; i++) {
            argValues.set(JAVA_INT, at + 4L * i, fields[i]);
        }
        return this;
    }

    /**
     * A {@code Strided} struct: rank, padding, then shape, the first operand's strides and the
     * second's, {@code MAX_AXES} longs each.
     */
    HipContext strided(int rank, long[] shape, long[] s0, long[] s1) {
        int axes = Axes.MAX_AXES;
        long at = slot(8 + 3 * 8 * axes);
        argValues.set(JAVA_INT, at, rank);
        argValues.set(JAVA_INT, at + 4, 0);
        for (int d = 0; d < axes; d++) {
            argValues.set(JAVA_LONG, at + 8 + 8L * d, d < rank ? shape[d] : 1);
            argValues.set(JAVA_LONG, at + 8 + 8L * (axes + d), d < rank ? s0[d] : 0);
            argValues.set(JAVA_LONG, at + 8 + 8L * (2 * axes + d), d < rank ? s1[d] : 0);
        }
        return this;
    }

    /** Launches {@code name} over {@code n} items: enough blocks of {@link #GROUP} threads, at most {@link #MAX_BLOCKS}. */
    void launch(String name, long n) {
        launch(name, (int) Math.max(1, Math.min(MAX_BLOCKS, (n + GROUP - 1) / GROUP)), 1, 1);
    }

    /** Launches {@code name} with a grid of blocks of {@link #GROUP} threads. */
    void launch(String name, int gridX, int gridY, int gridZ) {
        launch(name, gridX, gridY, gridZ, GROUP);
    }

    /** Launches {@code name} with a grid of blocks of {@code block} threads. */
    void launch(String name, int gridX, int gridY, int gridZ, int block) {
        long f = function(name);
        trace("launch " + name + " (" + gridX + " x " + gridY + " x " + gridZ + " blocks)");
        long g = gpuStart();
        long t0 = System.nanoTime();
        int r = cu.launch(f, gridX, gridY, gridZ, block, 0, stream, argPointers);
        callNanos += System.nanoTime() - t0;
        gpuEnd(g, name);
        launches++;
        if (r != HipRuntime.HIP_SUCCESS) {
            throw cu.error(r, "launching " + name + " (" + gridX + " x " + gridY + " x " + gridZ + " blocks)");
        }
        if (PROFILE) {
            cu.synchronize(stream);
            record(name, System.nanoTime() - t0);
        }
        if (DEBUG) {
            debugWait("kernel " + name + " (" + gridX + " x " + gridY + " x " + gridZ + " blocks)");
        }
    }

    /** In debug mode: waits for the device and reports an error as caused by {@code what}. */
    void debugWait(String what) {
        trace("hipStreamSynchronize after " + what);
        int r = cu.synchronize(stream);
        if (r != HipRuntime.HIP_SUCCESS) {
            throw cu.error(r, what);
        }
    }

    /** Waits until the device has run everything queued. */
    void finish() {
        waits++;
        trace("hipStreamSynchronize");
        long t0 = PROFILE ? System.nanoTime() : 0;
        int r = cu.synchronize(stream);
        record("wait", PROFILE ? System.nanoTime() - t0 : 0);
        if (r != HipRuntime.HIP_SUCCESS) {
            throw cu.error(r, "waiting for the device (run with -Dbrainiac.hip.debug=true to find the operation)");
        }
    }

    // ------------------------------------------------------------------ copies

    void upload(long dst, MemorySegment src) {
        long bytes = src.byteSize();
        if (bytes == 0) {
            return;
        }
        uploadedBytes += bytes;
        long t0 = PROFILE ? System.nanoTime() : 0;
        trace("hipMemcpyHtoDAsync " + bytes + " bytes" + (src.isNative() ? "" : " from the heap"));
        if (src.isNative()) {
            cu.copyToDevice(dst, src, bytes, stream);
        } else if (bytes <= SMALL_UPLOAD && cu.hasEvents() && !DEBUG) {
            // small (a step's positions or mask): through a ring of pinned buffers, each reused
            // once the copy that last read it is done, so the device need not go idle first
            int slot = nextUpload++ % UPLOAD_SLOTS;
            if (uploadBuffers[slot] == null) {
                uploadBuffers[slot] = cu.allocateHost(SMALL_UPLOAD);
                uploadEvents[slot] = cu.createEvent();
            } else {
                cu.synchronizeEvent(uploadEvents[slot]);   // long done, but for a burst of uploads
            }
            MemorySegment.copy(src, 0, uploadBuffers[slot], 0, bytes);
            cu.copyToDevice(dst, uploadBuffers[slot], bytes, stream);
            cu.recordEvent(uploadEvents[slot], stream);
        } else {
            // a Java array: copied off the heap first. HIP may still read pageable memory after
            // the call returns, so the copy is waited for before the memory is freed.
            try (Arena a = Arena.ofConfined()) {
                MemorySegment off = a.allocate(bytes, 8);
                MemorySegment.copy(src, 0, off, 0, bytes);
                cu.copyToDevice(dst, off, bytes, stream);
                finish();
            }
        }
        if (DEBUG) {
            debugWait("upload of " + bytes + " bytes");
        }
        if (PROFILE) {
            cu.synchronize(stream);
            record(src.isNative() ? "upload" : "upload from the heap", System.nanoTime() - t0);
        }
    }

    /** Copies device memory into {@code dst}, waiting for it. */
    void download(MemorySegment dst, long src) {
        long bytes = dst.byteSize();
        downloadedBytes += bytes;
        long t0 = PROFILE ? System.nanoTime() : 0;
        long chunk = Math.min(bytes, 64L << 20);
        if (staging == null || staging.byteSize() < chunk) {
            staging = cu.allocateHost(Math.max(chunk, 1L << 20));
        }
        for (long done = 0; done < bytes; done += chunk) {
            long n = Math.min(chunk, bytes - done);
            trace("hipMemcpyDtoHAsync " + n + " bytes");
            cu.copyToHost(staging, src + done, n, stream);
            finish();
            MemorySegment.copy(staging, 0, dst, done, n);
        }
        record("download", PROFILE ? System.nanoTime() - t0 : 0);
    }

    /** The largest upload from the Java heap that goes through {@link #uploadBuffers} without a wait. */
    static final long SMALL_UPLOAD = 64 << 10;
    private static final int UPLOAD_SLOTS = 32;
    /** Pinned buffers for small uploads from the heap, used in turn, and the events that mark their copies done. */
    private final MemorySegment[] uploadBuffers = new MemorySegment[UPLOAD_SLOTS];
    private final long[] uploadEvents = new long[UPLOAD_SLOTS];
    private int nextUpload;

    /** The largest copy {@link #downloadLater} takes: a pinned buffer of this size each. */
    static final long SMALL_READBACK = 64 << 10;
    /** Pinned buffers for {@link #downloadLater}, free for the next. */
    private final ArrayDeque<MemorySegment> readbacks = new ArrayDeque<>();

    /** A copy to the host in flight: its pinned buffer and the event that marks it done. */
    record Pending(MemorySegment host, long event, long bytes) {
    }

    /** Whether {@link #downloadLater} can take {@code bytes}: small, and with events to wait on. */
    boolean readsLater(long bytes) {
        return bytes <= SMALL_READBACK && cu.hasEvents() && !DEBUG;
    }

    /** Queues a copy of at most {@link #SMALL_READBACK} bytes to the host, without waiting for it. */
    Pending downloadLater(long src, long bytes) {
        downloadedBytes += bytes;
        MemorySegment h = readbacks.poll();
        if (h == null) {
            h = cu.allocateHost(SMALL_READBACK);
        }
        trace("hipMemcpyDtoHAsync " + bytes + " bytes, read later");
        cu.copyToHost(h, src, bytes, stream);
        long e = event();
        cu.recordEvent(e, stream);
        return new Pending(h, e, bytes);
    }

    /** Waits for a copy from {@link #downloadLater} (only for it, not for work queued after it), and gives its bytes. */
    MemorySegment read(Pending p) {
        bind();
        long t0 = PROFILE ? System.nanoTime() : 0;
        cu.synchronizeEvent(p.event());
        record("wait for a read-back", PROFILE ? System.nanoTime() - t0 : 0);
        return p.host().asSlice(0, p.bytes());
    }

    /** Takes back the buffer and event of a copy that was read. */
    void release(Pending p) {
        events.add(p.event());
        readbacks.add(p.host());
    }

    void copy(long dst, long src, long bytes) {
        if (bytes > 0) {
            trace("hipMemcpyDtoDAsync " + bytes + " bytes");
            long g = gpuStart();
            long t0 = System.nanoTime();
            cu.copy(dst, src, bytes, stream);
            callNanos += System.nanoTime() - t0;
            gpuEnd(g, "device copy");
            if (DEBUG) {
                debugWait("device copy of " + bytes + " bytes");
            }
        }
    }

    void fill32(long dst, int bits, long count) {
        if (count > 0) {
            trace("hipMemsetD32Async " + count);
            cu.fill32(dst, bits, count, stream);
        }
    }

    void fill16(long dst, short bits, long count) {
        if (count > 0) {
            trace("hipMemsetD16Async " + count);
            cu.fill16(dst, bits, count, stream);
        }
    }

    void fill8(long dst, byte bits, long count) {
        if (count > 0) {
            trace("hipMemsetD8Async " + count);
            cu.fill8(dst, bits, count, stream);
        }
    }

    // ------------------------------------------------------------------ memory

    /** Freed allocations from the collector's thread, returned to the pool by the next allocation. */
    private final ConcurrentLinkedQueue<long[]> collected = new ConcurrentLinkedQueue<>();
    private final Map<Long, ArrayDeque<Long>> pool = new HashMap<>();
    /**
     * Free blocks of the small sizes, most cut from slabs: hipMalloc takes tens of microseconds
     * and a decoded token asks for hundreds of small blocks, which come back only once the
     * garbage collector finds their tensors. Never trimmed; freed at {@link #close()}.
     */
    private final Map<Long, ArrayDeque<Long>> smallPool = new HashMap<>();
    /** The slabs: their addresses and sizes. */
    private final java.util.TreeMap<Long, Long> slabs = new java.util.TreeMap<>();
    /** The largest size class cut from slabs. */
    static final long SLAB_LIMIT = 1L << 20;
    /** Whether small blocks are cut from slabs ({@code -Dbrainiac.hip.slabs=false}: a hipMalloc each). */
    static final boolean SLABS = !"false".equals(System.getProperty("brainiac.hip.slabs"));
    private final Set<Long> live = new HashSet<>();
    private long inUse;
    private long peak;
    private long reserved;
    long collections;

    /**
     * Allocation sizes: multiples of 256 bytes up to 4 KB, then four steps per power of two up
     * to 1 MB, then 64 (at least 64 KB apart). Large blocks are mostly weights, kept for good,
     * where four steps wasted up to a quarter (190 MB on a 7B model's 1.1 GB embeddings).
     */
    static long sizeClass(long bytes) {
        long b = Math.max(256, (bytes + 255) & ~255L);
        if (b <= 4096) {
            return b;
        }
        long top = Long.highestOneBit(b);
        long step = b <= (1L << 20) ? top >> 2 : Math.max(64L << 10, top >> 6);
        return (b + step - 1) & -step;
    }

    /** Device memory of at least {@code bytes}, from the pool when it has a block of that size. */
    long allocate(long bytes) {
        drainCollected();
        long size = sizeClass(bytes);
        ArrayDeque<Long> free = freeList(size);
        long ptr;
        if ((free == null || free.isEmpty()) && collectSoon(size)) {
            free = freeList(size);
        }
        if (free != null && !free.isEmpty()) {
            ptr = free.pop();
        } else if (SLABS && size <= SLAB_LIMIT && reserved + slabBytes(size) <= budget) {
            ptr = cutSlab(size);
        } else {
            if (reserved + size > budget) {
                long t0 = PROFILE ? System.nanoTime() : 0;
                ptr = reclaim(size);
                record("reclaim (garbage collection)", PROFILE ? System.nanoTime() - t0 : 0);
                if (ptr != 0) {
                    live.add(ptr);
                    inUse += size;
                    peak = Math.max(peak, inUse);
                    site(ptr, size);
                    return ptr;
                }
            }
            trace("hipMalloc " + size + " bytes");
            long t0 = PROFILE ? System.nanoTime() : 0;
            ptr = allocateFromDriver(size);
            record("hipMalloc", PROFILE ? System.nanoTime() - t0 : 0);
            reserved += size;
        }
        live.add(ptr);
        inUse += size;
        peak = Math.max(peak, inUse);
        site(ptr, size);
        return ptr;
    }

    /**
     * Whether allocations are traced to the code that made them ({@code -Dbrainiac.hip.memorySites=true}):
     * the statistics then show what held the memory at its peak.
     */
    static final boolean SITES = Boolean.getBoolean("brainiac.hip.memorySites");
    /** Per live allocation with {@link #SITES}: its size and where it was made. */
    private final Map<Long, Object[]> sites = new HashMap<>();
    /** What held the memory at its peak, per site, as of the last peak recorded. */
    private Map<String, long[]> peakSites = Map.of();
    private long peakSitesAt;

    private void site(long ptr, long size) {
        if (!SITES) {
            return;
        }
        String where = StackWalker.getInstance().walk(frames -> frames
                .filter(f -> !f.getClassName().startsWith("io.github.zzuegg.brainiac.backend.hip")
                        && !f.getClassName().startsWith("io.github.zzuegg.brainiac.tensor.Tensor"))
                .limit(3).map(f -> f.getClassName().substring(f.getClassName().lastIndexOf('.') + 1) + "."
                        + f.getMethodName() + ":" + f.getLineNumber())
                .collect(java.util.stream.Collectors.joining(" < ")));
        sites.put(ptr, new Object[] {size, where});
        if (inUse >= peakSitesAt + (256L << 20)) {
            Map<String, long[]> now = new HashMap<>();
            for (Object[] e : sites.values()) {
                long[] t = now.computeIfAbsent((String) e[1], k -> new long[2]);
                t[0] += (Long) e[0];
                t[1]++;
            }
            peakSites = now;
            peakSitesAt = inUse;
        }
    }

    /** With {@link #SITES}: the sites that held the most memory at the last recorded peak. */
    String sitesReport() {
        if (!SITES) {
            return "";
        }
        StringBuilder b = new StringBuilder("\nmemory at " + (peakSitesAt >> 20) + " MB in use, by where it was allocated:");
        peakSites.entrySet().stream().sorted((x, y) -> Long.compare(y.getValue()[0], x.getValue()[0])).limit(20)
                .forEach(e -> b.append(String.format(java.util.Locale.ROOT, "%n  %8d MB in %6d blocks  %s",
                        e.getValue()[0] >> 20, e.getValue()[1], e.getKey())));
        return b.toString();
    }

    private ArrayDeque<Long> freeList(long size) {
        return (size <= SLAB_LIMIT ? smallPool : pool).get(size);
    }

    /** The bytes of a slab cut into blocks of {@code size}: 64 of them, within 1 to 8 MB. */
    static long slabBytes(long size) {
        long bytes = Math.min(8L << 20, Math.max(1L << 20, size * 64));
        return bytes / size * size;
    }

    /** A new slab cut into blocks of {@code size}: returns the first, pools the others. */
    private long cutSlab(long size) {
        long bytes = slabBytes(size);
        trace("hipMalloc " + bytes + " bytes (a slab of " + (bytes / size) + " blocks of " + size + ")");
        long t0 = PROFILE ? System.nanoTime() : 0;
        long base = allocateFromDriver(bytes);
        record("hipMalloc (slab)", PROFILE ? System.nanoTime() - t0 : 0);
        reserved += bytes;
        slabs.put(base, bytes);
        ArrayDeque<Long> free = smallPool.computeIfAbsent(size, x -> new ArrayDeque<>());
        for (long at = bytes - size; at > 0; at -= size) {
            free.push(base + at);
        }
        return base;
    }

    /** Bytes taken from the driver since the garbage collector was last asked for released tensors. */
    private long newSinceCollection;

    /**
     * Before taking new memory from the driver: every so often (128 MB, or a sixteenth of the
     * budget) asks the garbage collector for tensors no longer used, as the Vulkan backend
     * does. Memory returns to the pool only once their tensors are collected, which takes long
     * when little Java memory is allocated: a 7B model's prompt otherwise kept every layer's
     * intermediates, 6 GB, until the budget was reached. Returns whether it collected.
     */
    private boolean collectSoon(long size) {
        newSinceCollection += size;
        if (newSinceCollection <= Math.min(128L << 20, budget / 16)) {
            return false;
        }
        newSinceCollection = 0;
        collections++;
        long t0 = PROFILE ? System.nanoTime() : 0;
        System.gc();
        drainCollected();
        record("garbage collection for memory", PROFILE ? System.nanoTime() - t0 : 0);
        return true;
    }

    /** Whether {@code ptr} lies in a slab, and so is freed with it. */
    private boolean inSlab(long ptr) {
        Map.Entry<Long, Long> e = slabs.floorEntry(ptr);
        return e != null && ptr < e.getKey() + e.getValue();
    }

    private long lastCollection;

    /**
     * Makes room within the budget: frees pooled blocks, and asks the garbage collector for
     * tensors no longer used. On Windows HIP does not fail when the card is full but moves
     * memory to system memory, where every kernel reads it over PCIe, so the budget has to be
     * kept here. Returns a block of {@code size} the collection gave back, or 0.
     */
    private long reclaim(long size) {
        trim();
        for (int attempt = 0; reserved + size > budget && attempt < 3; attempt++) {
            long now = System.nanoTime();
            if (attempt == 0 && now - lastCollection < 50_000_000L) {
                break;   // collected a moment ago: the rest is in use
            }
            lastCollection = now;
            collections++;
            System.gc();
            try {
                Thread.sleep(5L << attempt);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } finally {
                bind();   // a virtual thread may continue on another carrier thread
            }
            drainCollected();
            ArrayDeque<Long> free = freeList(size);
            if (free != null && !free.isEmpty()) {
                return free.pop();
            }
            trim();
        }
        return 0;
    }

    private long allocateFromDriver(long size) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(JAVA_LONG);
            int r = cu.allocate(size, out);
            if (r == HipRuntime.HIP_ERROR_OUT_OF_MEMORY) {
                // free what the pool holds, then what the garbage collector can find
                trim();
                r = cu.allocate(size, out);
                for (int attempt = 0; r == HipRuntime.HIP_ERROR_OUT_OF_MEMORY && attempt < 3; attempt++) {
                    collections++;
                    System.gc();
                    try {
                        Thread.sleep(20L << attempt);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    } finally {
                        bind();   // a virtual thread may continue on another carrier thread
                    }
                    drainCollected();
                    trim();
                    r = cu.allocate(size, out);
                }
            }
            if (r != HipRuntime.HIP_SUCCESS) {
                throw cu.error(r, "allocating " + (size >> 20) + " MB of device memory (" + (inUse >> 20)
                        + " MB in use, " + (reserved >> 20) + " MB reserved, budget " + (budget >> 20) + " MB)");
            }
            return out.get(JAVA_LONG, 0);
        }
    }

    /** Returns memory to the pool; work queued later may use it. */
    void free(long ptr, long bytes) {
        if (live.remove(ptr)) {
            if (SITES) {
                sites.remove(ptr);
            }
            long size = sizeClass(bytes);
            inUse -= size;
            (size <= SLAB_LIMIT ? smallPool : pool).computeIfAbsent(size, s -> new ArrayDeque<>()).push(ptr);
        }
    }

    /** For the collector's thread: frees on the next allocation, on the thread that owns the context. */
    void freeLater(long ptr, long bytes) {
        collected.add(new long[] {ptr, bytes});
    }

    private void drainCollected() {
        long[] e;
        while ((e = collected.poll()) != null) {
            free(e[0], e[1]);
        }
    }

    /** Gives every pooled block back to the driver, after the device has finished with them. */
    void trim() {
        drainCollected();
        if (pool.values().stream().allMatch(ArrayDeque::isEmpty)) {
            return;
        }
        finish();
        for (Map.Entry<Long, ArrayDeque<Long>> e : pool.entrySet()) {
            for (Long ptr : e.getValue()) {
                trace("hipFree");
                cu.free(ptr);
                reserved -= e.getKey();
            }
            e.getValue().clear();
        }
    }

    /**
     * Gives everything back to the runtime at exit: memory, the rocBLAS handle, the modules and
     * the stream; {@link #resetDevice()} then resets the device. On Windows a process that left them to the runtime's
     * own exit handling hung in the driver and could not be killed.
     */
    void close() {
        bind();
        trace("closing: hipStreamSynchronize");
        cu.synchronize(stream);
        drainCollected();
        // every step goes on after an error, so the device is released as far as it can be
        int failed = 0;
        Set<Long> all = new HashSet<>(live);
        for (Map<Long, ArrayDeque<Long>> p : List.of(pool, smallPool)) {
            for (ArrayDeque<Long> free : p.values()) {
                all.addAll(free);
                free.clear();
            }
        }
        live.clear();
        all.removeIf(this::inSlab);
        all.addAll(slabs.keySet());
        slabs.clear();
        for (Long ptr : all) {
            if (cu.freeQuietly(ptr) != HipRuntime.HIP_SUCCESS) {
                failed++;
            }
        }
        trace("closing: freed " + all.size() + " blocks of device memory" + (failed > 0 ? ", " + failed + " refused" : ""));
        if (staging != null) {
            cu.freeHost(staging);
            staging = null;
        }
        for (int i = 0; i < UPLOAD_SLOTS; i++) {
            if (uploadBuffers[i] != null) {
                cu.freeHost(uploadBuffers[i]);
                uploadBuffers[i] = null;
            }
        }
        for (MemorySegment h : readbacks) {
            cu.freeHost(h);
        }
        readbacks.clear();
        if (rocblas != null) {
            trace("closing: rocblas_destroy_handle");
            rocblas.destroy();
        }
        for (Long m : modules.values()) {
            trace("closing: hipModuleUnload");
            cu.unloadModule(m);
        }
        modules.clear();
        functions.clear();
        trace("closing: hipStreamDestroy");
        cu.destroyStream(stream);
        trace("closed");
    }

    /**
     * Resets the device, for the whole process: only once every context on it is closed (see
     * {@code HipBackend.closeAllAtExit}).
     */
    void resetDevice() {
        cu.setDevice(ordinal);
        CURRENT.remove();
        trace("closing: hipDeviceReset of device " + ordinal);
        cu.resetDevice();
    }

    long inUse() {
        drainCollected();
        return inUse;
    }

    long peak() {
        return peak;
    }

    long reserved() {
        return reserved;
    }

    void resetPeak() {
        drainCollected();
        peak = inUse;
    }
}
