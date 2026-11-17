package io.github.zzuegg.brainiac.backend.cuda;

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
 * One CUDA device in use: its context, the stream every operation is queued on, the compiled
 * kernels and a pool of device memory.
 *
 * <p>Everything runs in order on one stream, so the host only waits when it reads values back.
 * Memory returned to the pool is handed out again only to work queued later, which the stream
 * runs after everything that used it before.
 *
 * <p>With {@code -Dbrainiac.cuda.debug=true} the host waits for the device after every kernel,
 * copy and product, so an error is reported by the operation that caused it (with the kernel's
 * name and launch size) instead of by a later one; kernels are also compiled with line
 * information for {@code compute-sanitizer}.
 */
final class CudaContext {

    static final boolean DEBUG = Boolean.getBoolean("brainiac.cuda.debug");
    static final int GROUP = 256;
    /** Most blocks a one-dimensional launch uses; the kernels loop over the rest. */
    private static final int MAX_BLOCKS = 1 << 20;

    final CudaDriver cu;
    final Nvrtc nvrtc;
    final Cublas cublas;
    final String cublasUnavailable;
    final int device;
    final String deviceName;
    final int computeCapability;
    final int multiprocessors;
    final long deviceMemory;
    final long budget;
    final int architecture;
    final long compileNanos;
    final boolean cachedKernels;
    private final long context;
    final long stream;
    private final long module;
    private final Map<String, Long> functions = new HashMap<>();

    // kernel arguments: values, and the void*[] pointing at each, reused for every launch
    private final MemorySegment argValues;
    private final MemorySegment argPointers;
    private int argOffset;
    private int argCount;

    // page-locked memory for reading values back
    private MemorySegment staging;

    // statistics
    long launches;
    long products;
    long waits;
    long uploadedBytes;
    long downloadedBytes;

    /** The driver and NVRTC, loaded once for every device. */
    private static CudaDriver sharedDriver;
    private static Nvrtc sharedNvrtc;

    static synchronized CudaDriver driver() {
        if (sharedDriver == null) {
            CudaDriver d = new CudaDriver();
            d.init();
            sharedDriver = d;
        }
        return sharedDriver;
    }

    private static synchronized Nvrtc nvrtc() {
        if (sharedNvrtc == null) {
            sharedNvrtc = new Nvrtc();
        }
        return sharedNvrtc;
    }

    /** The context current on each thread, as far as this backend knows (NCCL may change it). */
    private static final ThreadLocal<CudaContext> CURRENT = new ThreadLocal<>();

    /** Forgets which context is current on this thread, after a library that may switch devices. */
    static void forgetCurrent() {
        CURRENT.remove();
    }

    final int ordinal;

    /**
     * Opens a device.
     *
     * @param ordinal its index among the driver's devices
     */
    CudaContext(int ordinal) {
        cu = driver();
        int count = cu.deviceCount();
        if (count == 0) {
            throw new CudaException("the NVIDIA driver found no CUDA device", 0);
        }
        if (ordinal < 0 || ordinal >= count) {
            throw new CudaException("device " + ordinal + " asked for (-Dbrainiac.cuda.device) but there are " + count
                    + " devices", 0);
        }
        this.ordinal = ordinal;
        device = cu.device(ordinal);
        deviceName = cu.deviceName(device);
        computeCapability = cu.attribute(device, CudaDriver.ATTR_COMPUTE_CAPABILITY_MAJOR) * 10
                + cu.attribute(device, CudaDriver.ATTR_COMPUTE_CAPABILITY_MINOR);
        multiprocessors = cu.attribute(device, CudaDriver.ATTR_MULTIPROCESSOR_COUNT);
        deviceMemory = cu.totalMemory(device);
        context = cu.primaryContext(device);
        cu.setCurrent(context);
        CURRENT.set(this);
        stream = cu.createStream(CudaDriver.CU_STREAM_NON_BLOCKING);

        long mb = Long.getLong("brainiac.cuda.memory", 0);
        long free = cu.memoryInfo()[0];
        budget = mb > 0 ? mb << 20 : Math.max(256L << 20, free - Math.max(512L << 20, deviceMemory / 16));

        nvrtc = nvrtc();
        architecture = nvrtc.architecture(computeCapability);
        long start = System.nanoTime();
        boolean[] cached = new boolean[1];
        byte[] ptx = kernels(cached);
        cachedKernels = cached[0];
        compileNanos = System.nanoTime() - start;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment image = a.allocate(ptx.length + 1L);
            MemorySegment.copy(ptx, 0, image, JAVA_BYTE, 0, ptx.length);
            module = cu.loadModule(image);
        }

        Arena global = Arena.global();
        argValues = global.allocate(4096, 16);
        argPointers = global.allocate(ADDRESS, 64);

        Cublas blas = null;
        String why = null;
        if ("none".equals(System.getProperty("brainiac.cuda.cublas"))) {
            why = "turned off (-Dbrainiac.cuda.cublas=none)";
        } else {
            try {
                blas = new Cublas(stream, Boolean.getBoolean("brainiac.cuda.tf32"));
            } catch (CudaException | LinkageError | IllegalArgumentException e) {
                why = e.getMessage();
            }
        }
        cublas = blas;
        cublasUnavailable = why;
    }

    /** Makes the context current on the calling thread, if it is not already. */
    void bind() {
        if (CURRENT.get() != this) {
            cu.setCurrent(context);
            CURRENT.set(this);
        }
    }

    // ------------------------------------------------------------------ kernels

    /**
     * The PTX of kernels.cu for this device: from the cache folder when it was compiled
     * before by the same NVRTC for the same architecture, otherwise compiled now (a few
     * seconds) and stored there. {@code -Dbrainiac.cuda.cache} names the folder (default
     * {@code ~/.cache/brainiac/cuda}, {@code none} for no cache).
     */
    private byte[] kernels(boolean[] cached) {
        String source;
        try (InputStream in = CudaContext.class.getResourceAsStream("kernels.cu")) {
            if (in == null) {
                throw new CudaException("kernels.cu is missing from the brainiac-backend-cuda module", 0);
            }
            source = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new CudaException("reading kernels.cu: " + e.getMessage(), e);
        }
        List<String> options = new ArrayList<>(List.of("--gpu-architecture=compute_" + architecture, "--std=c++17"));
        if (DEBUG) {
            options.add("--generate-line-info");
        }
        String nvrtcVersion = nvrtc.version();
        Path file = cacheFile(source + "\n" + options + "\n" + nvrtcVersion + "\n" + nvrtc.library);
        if (file != null && Files.isRegularFile(file)) {
            try {
                byte[] ptx = Files.readAllBytes(file);
                cached[0] = true;
                return ptx;
            } catch (IOException e) {
                // compiled again below
            }
        }
        Nvrtc.Ptx result = nvrtc.compile(source, "kernels.cu", options);
        if (DEBUG && !result.log().isBlank()) {
            System.err.println("brainiac cuda: NVRTC messages:\n" + result.log());
        }
        byte[] ptx = result.code();
        int end = ptx.length;
        while (end > 0 && ptx[end - 1] == 0) {
            end--;
        }
        ptx = java.util.Arrays.copyOf(ptx, end);
        if (file != null) {
            try {
                Files.createDirectories(file.getParent());
                Path tmp = Files.createTempFile(file.getParent(), "kernels", ".tmp");
                Files.write(tmp, ptx);
                Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException | UnsupportedOperationException e) {
                // no cache then
            }
        }
        return ptx;
    }

    private static Path cacheFile(String key) {
        String dir = System.getProperty("brainiac.cuda.cache");
        if ("none".equals(dir)) {
            return null;
        }
        Path folder = dir != null ? Path.of(dir)
                : Path.of(System.getProperty("user.home"), ".cache", "brainiac", "cuda");
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.UTF_8));
            return folder.resolve("kernels-" + HexFormat.of().formatHex(hash, 0, 12) + ".ptx");
        } catch (NoSuchAlgorithmException e) {
            return null;
        }
    }

    long function(String name) {
        Long f = functions.get(name);
        if (f == null) {
            f = cu.function(module, name);
            functions.put(name, f);
        }
        return f;
    }

    /** Starts the arguments of the next launch. */
    CudaContext args() {
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
    CudaContext ptr(long v) {
        argValues.set(JAVA_LONG, slot(8), v);
        return this;
    }

    CudaContext i64(long v) {
        return ptr(v);
    }

    CudaContext i32(int v) {
        argValues.set(JAVA_INT, slot(4), v);
        return this;
    }

    CudaContext f32(float v) {
        argValues.set(JAVA_FLOAT, slot(4), v);
        return this;
    }

    CudaContext f64(double v) {
        argValues.set(JAVA_DOUBLE, slot(8), v);
        return this;
    }

    /** A struct of 4-byte fields passed by value; a float field goes as its raw bits. */
    CudaContext ints(int... fields) {
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
    CudaContext strided(int rank, long[] shape, long[] s0, long[] s1) {
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
        long f = function(name);
        int r = cu.launch(f, gridX, gridY, gridZ, GROUP, 0, stream, argPointers);
        launches++;
        if (r != CudaDriver.CUDA_SUCCESS) {
            throw cu.error(r, "launching " + name + " (" + gridX + " x " + gridY + " x " + gridZ + " blocks)");
        }
        if (DEBUG) {
            debugWait("kernel " + name + " (" + gridX + " x " + gridY + " x " + gridZ + " blocks)");
        }
    }

    /** In debug mode: waits for the device and reports an error as caused by {@code what}. */
    void debugWait(String what) {
        int r = cu.synchronize(stream);
        if (r != CudaDriver.CUDA_SUCCESS) {
            throw cu.error(r, what);
        }
    }

    /** Waits until the device has run everything queued. */
    void finish() {
        waits++;
        int r = cu.synchronize(stream);
        if (r != CudaDriver.CUDA_SUCCESS) {
            throw cu.error(r, "waiting for the device (run with -Dbrainiac.cuda.debug=true to find the operation)");
        }
    }

    // ------------------------------------------------------------------ copies

    void upload(long dst, MemorySegment src) {
        long bytes = src.byteSize();
        if (bytes == 0) {
            return;
        }
        uploadedBytes += bytes;
        if (src.isNative()) {
            cu.copyToDevice(dst, src, bytes, stream);
        } else {
            // a Java array: copied off the heap first. The driver has taken the values when
            // the call returns (pageable memory), so the copy can go right away.
            try (Arena a = Arena.ofConfined()) {
                MemorySegment off = a.allocate(bytes, 8);
                MemorySegment.copy(src, 0, off, 0, bytes);
                cu.copyToDevice(dst, off, bytes, stream);
            }
        }
        if (DEBUG) {
            debugWait("upload of " + bytes + " bytes");
        }
    }

    /** Copies device memory into {@code dst}, waiting for it. */
    void download(MemorySegment dst, long src) {
        long bytes = dst.byteSize();
        downloadedBytes += bytes;
        long chunk = Math.min(bytes, 64L << 20);
        if (staging == null || staging.byteSize() < chunk) {
            staging = cu.allocateHost(Math.max(chunk, 1L << 20));
        }
        for (long done = 0; done < bytes; done += chunk) {
            long n = Math.min(chunk, bytes - done);
            cu.copyToHost(staging, src + done, n, stream);
            finish();
            MemorySegment.copy(staging, 0, dst, done, n);
        }
    }

    void copy(long dst, long src, long bytes) {
        if (bytes > 0) {
            cu.copy(dst, src, bytes, stream);
            if (DEBUG) {
                debugWait("device copy of " + bytes + " bytes");
            }
        }
    }

    void fill32(long dst, int bits, long count) {
        if (count > 0) {
            cu.fill32(dst, bits, count, stream);
        }
    }

    void fill16(long dst, short bits, long count) {
        if (count > 0) {
            cu.fill16(dst, bits, count, stream);
        }
    }

    void fill8(long dst, byte bits, long count) {
        if (count > 0) {
            cu.fill8(dst, bits, count, stream);
        }
    }

    // ------------------------------------------------------------------ memory

    /** Freed allocations from the collector's thread, returned to the pool by the next allocation. */
    private final ConcurrentLinkedQueue<long[]> collected = new ConcurrentLinkedQueue<>();
    private final Map<Long, ArrayDeque<Long>> pool = new HashMap<>();
    private final Set<Long> live = new HashSet<>();
    private long inUse;
    private long peak;
    private long reserved;
    long collections;

    /** Allocation sizes: multiples of 256 bytes up to 4 KB, then four steps per power of two. */
    static long sizeClass(long bytes) {
        long b = Math.max(256, (bytes + 255) & ~255L);
        if (b <= 4096) {
            return b;
        }
        int shift = 63 - Long.numberOfLeadingZeros(b) - 2;
        long step = 1L << shift;
        return (b + step - 1) & -step;
    }

    /** Device memory of at least {@code bytes}, from the pool when it has a block of that size. */
    long allocate(long bytes) {
        drainCollected();
        long size = sizeClass(bytes);
        ArrayDeque<Long> free = pool.get(size);
        long ptr;
        if (free != null && !free.isEmpty()) {
            ptr = free.pop();
        } else {
            if (reserved + size > budget) {
                trim();
            }
            ptr = allocateFromDriver(size);
            reserved += size;
        }
        live.add(ptr);
        inUse += size;
        peak = Math.max(peak, inUse);
        return ptr;
    }

    private long allocateFromDriver(long size) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(JAVA_LONG);
            int r = cu.allocate(size, out);
            if (r == CudaDriver.CUDA_ERROR_OUT_OF_MEMORY) {
                // free what the pool holds, then what the garbage collector can find
                trim();
                r = cu.allocate(size, out);
                for (int attempt = 0; r == CudaDriver.CUDA_ERROR_OUT_OF_MEMORY && attempt < 3; attempt++) {
                    collections++;
                    System.gc();
                    try {
                        Thread.sleep(20L << attempt);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                    drainCollected();
                    trim();
                    r = cu.allocate(size, out);
                }
            }
            if (r != CudaDriver.CUDA_SUCCESS) {
                throw cu.error(r, "allocating " + (size >> 20) + " MB of device memory (" + (inUse >> 20)
                        + " MB in use, " + (reserved >> 20) + " MB reserved, budget " + (budget >> 20) + " MB)");
            }
            return out.get(JAVA_LONG, 0);
        }
    }

    /** Returns memory to the pool; work queued later may use it. */
    void free(long ptr, long bytes) {
        if (live.remove(ptr)) {
            long size = sizeClass(bytes);
            inUse -= size;
            pool.computeIfAbsent(size, s -> new ArrayDeque<>()).push(ptr);
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
                cu.free(ptr);
                reserved -= e.getKey();
            }
            e.getValue().clear();
        }
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
