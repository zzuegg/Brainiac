package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.PackedStorage;
import io.github.zzuegg.brainiac.tensor.spi.QuantizedRows;
import io.github.zzuegg.brainiac.tensor.spi.RowQuantizedStorage;
import io.github.zzuegg.brainiac.tensor.spi.RawValues;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A GPU backend on Vulkan compute shaders, called through LWJGL. Works with any Vulkan 1.2
 * driver that supports buffer device addresses and 64-bit shader integers, which covers
 * current AMD, NVIDIA and Intel GPUs.
 *
 * <p>Stores float32 tensors, and float64, float16 and bfloat16 where the device supports them
 * (doubles, and 16-bit storage buffers); float16 and bfloat16 are computed in float32, float64
 * in double precision. Operations are recorded into command buffers and submitted in
 * batches, so the host rarely waits for the device: only when values are read back.
 *
 * <p>System properties: {@code brainiac.vulkan.device} picks a device by index (the list is
 * in {@link #status()}); {@code brainiac.vulkan.devices=all} also allows CPU drivers such as
 * lavapipe; {@code brainiac.vulkan.memory} sets the memory budget in MB;
 * {@code brainiac.vulkan.fusion=false} runs element-wise steps one by one instead of fusing them
 * ({@link Fusion}); {@code brainiac.vulkan.cache} is where generated shaders are kept between runs
 * (default {@code ~/.cache/brainiac/vulkan}, {@code none} for nowhere);
 * {@code brainiac.vulkan.statistics=true} prints {@link #statistics()} at exit.
 */
public final class VulkanBackend implements Backend {

    private static final Cleaner CLEANER = Cleaner.create();
    /** Axes a single strided dispatch handles; higher ranks loop over the outer axes. */
    private static final int MAX_AXES = 6;
    private static final int GROUP = 256;

    private VkContext vk;
    private long initNanos;
    private boolean initialized;
    private String unavailable;
    private final Map<String, Tables> tables = new LinkedHashMap<>(16, 0.75f, true) {
        private static final long serialVersionUID = 1L;

        @Override
        protected boolean removeEldestEntry(Map.Entry<String, Tables> eldest) {
            return size() > 32;
        }
    };

    private record Tables(VkStorage im2col, VkStorage offsets, VkStorage entries) {
    }

    private static VulkanBackend first;

    /**
     * The first instance that found a device, normally the one behind the {@code vulkan}
     * {@link Device}; for {@link #statistics()}.
     *
     * @return the instance, if any
     */
    public static synchronized java.util.Optional<VulkanBackend> loaded() {
        return java.util.Optional.ofNullable(first);
    }

    /** Creates the backend; called by {@link java.util.ServiceLoader}. */
    public VulkanBackend() {
    }

    @Override
    public String name() {
        return vk == null ? "vulkan" : "vulkan (" + vk.deviceName + ")";
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
                long start = System.nanoTime();
                vk = new VkContext();
                initNanos = System.nanoTime() - start;
                if (FUSION) {
                    try {
                        compiler = new ShaderCompiler();
                    } catch (RuntimeException | LinkageError e) {
                        fusionUnavailable = e.getMessage() == null ? e.toString() : e.getMessage();
                    }
                }
                synchronized (VulkanBackend.class) {
                    if (first == null) {
                        first = this;
                    }
                }
                boolean statistics = Boolean.getBoolean("brainiac.vulkan.statistics");
                boolean teardown = !"false".equals(System.getProperty("brainiac.vulkan.teardown"));
                if (statistics || teardown) {
                    VkContext context = vk;
                    Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                        synchronized (VulkanBackend.this) {
                            if (statistics && vk == context && context.dispatches > 0) {
                                System.err.println(name() + ": " + statistics());
                            }
                            if (teardown && vk == context) {
                                vk = null;   // later calls fail plainly instead of using a destroyed device
                                try {
                                    context.shutdown();
                                } catch (RuntimeException | LinkageError e) {
                                    // exiting anyway
                                }
                            }
                        }
                    }, "vulkan-exit"));
                }
                String mode = System.getProperty("brainiac.vulkan.tune", "auto");
                if (!mode.equals("false") && !mode.equals("off")) {
                    // forced once per run: further instances read what the first measured
                    loadTuning(mode.equals("force") && FORCED.compareAndSet(false, true));
                }
            } catch (RuntimeException | LinkageError e) {
                unavailable = e.getMessage() == null ? e.toString() : e.getMessage();
                vk = null;
            }
        }
        return vk != null;
    }

    /**
     * Describes the device in use, or why none is.
     *
     * @return for example {@code "AMD Radeon RX 9070 XT, 16304 MB, 13043 MB budget; devices 0: ..."}
     */
    public synchronized String status() {
        if (!isAvailable()) {
            return "unavailable: " + unavailable;
        }
        return vk.deviceName + ", " + (vk.deviceMemory >> 20) + " MB, " + (vk.budget >> 20) + " MB budget, fusion "
                + (compiler != null ? "on" : FUSION ? "unavailable (" + fusionUnavailable + ")" : "off")
                + (vk.coopMatrix ? ", cooperative matrices" : "") + "; devices "
                + vk.deviceList;
    }

    @Override
    public synchronized Device.Memory memory() {
        if (vk == null) {
            return null;
        }
        long now = vk.inUse();
        return new Device.Memory(now, vk.peakInUse(), vk.allocated(), vk.budget);
    }

    @Override
    public synchronized void resetPeakMemory() {
        if (vk != null) {
            vk.resetPeak();
        }
    }

    /**
     * Counts of work done so far, for tuning.
     *
     * @return dispatches, barriers, submits, host waits and memory use
     */
    public synchronized String statistics() {
        if (vk == null) {
            return "not initialized";
        }
        return "init " + initNanos / 1_000_000 + " ms, " + (vk.uploaded >> 20) + " MB uploaded in "
                + vk.uploadNanos / 1_000_000 + " ms, allocating " + vk.allocNanos / 1_000_000 + " ms, pipelines "
                + vk.pipelineNanos / 1_000_000 + " ms, waiting " + vk.waitNanos / 1_000_000 + " ms; "
                + vk.dispatches + " dispatches (" + fusedDispatches + " fused, " + fusedShaders.size()
                + " shaders generated in " + compileNanos / 1_000_000 + " ms), " + vk.barriers + " barriers, " + vk.submits + " submits, " + vk.waits
                + " waits, " + vk.finishes + " full waits, " + vk.reclaims + " reclaims, " + vk.collections + " collections, " + (vk.allocated() >> 20) + " MB allocated, " + (vk.spilled() >> 20) + " MB spilled, " + (vk.peakInUse() >> 20) + " MB peak in use; most dispatched: "
                + vk.perShader.entrySet().stream().sorted(java.util.Map.Entry.<String, Long>comparingByValue().reversed())
                        .limit(12).map(e -> e.getKey() + " " + e.getValue()).collect(java.util.stream.Collectors.joining(", "))
                + (VkContext.PROFILE ? profile() : "")
                + (VkContext.TRACE_FINISH ? "; full waits from: " + vk.finishCallers.entrySet().stream()
                        .sorted(java.util.Map.Entry.<String, Long>comparingByValue().reversed()).limit(10)
                        .map(e -> e.getKey() + " " + e.getValue()).collect(java.util.stream.Collectors.joining(", ")) : "");
    }

    /** Device time per shader ({@code -Dbrainiac.vulkan.profile=true}): total, then the largest. */
    private String profile() {
        long total = vk.shaderNanos.values().stream().mapToLong(Long::longValue).sum();
        return String.format(java.util.Locale.ROOT, "; device time %.1f ms: ", total / 1e6)
                + vk.shaderNanos.entrySet().stream()
                        .sorted(java.util.Map.Entry.<String, Long>comparingByValue().reversed()).limit(16)
                        .map(e -> {
                            long n = vk.shaderCalls.getOrDefault(e.getKey(), 1L);
                            return String.format(java.util.Locale.ROOT, "%s %.1f ms (%.0f%%, %d x %.1f us)", e.getKey(),
                                    e.getValue() / 1e6, 100.0 * e.getValue() / Math.max(1, total), n,
                                    e.getValue() / 1e3 / Math.max(1, n));
                        })
                        .collect(java.util.stream.Collectors.joining(", "))
                + (fusedShaders.isEmpty() ? "" : "; generated shaders in " + FUSED_DUMP);
    }

    /** Where {@code -Dbrainiac.vulkan.profile=true} writes the generated shaders, by name. */
    private static final java.nio.file.Path FUSED_DUMP = java.nio.file.Path.of(System.getProperty("java.io.tmpdir"),
            "brainiac-fused");

    /** Tests: new device-local blocks fail from now on, so memory spills; false lifts that. */
    synchronized void simulateFullDevice(boolean full) {
        if (full) {
            vk.reclaim();   // pooled blocks freed, so new tensors need new blocks
        }
        vk.simulatedLimit = full ? Math.max(1, vk.allocated()) : 0;
    }

    /** @return bytes spilled from device-local memory; for tests */
    synchronized long spilled() {
        return vk == null ? 0 : vk.spilled();
    }

    /** Commands recorded so far, fused or not; for tests. */
    synchronized long dispatchCount() {
        return vk == null ? 0 : vk.dispatches;
    }

    /**
     * Uses the kept tuning for this device, or measures one (a few seconds) when there is none
     * and the device is not the CPU, or always with {@code force}.
     */
    private static final java.util.concurrent.atomic.AtomicBoolean FORCED = new java.util.concurrent.atomic.AtomicBoolean();

    private void loadTuning(boolean force) {
        java.nio.file.Path file = Tuning.file(vk.deviceKey);
        Tuning kept = force ? null : Tuning.load(file);
        if (kept != null && (kept.complete || vk.softwareDevice)) {
            tuning = kept;
            tuningNote = "tuned (" + file + ")";
            return;
        }
        if (vk.softwareDevice && !force) {
            return;
        }
        try {
            tune(file);
        } catch (RuntimeException e) {
            tuning = new Tuning();
            tuningNote = "defaults (tuning failed: " + e + ")";
            System.err.println("brainiac: tuning the Vulkan kernels failed, using defaults: " + e);
        }
    }

    private void tune(java.nio.file.Path file) {
        System.err.println("brainiac: tuning the Vulkan kernels for " + vk.deviceName + " (once per device and driver)...");
        long start = System.nanoTime();
        Tuning t = new Tuning();
        tuning = t;
        String log = new Tuner(this, t).run(vk.coopMatrix);
        double seconds = (System.nanoTime() - start) / 1e9;
        if (Boolean.getBoolean("brainiac.vulkan.tuneLog")) {
            System.err.print(log);
        }
        try {
            t.save(file, vk.deviceName + ", tuned in " + Math.round(seconds) + " s; delete or run with "
                    + "-Dbrainiac.vulkan.tune=force to measure again");
            tuningNote = "tuned now (" + file + ")";
        } catch (java.io.UncheckedIOException e) {
            tuningNote = "tuned now, not kept: " + e.getMessage();
        }
        System.err.printf(java.util.Locale.ROOT, "brainiac: tuned in %.1f s: %s%n", seconds, t);
    }

    /**
     * Measures this device's kernel choices again and keeps them, as {@code -Dbrainiac.vulkan.tune=force} does.
     *
     * @return the choices
     */
    public synchronized String retune() {
        if (!isAvailable()) {
            return "unavailable: " + unavailable;
        }
        tune(Tuning.file(vk.deviceKey));
        return tuning.toString();
    }

    /** @return the kernel choices in use and where they come from */
    public synchronized String tuning() {
        return isAvailable() ? tuning + "; " + tuningNote : "unavailable: " + unavailable;
    }

    /** Waits until all recorded work is done; for timing. */
    public synchronized void finish() {
        if (vk != null) {
            vk.finish();
        }
    }


    // ------------------------------------------------------------------ fusion

    /** Whether element-wise chains are fused ({@code -Dbrainiac.vulkan.fusion=false} runs every step on its own). */
    private static final boolean FUSION = !"false".equalsIgnoreCase(System.getProperty("brainiac.vulkan.fusion"));
    /** Compiles generated shaders; null when fusion is off or shaderc could not be loaded. */
    private ShaderCompiler compiler;
    private String fusionUnavailable;
    /** Pipeline names of generated shaders, by generated source (and format). */
    private final Map<String, String> fusedShaders = new HashMap<>();
    /**
     * Counts commands recorded for other operations. A pending result recorded before the
     * latest of them is computed before anything builds on it, so chains end at the same
     * points in every training step and their generated shaders repeat instead of growing.
     */
    private long generation;
    private long fusedDispatches;
    private long compileNanos;

    /** Whether 16-bit chains fuse too ({@code -Dbrainiac.vulkan.fuse16=false} rounds after every step). */
    private static final boolean FUSE_16 = !"false".equalsIgnoreCase(System.getProperty("brainiac.vulkan.fuse16"));

    /** Whether an element-wise result can be recorded instead of computed now. */
    private boolean fusable(VkStorage out, Storage... inputs) {
        // float16 and bfloat16 chains compute in float32 and round once, at the end
        if (compiler == null || out.size == 0 || !FUSE_16 && (out.dtype == DType.FLOAT16 || out.dtype == DType.BFLOAT16)) {
            return false;
        }
        for (Storage in : inputs) {
            if (vs(in).dtype != out.dtype) {
                return false;
            }
        }
        return true;
    }

    /** Records {@code e} as the pending value of {@code out}, computing inputs first where it pays. */
    private void defer(VkStorage out, Fusion.Expr e) {
        // in place: the output's value so far is computed first and read from memory, and the
        // result is computed now, as the output cannot both stay pending and be read by it
        boolean inPlace = Fusion.reads(e, out);
        if (inPlace) {
            ready(out);
        }
        // a view adds no step of its own, so it cannot grow a chain
        boolean view = e instanceof Fusion.Src;
        for (Fusion.Src in : Fusion.sources(e)) {
            VkStorage st = in.storage();
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
        if (Fusion.cost(e) > Fusion.MAX_NODES
                || out.dtype == DType.FLOAT64 && Fusion.transcendentals(e) > 1) {
            for (Fusion.Src in : Fusion.sources(e)) {
                ready(in.storage());
            }
        }
        out.pending = e;
        out.pendingNodes = Math.min(Fusion.nodes(e), 1 << 20);
        out.pendingGeneration = generation;
        if (inPlace) {
            ready(out);
        }
    }

    /** Pending values that cost nothing to compute again wherever they are read. */
    private static boolean cheap(Fusion.Expr e) {
        return e instanceof Fusion.Const || e instanceof Fusion.Random
                || e instanceof Fusion.Src s && s.storage().pending == null;
    }

    /** Computes pending values a command recorded now reads or writes; starts a new fusion generation. */
    private void eager(VkStorage... storages) {
        for (VkStorage s : storages) {
            ready(s);
        }
        generation++;
    }

    /** Computes a pending storage, so that its slot holds its values. */
    private void ready(VkStorage s) {
        Fusion.Expr e = s.pending;
        if (e == null) {
            return;
        }
        s.pending = null;
        s.pendingNodes = 0;
        if (e instanceof Fusion.Const c) {
            fillNow(s, c.value());
            return;
        }
        if (e instanceof Fusion.Src v && v.storage().pending == null && v.dims().length == 1
                && v.strides()[0] == 1 && v.storage().dtype == s.dtype) {
            // a contiguous range of stored values
            int bytes = s.dtype.bytes();
            vk.copy(v.storage().slot(), (long) v.offset() * bytes, s.slot(), 0, s.size * bytes);
            return;
        }
        Fusion.Kernel k = Fusion.generate(e, s.dtype);
        String name = fusedPipeline(k.source(), s.dtype);
        VkContext.Slot[] reads = new VkContext.Slot[k.loads().size()];
        for (int i = 0; i < reads.length; i++) {
            reads[i] = k.loads().get(i).slot();
        }
        if (k.inPush()) {
            ByteBuffer p = vk.push().putLong(s.address()).putInt((int) s.size).putInt(0);
            k.writeMeta(p, Fusion.Kernel.PUSH_TABLE);
            p.position(Fusion.Kernel.PUSH_TABLE + k.metaBytes());
        } else {
            long meta = vk.hostData(k.metaBytes(), k::writeMeta);
            vk.push().putLong(meta).putLong(s.address()).putInt((int) s.size);
        }
        vk.dispatch(name, groups(s.size), s.slot(), reads);
        fusedDispatches++;
    }

    /** The pipeline of a generated shader, compiled on first use. */
    private String fusedPipeline(String body, DType dtype) {
        // looked up by the generated part only: the shared prelude would make every lookup hash
        // and compare kilobytes, which costs more than a small dispatch saves
        String key = dtype == DType.FLOAT32 ? body : dtype.ordinal() + body;
        String name = fusedShaders.get(key);
        if (name == null) {
            long start = System.nanoTime();
            byte[] code = compiler.compile(compiler.source(body, dtype));
            name = "fused#" + fusedShaders.size();
            vk.register(name, code);
            if (VkContext.PROFILE) {
                try {
                    java.nio.file.Files.createDirectories(FUSED_DUMP);
                    java.nio.file.Files.writeString(FUSED_DUMP.resolve(name.replace('#', '_') + ".comp"), body);
                } catch (java.io.IOException e) {
                    // only a help for reading the profile
                }
            }
            compileNanos += System.nanoTime() - start;
            fusedShaders.put(key, name);
        }
        return name;
    }

    /**
     * Compiles a shader body (after the shared prelude) and creates its pipeline, without
     * running it; for finding which code a driver's compiler fails on.
     */
    synchronized void compileForDiagnosis(String body, DType type, boolean optimize) {
        if (!isAvailable() || compiler == null) {
            throw new IllegalStateException(status());
        }
        vk.register("diagnosis#" + fusedShaders.size() + "#" + System.nanoTime(),
                compiler.compile(compiler.source(body, type), optimize));
    }

    @Override
    public synchronized void evaluate(List<Storage> storages) {
        for (Storage s : storages) {
            ready(vs(s));
        }
    }

    // ------------------------------------------------------------------ storage

    private VkStorage raw(long size, DType dtype, float[] host) {
        if (size > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("the vulkan backend holds at most 2^31 values per tensor");
        }
        isAvailable();
        if (!supports(dtype)) {
            throw new UnsupportedOperationException("the " + name() + " backend does not support " + dtype
                    + " tensors on this device");
        }
        return new VkStorage(this, size, dtype, host, CLEANER, vk);
    }

    private VkStorage raw(long size, float[] host) {
        return raw(size, DType.FLOAT32, host);
    }

    private VkStorage vs(Storage s) {
        if (s instanceof VkStorage v && v.owner == this) {
            return v;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to " + name());
    }

    /** The shader compiled for a number format. */
    private static String shader(String name, DType dtype) {
        return name + switch (dtype) {
            case FLOAT32 -> "_f32";
            case FLOAT64 -> "_f64";
            case FLOAT16 -> "_f16";
            case BFLOAT16 -> "_bf16";
            default -> throw new IllegalArgumentException("no Vulkan shaders for " + dtype);
        };
    }

    /**
     * The shader for one operation of a family: float64 has one per operation, as AMD's LLPC
     * compiler crashes on a float64 shader that switches over all of them.
     */
    private static String perOp(String name, DType dtype, int op) {
        return dtype == DType.FLOAT64 ? shader(name, dtype) + "_" + op : shader(name, dtype);
    }

    /**
     * {@code s} in another format: itself if it already is, else a converted copy that the
     * caller releases (with {@link #done}) once the command that reads it is recorded.
     */
    private VkStorage as(VkStorage s, DType dtype) {
        if (s.dtype == dtype) {
            return s;
        }
        VkStorage c = raw(s.size, dtype, null);
        convert(s, c);
        ready(c);   // read directly by the command that asked for it
        return c;
    }

    /** Releases conversions made by {@link #as}: those that differ from the originals. */
    private static void done(VkStorage converted, VkStorage original) {
        if (converted != original) {
            converted.release();
        }
    }

    @Override
    public boolean supports(DType dtype) {
        if (dtype == DType.FLOAT32) {
            return true;
        }
        if (!isAvailable()) {
            return false;
        }
        return switch (dtype) {
            case FLOAT64 -> vk.fp64;
            case FLOAT16 -> vk.storage16;
            case BFLOAT16 -> vk.storage16 && vk.int16;
            default -> false;
        };
    }

    @Override
    public synchronized Storage allocate(long size) {
        return allocate(size, DType.FLOAT32);
    }

    @Override
    public synchronized Storage allocate(long size, DType dtype) {
        VkStorage s = raw(size, dtype, null);
        if (fusable(s)) {
            constant(s, 0);
        } else {
            // zero bits are 0.0 in every format; fills cover whole 32-bit words, inside the slot
            vk.fill(s.slot(), 0, (s.bytes() + 3) & ~3L, 0);
        }
        return s;
    }

    @Override
    public synchronized Storage allocateUninitialized(long size, DType dtype) {
        return raw(size, dtype, null);
    }

    @Override
    public synchronized void release(Storage storage) {
        if (storage instanceof QRows r) {
            r.levels().release();
            r.scales().release();
            return;
        }
        if (storage instanceof QuantizedRows) {
            return;
        }
        if (storage instanceof Deferred) {
            return;   // uploaded weights: freed with the tensor that holds them
        }
        if (storage instanceof Packed p) {
            if (p.host() == null) {   // quantized here, such as optimizer state: freed now
                p.levels().release();
                p.scales().release();
            }
            return;   // uploaded weights: freed with the tensor that holds them
        }
        VkStorage s = vs(storage);
        if (s.pending == null) {
            s.release();   // pending results may still be inlined elsewhere: left to the collector
        }
    }

    @Override
    public synchronized Storage upload(float[] data) {
        VkStorage s = raw(data.length, data.length <= 4096 ? data.clone() : null);
        if (java.nio.ByteOrder.nativeOrder() == java.nio.ByteOrder.LITTLE_ENDIAN) {
            vk.upload(s.slot(), java.lang.foreign.MemorySegment.ofArray(data));
        } else {
            vk.upload(s.slot(), s.bytes(), b -> b.asFloatBuffer().put(data));
        }
        return s;
    }

    @Override
    public synchronized Storage upload(double[] data, DType dtype) {
        if (dtype == DType.FLOAT32) {
            float[] f = new float[data.length];
            for (int i = 0; i < f.length; i++) {
                f[i] = (float) data[i];
            }
            return upload(f);
        }
        VkStorage s = raw(data.length, dtype, null);
        vk.upload(s.slot(), s.bytes(), b -> {
            switch (dtype) {
                case FLOAT64 -> b.asDoubleBuffer().put(data);
                case FLOAT16 -> {
                    var h = b.asShortBuffer();
                    for (double v : data) {
                        h.put(Float.floatToFloat16((float) v));
                    }
                }
                default -> {
                    var h = b.asShortBuffer();
                    for (double v : data) {
                        h.put(DType.floatToBfloat16((float) v));
                    }
                }
            }
        });
        return s;
    }

    @Override
    public synchronized Storage upload(java.lang.foreign.MemorySegment data, DType dtype) {
        if (!RawValues.nativeLittleEndian()) {
            return Backend.super.upload(data, dtype);
        }
        long n = data.byteSize() / dtype.bytes();
        VkStorage s = raw(n, dtype, null);
        vk.upload(s.slot(), data);
        return s;
    }

    private VkStorage uploadInts(int[] data) {
        VkStorage s = raw(Math.max(1, data.length), null);
        vk.upload(s.slot(), (long) data.length * Integer.BYTES, b -> b.asIntBuffer().put(data));
        return s;
    }

    @Override
    public synchronized float[] download(Storage src) {
        VkStorage s = vs(src);
        eager(s);
        float[] out = new float[Math.toIntExact(s.size)];
        if (s.dtype == DType.FLOAT32) {
            vk.download(s.slot(), s.bytes()).asFloatBuffer().get(out);
        } else {
            double[] d = downloadDouble(s);
            for (int i = 0; i < out.length; i++) {
                out[i] = (float) d[i];
            }
        }
        return out;
    }

    @Override
    public synchronized java.util.function.Supplier<float[]> downloadLater(Storage src) {
        VkStorage s = vs(src);
        eager(s);
        if (s.dtype != DType.FLOAT32 || s.bytes() > VkContext.SMALL_READBACK) {
            float[] now = download(s);
            return () -> now;
        }
        VkContext.Pending p = vk.downloadLater(s.slot(), s.bytes());
        int n = Math.toIntExact(s.size);
        return new java.util.function.Supplier<>() {
            private float[] got;

            @Override
            public float[] get() {
                synchronized (VulkanBackend.this) {
                    if (got == null) {
                        got = new float[n];
                        vk.read(p).asFloatBuffer().get(got);
                        vk.release(p);
                    }
                    return got;
                }
            }
        };
    }

    @Override
    public synchronized double[] downloadDouble(Storage src) {
        VkStorage s = vs(src);
        eager(s);
        double[] out = new double[Math.toIntExact(s.size)];
        ByteBuffer b = vk.download(s.slot(), s.bytes());
        switch (s.dtype) {
            case FLOAT64 -> b.asDoubleBuffer().get(out);
            case FLOAT32 -> {
                var f = b.asFloatBuffer();
                for (int i = 0; i < out.length; i++) {
                    out[i] = f.get(i);
                }
            }
            case FLOAT16 -> {
                var h = b.asShortBuffer();
                for (int i = 0; i < out.length; i++) {
                    out[i] = Float.float16ToFloat(h.get(i));
                }
            }
            case BFLOAT16 -> {
                var h = b.asShortBuffer();
                for (int i = 0; i < out.length; i++) {
                    out[i] = DType.bfloat16ToFloat(h.get(i));
                }
            }
        }
        return out;
    }

    @Override
    public synchronized void convert(Storage in, Storage out) {
        VkStorage x = vs(in);
        VkStorage y = vs(out);
        if (x.dtype != y.dtype && x.dtype != DType.FLOAT64 && y.dtype != DType.FLOAT64 && fusable(y)
                && x.size == y.size) {
            // between float32, float16 and bfloat16: read in the other format by what uses it
            defer(y, Fusion.whole(x));
            return;
        }
        eager(x, y);
        if (x.dtype == y.dtype) {
            vk.copy(x.slot(), 0, y.slot(), 0, x.bytes());
        } else if (x.dtype == DType.FLOAT32) {
            vk.push().putLong(x.address()).putLong(y.address()).putInt((int) y.size);
            vk.dispatch(shader("from_float", y.dtype), groups(y.size), y.slot(), x.slot());
        } else if (y.dtype == DType.FLOAT32) {
            vk.push().putLong(x.address()).putLong(y.address()).putInt((int) y.size);
            vk.dispatch(shader("to_float", x.dtype), groups(y.size), y.slot(), x.slot());
        } else {
            // through float32, which is also how the CPU rounds between these formats
            VkStorage f = raw(x.size, null);
            convert(x, f);
            convert(f, y);
            f.release();
        }
    }

    @Override
    public synchronized void fill(Storage dst, float value) {
        VkStorage s = vs(dst);
        if (fusable(s)) {
            constant(s, value);
            return;
        }
        s.pending = null;   // overwritten
        fillNow(s, value);
    }

    private void constant(VkStorage s, float value) {
        s.pending = new Fusion.Const(value);
        s.pendingNodes = 1;
        s.pendingGeneration = generation;
    }

    private void fillNow(VkStorage s, float value) {
        switch (s.dtype) {
            case FLOAT32 -> vk.fill(s.slot(), 0, s.bytes(), Float.floatToRawIntBits(value));
            case FLOAT16, BFLOAT16 -> {
                int h = (s.dtype == DType.FLOAT16 ? Float.floatToFloat16(value) : DType.floatToBfloat16(value))
                        & 0xFFFF;
                // whole 32-bit words: an odd count writes one value past the end, inside the slot
                vk.fill(s.slot(), 0, (s.bytes() + 3) & ~3L, h | h << 16);
            }
            case FLOAT64 -> {
                if (Double.doubleToRawLongBits(value) == 0) {
                    vk.fill(s.slot(), 0, s.bytes(), 0);
                } else {
                    vk.push().putLong(s.address()).putInt((int) s.size).putFloat(value);
                    vk.dispatch(shader("fill", s.dtype), groups(s.size), s.slot());
                }
            }
        }
    }

    @Override
    public synchronized Storage random(long size, long seed) {
        VkStorage s = raw(size, null);
        if (fusable(s)) {
            // generated inside the shader that uses the values, never stored
            s.pending = new Fusion.Random(seed);
            s.pendingNodes = 1;
            s.pendingGeneration = generation;
            return s;
        }
        vk.push().putLong(s.address()).putLong(seed).putInt((int) size);
        vk.dispatch("random", groups(size), s.slot());
        return s;
    }

    private static long groups(long n) {
        return (n + GROUP - 1) / GROUP;
    }

    // ------------------------------------------------------------------ element-wise

    @Override
    public synchronized void unary(UnaryOp op, Storage in, Storage out) {
        VkStorage y = vs(out);
        if (fusable(y, in)) {
            defer(y, new Fusion.Un(op.ordinal(), Fusion.whole(vs(in))));
            return;
        }
        eager(vs(in), y);
        VkStorage x = as(vs(in), y.dtype);
        vk.push().putLong(x.address()).putLong(y.address()).putInt((int) y.size).putInt(op.ordinal());
        vk.dispatch(perOp("unary", y.dtype, op.ordinal()), groups(y.size), y.slot(), x.slot());
        done(x, vs(in));
    }

    @Override
    public synchronized void activation(ActivationOp op, float param, Storage in, Storage out) {
        VkStorage y = vs(out);
        if (fusable(y, in)) {
            defer(y, new Fusion.Act(op.ordinal(), param, Fusion.whole(vs(in))));
            return;
        }
        eager(vs(in), y);
        VkStorage x = as(vs(in), y.dtype);
        vk.push().putLong(x.address()).putLong(y.address()).putInt((int) y.size).putInt(op.ordinal()).putFloat(param);
        vk.dispatch(perOp("activation", y.dtype, op.ordinal()), groups(y.size), y.slot(), x.slot());
        done(x, vs(in));
    }

    @Override
    public synchronized void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        VkStorage y = vs(out);
        if (fusable(y, in, grad)) {
            defer(y, new Fusion.ActBack(op.ordinal(), param, Fusion.whole(vs(in)), Fusion.whole(vs(grad))));
            return;
        }
        eager(vs(in), vs(grad), y);
        VkStorage x = as(vs(in), y.dtype);
        VkStorage g = as(vs(grad), y.dtype);
        vk.push().putLong(x.address()).putLong(g.address()).putLong(y.address()).putInt((int) y.size)
                .putInt(op.ordinal()).putFloat(param);
        vk.dispatch(shader("activation_backward", y.dtype), groups(y.size), y.slot(), x.slot(), g.slot());
        done(x, vs(in));
        done(g, vs(grad));
    }

    @Override
    public synchronized void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape,
            Storage out) {
        VkStorage z = vs(out);
        long n = z.size;
        if (n == 0) {
            return;
        }
        if (fusable(z, a, b)) {
            Fusion.Src x = Fusion.strided(vs(a), 0, shape, aStrides);
            Fusion.Src y = Fusion.strided(vs(b), 0, shape, bStrides);
            if (x != null && y != null) {
                defer(z, new Fusion.Bin(op.ordinal(), x, y));
                return;
            }
        }
        eager(vs(a), vs(b), z);
        VkStorage x = as(vs(a), z.dtype);
        VkStorage y = as(vs(b), z.dtype);
        binary(op, x, aStrides, y, bStrides, shape, z, n);
        done(x, vs(a));
        done(y, vs(b));
    }

    private void binary(BinaryOp op, VkStorage x, long[] aStrides, VkStorage y, long[] bStrides, int[] shape,
            VkStorage z, long n) {
        long[] contiguous = contiguous(shape);
        if (Arrays.equals(aStrides, contiguous) && Arrays.equals(bStrides, contiguous)) {
            vk.push().putLong(x.address()).putLong(y.address()).putLong(z.address()).putInt((int) n)
                    .putInt(op.ordinal());
            vk.dispatch(shader("binary_contiguous", z.dtype), groups(n), z.slot(), x.slot(), y.slot());
            return;
        }
        Axes ax = Axes.merge(shape, aStrides, bStrides, contiguous);
        ax.forEachBlock((offsets, inner, count) -> {
            ByteBuffer p = vk.push().putLong(x.address(offsets[0])).putLong(y.address(offsets[1]))
                    .putLong(z.address(offsets[2])).putInt((int) count).putInt(op.ordinal()).putInt(inner)
                    .putInt(0).putInt(0);
            ax.put(p, inner);
            vk.dispatch(shader("binary", z.dtype), groups(count), z.slot(), x.slot(), y.slot());
        });
    }

    @Override
    public void copy(Storage in, long[] strides, int[] shape, Storage out) {
        copy(in, 0, strides, shape, out, 0, contiguous(shape));
    }

    @Override
    public synchronized void copy(Storage in, long inOffset, long[] inStrides, int[] shape, Storage out,
            long outOffset, long[] outStrides) {
        VkStorage z = vs(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        if (outOffset == 0 && n == z.size && Arrays.equals(outStrides, contiguous(shape)) && fusable(z, in)) {
            // a view: recorded, and read through its strides by whatever uses it
            Fusion.Src x = Fusion.strided(vs(in), inOffset, shape, inStrides);
            if (x != null) {
                defer(z, x);
                return;
            }
        }
        if (vs(in).pending != null && inOffset == 0 && vs(in).dtype == z.dtype && n == vs(in).size
                && Arrays.equals(inStrides, contiguous(shape)) && n < Integer.MAX_VALUE) {
            int[] rows = rows(shape, outStrides);
            if (rows != null) {
                // a pending value written into part of an array (a key/value cache step): its
                // shader stores there directly, without computing it into a temporary first
                eager(z);
                writePending(vs(in), z, outOffset, n, rows[0], rows[1]);
                return;
            }
        }
        eager(vs(in), z);
        VkStorage x = as(vs(in), z.dtype);
        copy(x, inOffset, inStrides, shape, z, outOffset, outStrides, n);
        done(x, vs(in));
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
    private void writePending(VkStorage in, VkStorage z, long outOffset, long n, int inner, int outerStride) {
        Fusion.Kernel k = Fusion.generate(in.pending, z.dtype, inner, outerStride);
        String name = fusedPipeline(k.source(), z.dtype);
        VkContext.Slot[] reads = new VkContext.Slot[k.loads().size()];
        for (int i = 0; i < reads.length; i++) {
            reads[i] = k.loads().get(i).slot();
        }
        long out = z.address(outOffset);
        if (k.inPush()) {
            ByteBuffer p = vk.push().putLong(out).putInt((int) n).putInt(0);
            k.writeMeta(p, Fusion.Kernel.PUSH_TABLE);
            p.position(Fusion.Kernel.PUSH_TABLE + k.metaBytes());
        } else {
            long meta = vk.hostData(k.metaBytes(), k::writeMeta);
            vk.push().putLong(meta).putLong(out).putInt((int) n);
        }
        vk.dispatch(name, groups(n), z.slot(), reads);
        fusedDispatches++;
    }

    private void copy(VkStorage x, long inOffset, long[] inStrides, int[] shape, VkStorage z, long outOffset,
            long[] outStrides, long n) {
        int bytes = z.dtype.bytes();
        Axes ax = Axes.merge(shape, inStrides, outStrides, null);
        if (ax.rank() == 1 && ax.stride(0, 0) == 1 && ax.stride(1, 0) == 1) {
            vk.copy(x.slot(), inOffset * bytes, z.slot(), outOffset * bytes, n * bytes);
            return;
        }
        ax.forEachBlock((offsets, inner, count) -> {
            ByteBuffer p = vk.push().putLong(x.address(inOffset + offsets[0]))
                    .putLong(z.address(outOffset + offsets[1])).putInt((int) count).putInt(inner).putInt(0).putInt(0);
            ax.put(p, inner);
            vk.dispatch(shader("copy", z.dtype), groups(count), z.slot(), x.slot());
        });
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

    /**
     * A shape with one stride array per operand, with size-1 axes dropped and adjacent axes
     * merged where every operand allows it. Ranks above {@link #MAX_AXES} are run as several
     * dispatches over the outer axes.
     */
    private static final class Axes {
        final int[] shape;
        final long[][] strides;

        private Axes(int[] shape, long[][] strides) {
            this.shape = shape;
            this.strides = strides;
        }

        /** Merges axes for two operands, plus an optional third (the contiguous output). */
        static Axes merge(int[] shape, long[] s0, long[] s1, long[] s2) {
            long[][] in = s2 == null ? new long[][] {s0, s1} : new long[][] {s0, s1, s2};
            int ops = in.length;
            int[] dims = new int[shape.length];
            long[][] st = new long[ops][shape.length];
            int r = 0;
            for (int d = 0; d < shape.length; d++) {
                if (shape[d] == 1) {
                    continue;
                }
                boolean mergeable = r > 0;
                for (int o = 0; o < ops && mergeable; o++) {
                    mergeable = st[o][r - 1] == in[o][d] * shape[d];
                }
                if (mergeable) {
                    dims[r - 1] *= shape[d];
                    for (int o = 0; o < ops; o++) {
                        st[o][r - 1] = in[o][d];
                    }
                } else {
                    dims[r] = shape[d];
                    for (int o = 0; o < ops; o++) {
                        st[o][r] = in[o][d];
                    }
                    r++;
                }
            }
            if (r == 0) {
                dims[0] = 1;
                for (int o = 0; o < ops; o++) {
                    st[o][0] = 1;
                }
                r = 1;
            }
            long[][] out = new long[ops][];
            for (int o = 0; o < ops; o++) {
                out[o] = Arrays.copyOf(st[o], r);
                for (long v : out[o]) {
                    if (Math.abs(v) > Integer.MAX_VALUE) {
                        throw new UnsupportedOperationException("strides beyond 2^31 on the vulkan backend");
                    }
                }
            }
            return new Axes(Arrays.copyOf(dims, r), out);
        }

        int rank() {
            return shape.length;
        }

        long stride(int operand, int axis) {
            return strides[operand][axis];
        }

        interface Block {
            void run(long[] offsets, int innerRank, long count);
        }

        /** Calls {@code block} once per index of the outer axes, with each operand's offset. */
        void forEachBlock(Block block) {
            int rank = shape.length;
            int innerRank = Math.min(rank, MAX_AXES);
            int outer = rank - innerRank;
            long count = 1;
            for (int d = outer; d < rank; d++) {
                count *= shape[d];
            }
            long blocks = 1;
            for (int d = 0; d < outer; d++) {
                blocks *= shape[d];
            }
            long[] offsets = new long[strides.length];
            for (long i = 0; i < blocks; i++) {
                long rest = i;
                Arrays.fill(offsets, 0);
                for (int d = outer - 1; d >= 0; d--) {
                    long idx = rest % shape[d];
                    rest /= shape[d];
                    for (int o = 0; o < strides.length; o++) {
                        offsets[o] += idx * strides[o][d];
                    }
                }
                block.run(offsets, innerRank, count);
            }
        }

        /** Writes the inner axes' shape and the first two operands' strides, padded to {@link #MAX_AXES}. */
        void put(ByteBuffer p, int innerRank) {
            int outer = shape.length - innerRank;
            for (int d = 0; d < MAX_AXES; d++) {
                p.putInt(d < innerRank ? shape[outer + d] : 1);
            }
            for (int o = 0; o < 2; o++) {
                for (int d = 0; d < MAX_AXES; d++) {
                    p.putInt(d < innerRank ? (int) strides[o][outer + d] : 0);
                }
            }
        }
    }

    // ------------------------------------------------------------------ reductions

    @Override
    public synchronized void reduce(ReduceOp op, Storage in, long outer, long length, long inner, Storage out) {
        VkStorage y = vs(out);
        long rows = outer * inner;
        if (rows == 0) {
            return;
        }
        eager(vs(in), y);
        VkStorage x = as(vs(in), y.dtype);
        if (op == ReduceOp.ARGMAX) {
            argmax(x, y, rows, length, inner);
            y.indexBound = length;
        } else {
            reduce(op.ordinal(), x, y, rows, length, inner);
        }
        done(x, vs(in));
    }

    /**
     * The index of each row's first largest value. Few long rows (the logits of a sampled
     * token) go to work groups, split into pieces when there are too few rows to fill the
     * device; otherwise one invocation per row.
     */
    private void argmax(VkStorage x, VkStorage y, long rows, long length, long inner) {
        if (rows >= 1024 || length <= 256 || x.dtype == DType.FLOAT64 || length >= (1L << 31)) {
            vk.push().putLong(x.address()).putLong(y.address()).putInt((int) rows).putInt((int) length)
                    .putInt((int) inner);
            vk.dispatch(shader("argmax", y.dtype), groups(rows), y.slot(), x.slot());
            return;
        }
        int pieces = (int) Math.max(1, Math.min(ceil(length, 2048), 256 / rows));
        if (pieces == 1) {
            vk.push().putLong(x.address()).putLong(0).putLong(y.address()).putInt((int) rows).putInt((int) length)
                    .putInt((int) inner).putInt(1).putInt(0);
            vk.dispatch(shader("argmax_group", y.dtype), rows, y.slot(), x.slot());
            return;
        }
        VkStorage part = raw(rows * pieces * 2, DType.FLOAT32, null);
        vk.push().putLong(x.address()).putLong(part.address()).putLong(y.address()).putInt((int) rows)
                .putInt((int) length).putInt((int) inner).putInt(pieces).putInt(0);
        vk.dispatch(shader("argmax_group", y.dtype), rows * pieces, part.slot(), x.slot());
        vk.push().putLong(x.address()).putLong(part.address()).putLong(y.address()).putInt((int) rows)
                .putInt((int) length).putInt((int) inner).putInt(pieces).putInt(1);
        vk.dispatch(shader("argmax_group", y.dtype), rows, y.slot(), x.slot(), part.slot());
        part.release();
    }

    /** Whether contiguous rows are reduced by 32 invocations each ({@code -Dbrainiac.vulkan.reduceRows=false}: one). */
    private static final boolean REDUCE_ROWS = !"false".equalsIgnoreCase(System.getProperty("brainiac.vulkan.reduceRows"));

    private void reduce(int op, VkStorage x, VkStorage y, long rows, long length, long inner) {
        vk.push().putLong(x.address()).putLong(y.address()).putInt((int) rows).putInt((int) length)
                .putInt((int) inner).putInt(op);
        if (rows < 1024 && length > 256) {
            vk.dispatch(shader("reduce_group", y.dtype), rows, y.slot(), x.slot());
        } else if (inner == 1 && length >= 32 && REDUCE_ROWS) {
            // contiguous rows: 32 invocations per row, 8 rows per work group
            vk.dispatch(shader("reduce_rows", y.dtype), (rows + 7) / 8, y.slot(), x.slot());
        } else {
            vk.dispatch(shader("reduce", y.dtype), groups(rows), y.slot(), x.slot());
        }
    }

    // ------------------------------------------------------------------ matrix products

    @Override
    public void matmul(Storage a, Storage b, int m, int k, int n, Storage out) {
        matmulBatched(a, b, 1, m, k, n, out);
    }

    /** Product tiles at least this many 64 x 64 tiles use them; smaller products use 32 x 32. */
    private long bigTiles() {
        return tuning.get("bigTileGroups");
    }
    /** Float16 products with at least this many rows use cooperative matrices where the device has them. */
    private int coopRows() {
        return (int) Math.min(Integer.MAX_VALUE, tuning.get("coopRows"));
    }

    /** The cooperative-matrix tile shape in use; null if they are not. */
    private Tuning.CoopShape coopShape() {
        long i = tuning.get("coopShape");
        return i < 0 ? null : Tuning.COOP_SHAPES.get((int) i);
    }

    /** Kernel choices for this device: see {@link Tuning} and {@link #tune(boolean)}. */
    Tuning tuning = new Tuning();
    private String tuningNote = "defaults";
    /**
     * Products with fewer tiles than this, and a long k, split k into pieces summed afterwards
     * ({@code -Dbrainiac.vulkan.splitBelow}). Weight gradients (width x width over a k of batch
     * times sequence) otherwise run as a few dozen work groups each walking all of k.
     */
    private long splitBelow() {
        return tuning.get("splitBelow");
    }

    @Override
    public synchronized void matmulBatched(Storage a, Storage b, int batch, int m, int k, int n, Storage out) {
        VkStorage z = vs(out);
        if ((long) batch * m * n == 0) {
            return;
        }
        // few rows times a transposed matrix (a tied output head): its rows read in place
        if (m <= Math.min(VEC_ROWS, VECT_ROWS) && batch == 1 && k > 0) {
            Operand vb = view(vs(b), 1, k, n, z.dtype);
            if (vb != null && vb.row() == 1 && vb.col() != 1) {
                eager(vs(a), vb.values(), z);
                VkStorage x = as(vs(a), z.dtype);
                matvecTransposed(x, vb, m, k, n, z);
                done(x, vs(a));
                return;
            }
        }
        // a transposed or otherwise strided operand still pending is read in place by the tiles
        boolean tiled = k > 0 && !(m <= VEC_ROWS && m <= VECT_ROWS && batch == 1) && !coop(z.dtype, m);
        Operand va = tiled ? view(vs(a), batch, m, k, z.dtype) : null;
        Operand vb = tiled ? view(vs(b), batch, k, n, z.dtype) : null;
        if (va != null || vb != null) {
            eager(va != null ? va.values() : vs(a), vb != null ? vb.values() : vs(b), z);
            VkStorage x = va != null ? va.values() : as(vs(a), z.dtype);
            VkStorage y = vb != null ? vb.values() : as(vs(b), z.dtype);
            matmul(va != null ? va : Operand.of(x, m, k), vb != null ? vb : Operand.of(y, k, n), batch, m, k, n, z);
            if (va == null) {
                done(x, vs(a));
            }
            if (vb == null) {
                done(y, vs(b));
            }
            return;
        }
        eager(vs(a), vs(b), z);
        if (k == 0) {
            vk.fill(z.slot(), 0, (z.bytes() + 3) & ~3L, 0);
            return;
        }
        VkStorage x = as(vs(a), z.dtype);
        VkStorage y = as(vs(b), z.dtype);
        matmul(x, y, batch, m, k, n, z);
        done(x, vs(a));
        done(y, vs(b));
    }

    /** Whether products read transposed and strided operands in place ({@code -Dbrainiac.vulkan.stridedMatmul=false}: never). */
    private static final boolean STRIDED_MATMUL =
            !"false".equalsIgnoreCase(System.getProperty("brainiac.vulkan.stridedMatmul"));

    /**
     * How the tiled product reads a matrix operand: from {@code values} at {@code offset}, with
     * the given strides between matrices of the batch, rows and columns.
     */
    private record Operand(VkStorage values, long offset, long batch, long row, long col) {

        /** Stored {@code rows x cols} matrices one after another. */
        static Operand of(VkStorage values, int rows, int cols) {
            return new Operand(values, 0, (long) rows * cols, cols, 1);
        }
    }

    /** A pending strided read of stored values ({@code [batch,] rows, cols}), as an operand; null if none. */
    private static Operand view(VkStorage s, int batch, int rows, int cols, DType type) {
        if (!STRIDED_MATMUL || !(s.pending instanceof Fusion.Src src) || src.direct() || src.storage().dtype != type) {
            return null;
        }
        int[] d = src.dims();
        int[] st = src.strides();
        for (int x : st) {
            if (x < 0) {
                return null;
            }
        }
        Operand o = null;
        if (batch == 1 && d.length == 2 && d[0] == rows && d[1] == cols) {
            o = new Operand(src.storage(), src.offset(), 0, st[0], st[1]);
        } else if (d.length == 3 && d[0] == batch && d[1] == rows && d[2] == cols) {
            o = new Operand(src.storage(), src.offset(), st[0], st[1], st[2]);
        }
        // the product reads along a contiguous row or column (see matmul.comp)
        return o != null && (o.col() == 1 || o.row() == 1) ? o : null;
    }

    /** Whether a float16 product has enough rows for the matrix cores, where the device has them. */
    private boolean coop(DType type, int m) {
        return vk.coopMatrix && type == DType.FLOAT16 && m >= coopRows() && coopShape() != null
                && !"false".equals(System.getProperty("brainiac.vulkan.coopmat"));
    }

    // from how many rows matmul_vect sums each column with 32 invocations (8 columns per group) instead of 64
    // (8: batched decoding 1-5% faster on an RX 9070 XT)
    private static final int VECT_NARROW = Integer.getInteger("brainiac.vulkan.vectNarrow", 8);
    // the most rows of a product with a transposed matrix read by matmul_vect
    private static final int VECT_ROWS = Integer.getInteger("brainiac.vulkan.vectRows", 8);
    /** Products of at most this many rows read the matrix once ({@code -Dbrainiac.vulkan.matvec=false}: never). */
    private static final int VEC_ROWS = "false".equals(System.getProperty("brainiac.vulkan.matvec")) ? 0 : 8;

    private void matmul(VkStorage x, VkStorage y, int batch, int m, int k, int n, VkStorage z) {
        if (m <= VEC_ROWS && batch == 1) {
            matvec(x, y, m, k, n, z);
            return;
        }
        matmul(new Operand(x, 0, (long) m * k, k, 1), new Operand(y, 0, (long) k * n, n, 1), batch, m, k, n, z);
    }

    private void matmul(Operand xo, Operand yo, int batch, int m, int k, int n, VkStorage z) {
        VkStorage x = xo.values();
        VkStorage y = yo.values();
        long big = (long) batch * ceil(m, 64) * ceil(n, 64);
        // float16 with enough rows to fill 16 x 16 pieces: the matrix cores
        boolean coop = coop(z.dtype, m);
        Tuning.CoopShape shape = coop ? coopShape() : null;
        int tileM = coop ? shape.rows() : big >= bigTiles() ? 64 : 32;
        int tileN = coop ? shape.cols() : tileM;
        long tiles = (long) batch * ceil(m, tileM) * ceil(n, tileN);
        int splits = 1;
        int chunk = k;
        if (tiles < splitBelow() && k >= 1024) {
            splits = (int) Math.min(ceil(k, 256), Math.max(2, 2 * splitBelow() / tiles));
            chunk = ceil(ceil(k, splits), 16) * 16;
            splits = ceil(k, chunk);
        }
        long mn = (long) m * n;
        VkStorage target = splits == 1 ? z : raw((long) batch * splits * mn, z.dtype, null);
        int maxZ = 65535;
        int pieces = batch * splits;
        // batches beyond the z limit go in several dispatches; each starts at a whole batch
        int perDispatch = Math.max(1, maxZ / splits) * splits;
        for (int first = 0; first < pieces; first += perDispatch) {
            int count = Math.min(perDispatch, pieces - first);
            long firstBatch = first / splits;
            vk.push().putLong(x.address(xo.offset() + firstBatch * xo.batch()))
                    .putLong(y.address(yo.offset() + firstBatch * yo.batch()))
                    .putLong(target.address(first * mn)).putInt(m).putInt(k).putInt(n).putInt((int) xo.batch())
                    .putInt((int) yo.batch()).putInt((int) mn).putInt(splits).putInt(chunk).putInt((int) xo.row())
                    .putInt((int) xo.col()).putInt((int) yo.row()).putInt((int) yo.col());
            // the strided variant only where an operand needs it: its addressing costs the plain case
            boolean plain = xo.row() == k && xo.col() == 1 && yo.row() == n && yo.col() == 1;
            String name = coop ? "matmul_coop" + shape.suffix() : (tileM == 64 ? "matmul64" : "matmul32")
                    + (plain ? "" : "s" + (xo.col() == 1 ? "" : "a") + (yo.col() == 1 ? "" : "b"));
            vk.dispatch(shader(name, z.dtype), ceil(n, tileN), ceil(m, tileM), count,
                    target.slot(), x.slot(), y.slot());
        }
        if (splits > 1) {
            reduce(ReduceOp.SUM.ordinal(), target, z, (long) batch * mn, splits, mn);
            target.release();
        }
    }

    /** Few rows times a transposed matrix, whose columns are stored as rows: see matmul_vect.comp. */
    private void matvecTransposed(VkStorage x, Operand b, int m, int k, int n, VkStorage z) {
        VkStorage y = b.values();
        int rows = m == 1 ? 1 : m == 2 ? 2 : m <= 4 ? 4 : 8;
        long first = y.address(b.offset());
        long align = 4L * y.dtype.bytes();
        boolean vectors = k % 4 == 0 && b.col() % 4 == 0 && first % align == 0 && x.address() % align == 0;
        vk.push().putLong(x.address()).putLong(x.address()).putLong(first).putLong(first).putLong(z.address()).putInt(m).putInt(k)
                .putInt(n).putInt((int) b.col()).putInt(vectors ? 1 : 0);
        boolean narrow = rows >= VECT_NARROW;
        vk.dispatch(shader("matmul_vect" + rows + (narrow ? "n" : ""), z.dtype), ceil(n, narrow ? 8 : 4), z.slot(), x.slot(), y.slot());
    }

    /** Few rows times a matrix, limited by reading the matrix: see matmul_vec.comp. */
    private void matvec(VkStorage x, VkStorage y, int m, int k, int n, VkStorage z) {
        int columnGroups = ceil(n, 128);
        // enough pieces to fill the device, of at least 64 rows each
        int pieces = Math.max(1, Math.min(ceil(k, 64), PACKED_GROUPS / columnGroups));
        int chunk = ceil(k, pieces);
        pieces = ceil(k, chunk);
        int rows = m == 1 ? 1 : m == 2 ? 2 : m <= 4 ? 4 : 8;
        // the vector loads need four aligned columns
        boolean vectors = n % 4 == 0 && y.address() % (4L * y.dtype.bytes()) == 0;
        Pieces split = pieces(pieces, m, n, columnGroups, z.dtype);
        vk.push().putLong(x.address()).putLong(y.address()).putLong(y.address()).putLong(z.address())
                .putLong(split.partsAddress()).putLong(counters.address()).putInt(m).putInt(k).putInt(n).putInt(chunk)
                .putInt(vectors ? 1 : 0).putInt(pieces).putInt(split.counter());
        vk.dispatch(shader("matmul_vec" + rows, z.dtype), columnGroups, 1, pieces, split.writes(z), x.slot(),
                y.slot());
        split.release();
    }

    /**
     * Scratch for a product split into pieces along k: each piece's sums and a run of
     * counters (one per column group) by which the last piece to finish adds them up and
     * resets the counter. Runs of the counter buffer rotate, so products running at once use
     * different counters; the buffer is not listed as written, which would order them.
     */
    private record Pieces(VkStorage parts, int counter) {

        long partsAddress() {
            return parts == null ? 0 : parts.address();
        }

        VkContext.Slot[] writes(VkStorage z) {
            return parts == null ? new VkContext.Slot[] {z.slot()} : new VkContext.Slot[] {z.slot(), parts.slot()};
        }

        void release() {
            if (parts != null) {
                parts.release();
            }
        }
    }

    /** Counters of products split into pieces, all 0 between products. */
    private static final int COUNTERS = 1 << 16;
    private VkStorage counters;
    private int nextCounter;

    private Pieces pieces(int pieces, int m, int n, int columnGroups, DType type) {
        if (counters == null) {
            counters = raw(COUNTERS, null);
            fillNow(counters, 0);
        }
        if (pieces == 1) {
            return new Pieces(null, 0);
        }
        if (nextCounter + columnGroups > COUNTERS) {
            nextCounter = 0;
        }
        int counter = nextCounter;
        nextCounter += columnGroups;
        long vectors = (long) pieces * m * ceil(n, 4);
        VkStorage parts = type == DType.FLOAT64 ? raw(vectors * 4, DType.FLOAT64, null) : raw(vectors * 4, null);
        return new Pieces(parts, counter);
    }

    private static int ceil(long a, int b) {
        return (int) ((a + b - 1) / b);
    }

    // ------------------------------------------------------------------ attention

    /**
     * When the fused attention kernel is used: {@code -Dbrainiac.vulkan.fusedAttention=false}
     * never; by default for inference (it has no backward kernel yet) with head sizes up to
     * 256, in float32, float16 and bfloat16.
     */
    private static final boolean FUSED_ATTENTION = !"false".equals(System.getProperty("brainiac.vulkan.fusedAttention"));

    /**
     * {@code -Dbrainiac.vulkan.fusedAttention=true} also uses it for training, with the
     * backward shaders (heads up to {@link #BACKWARD_SIZE}): they keep a row in registers, one
     * invocation per row, so they pay off only where the [queries, keys] weights would not fit.
     */
    private static final boolean FUSED_TRAINING = "true".equals(System.getProperty("brainiac.vulkan.fusedAttention"));

    /** The largest head size of the backward shaders ({@code ATT_D} in attention_dq.comp). */
    private static final int BACKWARD_SIZE = 64;
    /** Rows per work group of attention_dq and attention_dkv (their ROWS). */
    private static final int BACKWARD_ROWS = 64;

    @Override
    public boolean hasAttention(int[] shape, DType type, boolean backward) {
        if (backward && (!FUSED_TRAINING || shape[5] > BACKWARD_SIZE || shape[6] > BACKWARD_SIZE)) {
            return false;
        }
        return FUSED_ATTENTION && type != DType.FLOAT64 && shape[5] <= GROUP_SIZE && shape[6] <= GROUP_SIZE
                && supports(type);
    }

    @Override
    public synchronized void attentionBackward(Storage q, Storage k, Storage v, Storage out, Storage lse,
            Storage grad, int[] shape, float scale, boolean causal, Storage dq, Storage dk, Storage dv) {
        VkStorage gq = vs(dq);
        DType type = gq.dtype;
        if (type == DType.FLOAT64 || shape[5] > BACKWARD_SIZE || shape[6] > BACKWARD_SIZE) {
            throw new UnsupportedOperationException("fused attention backward: " + type + ", heads of " + shape[5]
                    + " and " + shape[6]);
        }
        VkStorage gk = vs(dk);
        VkStorage gv = vs(dv);
        eager(vs(q), vs(k), vs(v), vs(out), vs(lse), vs(grad), gq, gk, gv);
        VkStorage xq = as(vs(q), type);
        VkStorage xk = as(vs(k), type);
        VkStorage xv = as(vs(v), type);
        VkStorage xo = as(vs(out), type);
        VkStorage xl = as(vs(lse), type);
        VkStorage xg = as(vs(grad), type);
        long rows = (long) shape[0] * shape[1] * shape[3];
        long keys = (long) shape[0] * shape[2] * shape[4];
        VkStorage dsum = raw(rows, type, null);
        vk.push().putLong(xq.address()).putLong(xk.address()).putLong(xv.address()).putLong(xo.address())
                .putLong(xl.address()).putLong(xg.address()).putLong(gq.address()).putLong(dsum.address())
                .putInt(shape[1]).putInt(shape[2]).putInt(shape[3]).putInt(shape[4]).putInt(shape[5])
                .putInt(shape[6]).putFloat(scale).putInt(causal ? 1 : 0).putInt(Math.toIntExact(rows));
        String size = Math.max(shape[5], shape[6]) <= 32 ? "32" : "";   // smaller register arrays
        long heads = (long) shape[0] * shape[1];
        long keyHeads = (long) shape[0] * shape[2];
        vk.dispatch(shader("attention_dq" + size, type), heads * ((shape[3] + BACKWARD_ROWS - 1) / BACKWARD_ROWS),
                new VkContext.Slot[] {gq.slot(), dsum.slot()},
                xq.slot(), xk.slot(), xv.slot(), xo.slot(), xl.slot(), xg.slot());
        vk.push().putLong(xq.address()).putLong(xk.address()).putLong(xv.address()).putLong(xl.address())
                .putLong(xg.address()).putLong(dsum.address()).putLong(gk.address()).putLong(gv.address())
                .putInt(shape[1]).putInt(shape[2]).putInt(shape[3]).putInt(shape[4]).putInt(shape[5])
                .putInt(shape[6]).putFloat(scale).putInt(causal ? 1 : 0).putInt(Math.toIntExact(keys));
        vk.dispatch(shader("attention_dkv" + size, type), keyHeads * ((shape[4] + BACKWARD_ROWS - 1) / BACKWARD_ROWS),
                new VkContext.Slot[] {gk.slot(), gv.slot()},
                xq.slot(), xk.slot(), xv.slot(), xl.slot(), xg.slot(), dsum.slot());
        dsum.release();
        done(xq, vs(q));
        done(xk, vs(k));
        done(xv, vs(v));
        done(xo, vs(out));
        done(xl, vs(lse));
        done(xg, vs(grad));
    }

    @Override
    public boolean hasAttentionWithBias(int[] shape, DType type) {
        return hasAttention(shape, type, false);
    }

    @Override
    public synchronized void attention(Storage q, Storage k, Storage v, int[] shape, float scale, boolean causal,
            Storage out, Storage lse) {
        attention(q, k, v, shape, scale, causal, null, new long[4], out, lse);
    }

    @Override
    public synchronized void attention(Storage q, Storage k, Storage v, int[] shape, float scale, boolean causal,
            Storage bias, long[] strides, Storage out, Storage lse) {
        VkStorage z = vs(out);
        if (z.dtype == DType.FLOAT64 || shape[5] > GROUP_SIZE || shape[6] > GROUP_SIZE) {
            throw new UnsupportedOperationException("fused attention: " + z.dtype + ", heads of " + shape[5] + " and "
                    + shape[6]);
        }
        VkStorage l = vs(lse);
        eager(vs(q), vs(k), vs(v), z, l);
        VkStorage xq = as(vs(q), z.dtype);
        VkStorage xk = as(vs(k), z.dtype);
        VkStorage xv = as(vs(v), z.dtype);
        if (l.dtype != z.dtype) {
            throw new IllegalArgumentException("lse " + l.dtype + ", output " + z.dtype);
        }
        VkStorage xl = l;
        VkStorage xb = bias == null ? xq : as(vs(bias), z.dtype);   // no bias: an address never read
        if (bias != null) {
            eager(vs(bias));
        }
        long rows = (long) shape[0] * shape[1] * shape[3];
        // flash decoding: few rows (one query per head while generating) and many keys share
        // each row's keys out among work groups, enough to fill the device
        int parts = (int) Math.max(1, Math.min(ceil(shape[4], ATTENTION_PIECE), ATTENTION_GROUPS / rows));
        int chunk = ceil(shape[4], parts);   // short pieces too: a 100-key cache over 14 heads still splits
        parts = ceil(shape[4], chunk);
        VkStorage pieces = parts > 1 ? raw(rows * parts * (shape[6] + 2), null) : null;
        var push = vk.push().putLong(xq.address()).putLong(xk.address()).putLong(xv.address()).putLong(z.address())
                .putLong(xl.address()).putLong(xb.address());
        if (pieces != null) {
            push.putLong(pieces.address());
        }
        push.putInt(shape[1]).putInt(shape[2]).putInt(shape[3])
                .putInt(shape[4]).putInt(shape[5]).putInt(shape[6]).putFloat(scale).putInt(causal ? 1 : 0)
                .putInt((int) rows).putInt(bias != null ? 1 : 0).putInt(Math.toIntExact(strides[0]))
                .putInt(Math.toIntExact(strides[1])).putInt(Math.toIntExact(strides[2]))
                .putInt(Math.toIntExact(strides[3]));
        if (pieces == null) {
            vk.dispatch(shader("attention", z.dtype), rows, new VkContext.Slot[] {z.slot(), xl.slot()}, xq.slot(),
                    xk.slot(), xv.slot(), xb.slot());
        } else {
            push.putInt(parts).putInt(chunk);
            vk.dispatch(shader("attention_split", z.dtype), rows * parts, pieces.slot(), xq.slot(), xk.slot(),
                    xv.slot(), xb.slot());
            vk.push().putLong(pieces.address()).putLong(z.address()).putLong(xl.address()).putInt(parts)
                    .putInt(shape[6]).putInt((int) rows);
            vk.dispatch(shader("attention_combine", z.dtype), rows, new VkContext.Slot[] {z.slot(), xl.slot()},
                    pieces.slot());
            pieces.release();
        }
        if (bias != null) {
            done(xb, vs(bias));
        }
        done(xq, vs(q));
        done(xk, vs(k));
        done(xv, vs(v));
    }

    // ------------------------------------------------------------------ 8-bit rows

    /** Rows of 8-bit levels on the device (raw bytes read as words) with a float scale per row. */
    private record QRows(int rows, int cols, DType dtype, VkStorage levels, VkStorage scales)
            implements RowQuantizedStorage {

        @Override
        public long size() {
            return (long) rows * cols;
        }

        @Override
        public long bytes() {
            return levels.bytes() + scales.bytes();
        }
    }

    @Override
    public synchronized RowQuantizedStorage allocateQuantizedRows(long rows, int cols, DType dtype) {
        if (cols % 4 != 0 || cols > 4 * GROUP_SIZE || rows * cols / 4 > Integer.MAX_VALUE || rows > Integer.MAX_VALUE) {
            return Backend.super.allocateQuantizedRows(rows, cols, dtype);
        }
        VkStorage levels = raw(Math.max(1, rows * cols / 4), null);
        VkStorage scales = raw(Math.max(1, rows), DType.FLOAT32, null);
        fillNow(levels, 0);   // not deferred: the shaders below write these buffers directly
        fillNow(scales, 0);
        return new QRows((int) rows, cols, dtype, levels, scales);
    }

    @Override
    public synchronized void writeQuantizedRows(RowQuantizedStorage target, Storage values, long outer,
            int blockRows, int sourceRows, int start) {
        if (!(target instanceof QRows t)) {
            Backend.super.writeQuantizedRows(target, values, outer, blockRows, sourceRows, start);
            return;
        }
        VkStorage x = vs(values);
        eager(x);
        long rows = outer * sourceRows;
        if (rows == 0) {
            return;
        }
        vk.push().putLong(x.address()).putLong(t.levels().address()).putLong(t.scales().address())
                .putInt(t.cols() / 4).putInt(blockRows).putInt(sourceRows).putInt(start).putInt(Math.toIntExact(rows));
        vk.dispatch(shader("quantize_rows", x.dtype), rows, new VkContext.Slot[] {t.levels().slot(),
                t.scales().slot()}, x.slot());
    }

    private void unpackRows(QRows r, VkStorage z) {
        eager(z);
        vk.push().putLong(r.levels().address()).putLong(r.scales().address()).putLong(z.address()).putInt(r.cols())
                .putInt(Math.toIntExact(r.size()));
        vk.dispatch(shader("unpack_rows", z.dtype), groups(r.size()), z.slot(), r.levels().slot(), r.scales().slot());
    }

    @Override
    public boolean hasQuantizedAttention(int[] shape, DType type) {
        return hasAttention(shape, type, false) && shape[5] % 4 == 0 && shape[6] % 4 == 0;
    }

    @Override
    public synchronized void attentionQuantized(Storage q, RowQuantizedStorage k, RowQuantizedStorage v, int[] shape,
            float scale, boolean causal, Storage bias, long[] strides, Storage out, Storage lse) {
        VkStorage z = vs(out);
        if (!(k instanceof QRows qk && v instanceof QRows qv) || !hasQuantizedAttention(shape, z.dtype)) {
            Backend.super.attentionQuantized(q, k, v, shape, scale, causal, bias, strides, out, lse);
            return;
        }
        VkStorage l = vs(lse);
        eager(vs(q), z, l);
        VkStorage xq = as(vs(q), z.dtype);
        VkStorage xb = bias == null ? xq : as(vs(bias), z.dtype);   // no bias: an address never read
        if (bias != null) {
            eager(vs(bias));
        }
        long rows = (long) shape[0] * shape[1] * shape[3];
        vk.push().putLong(xq.address()).putLong(qk.levels().address()).putLong(qk.scales().address())
                .putLong(qv.levels().address()).putLong(qv.scales().address()).putLong(z.address())
                .putLong(l.address()).putLong(xb.address()).putInt(shape[1]).putInt(shape[2]).putInt(shape[3])
                .putInt(shape[4]).putInt(shape[5]).putInt(shape[6]).putFloat(scale).putInt(causal ? 1 : 0)
                .putInt((int) rows).putInt(bias != null ? 1 : 0).putInt(Math.toIntExact(strides[0]))
                .putInt(Math.toIntExact(strides[1])).putInt(Math.toIntExact(strides[2]))
                .putInt(Math.toIntExact(strides[3]));
        vk.dispatch(shader("attention_q8", z.dtype), rows, new VkContext.Slot[] {z.slot(), l.slot()}, xq.slot(),
                qk.levels().slot(), qk.scales().slot(), qv.levels().slot(), qv.scales().slot(), xb.slot());
        if (bias != null) {
            done(xb, vs(bias));
        }
        done(xq, vs(q));
    }

    /** The longest power-of-two transform one work group computes in shared memory (fft.comp). */
    private static final int FFT_MAX = 2048;

    @Override
    public synchronized Storage fft(Storage in, long rows, int n, boolean inverse) {
        VkStorage x = vs(in);
        if (Integer.bitCount(n) != 1 || n > FFT_MAX || x.dtype == DType.FLOAT64) {
            return Backend.super.fft(in, rows, n, inverse);   // on the host: Bluestein, or long rows
        }
        eager(x);
        VkStorage z = raw(rows * n * 2, DType.FLOAT32, null);
        vk.push().putLong(x.address()).putLong(z.address()).putInt(Math.toIntExact(rows)).putInt(n)
                .putInt(Integer.numberOfTrailingZeros(n)).putInt(inverse ? 1 : 0);
        vk.dispatch(shader("fft", x.dtype), rows, z.slot(), x.slot());
        return z;
    }

    @Override
    public boolean hasRmsNorm(DType type) {
        return type != DType.FLOAT64 && supports(type);
    }

    @Override
    public synchronized void rmsNorm(Storage x, Storage weight, long rows, int cols, float eps, Storage out) {
        VkStorage z = vs(out);
        if (!hasRmsNorm(z.dtype)) {
            throw new UnsupportedOperationException("fused RMSNorm: " + z.dtype);
        }
        eager(vs(x), vs(weight), z);
        VkStorage xx = as(vs(x), z.dtype);
        VkStorage xw = as(vs(weight), z.dtype);
        vk.push().putLong(xx.address()).putLong(xw.address()).putLong(z.address()).putInt(Math.toIntExact(rows))
                .putInt(cols).putFloat(eps);
        vk.dispatch(shader("rms_norm", z.dtype), rows, z.slot(), xx.slot(), xw.slot());
        done(xx, vs(x));
        done(xw, vs(weight));
    }

    @Override
    public boolean hasAddRmsNorm(DType type) {
        return hasRmsNorm(type);
    }

    @Override
    public synchronized void addRmsNorm(Storage x, Storage r, Storage weight, long rows, int cols, float eps,
            Storage sum, Storage out) {
        VkStorage z = vs(out);
        VkStorage s = vs(sum);
        if (!hasAddRmsNorm(z.dtype) || s.dtype != z.dtype) {
            throw new UnsupportedOperationException("fused add and RMSNorm: " + z.dtype);
        }
        eager(vs(x), vs(r), vs(weight), s, z);
        VkStorage xx = as(vs(x), z.dtype);
        VkStorage xr = as(vs(r), z.dtype);
        VkStorage xw = as(vs(weight), z.dtype);
        vk.push().putLong(xx.address()).putLong(xr.address()).putLong(xw.address()).putLong(s.address())
                .putLong(z.address()).putInt(Math.toIntExact(rows)).putInt(cols).putFloat(eps);
        vk.dispatch(shader("add_rms_norm", z.dtype), rows, new VkContext.Slot[] {s.slot(), z.slot()}, xx.slot(),
                xr.slot(), xw.slot());
        done(xx, vs(x));
        done(xr, vs(r));
        done(xw, vs(weight));
    }

    /** The work group size of the shaders ({@code GROUP} in common.glsl). */
    private static final int GROUP_SIZE = 256;

    // ------------------------------------------------------------------ packed weights

    /**
     * A packed matrix on the device. {@code levels} holds one signed byte per value, or with
     * {@link #NIBBLES} two per byte (see PackedMatrix); with {@code planeBits} (5- and 6-bit
     * levels) it holds the low four bits of {@code level + 2^(bits-1)} as nibbles and {@code hi}
     * the rest, {@code 8 / planeBits} rows of a column per byte at {@code (r / (8 / planeBits)) * cols + c}.
     * The scale of group {@code g} in column {@code c} is its code (or 1) times
     * {@code scales[(g / perSuper) * cols + c]}, float32 or with {@link #HALF} float16; the
     * offset likewise from {@code offsets}, plus the scale times {@code shift}.
     * {@code host} is the matrix it was uploaded from, or null for one quantized on the device,
     * which {@link #release} frees.
     */
    private record Packed(int rows, int cols, int bits, int group, int flags, int perSuper, float shift,
            PackedMatrix host, VkStorage levels, VkStorage hi, VkStorage scaleCodes, VkStorage offsetCodes,
            VkStorage scales, VkStorage offsets) implements PackedStorage {

        static final int NIBBLES = 1;
        static final int OFFSETS = 2;
        static final int SCALE_CODES = 4;
        static final int OFFSET_CODES = 8;
        static final int HALF = 16;
        static final int CODEBOOK = 32;

        @Override
        public long size() {
            return (long) rows * cols;
        }

        @Override
        public long bytes() {
            long b = 0;
            for (VkStorage s : new VkStorage[] {levels, hi, scaleCodes, offsetCodes, scales, offsets}) {
                b += s == null ? 0 : s.bytes();
            }
            return b;
        }

        /** @return the flags the shaders read: the above, the plane bits from bit 8 and the level bits from bit 12 */
        int layout() {
            int planeBits = hi == null || (flags & CODEBOOK) != 0 ? 0 : bits - 4;
            return flags | planeBits << 8 | bits << 12;
        }

        /** Pushes the addresses of levels, high bits, scale codes, offset codes, scales and offsets. */
        void push(java.nio.ByteBuffer b) {
            long any = levels.address();
            b.putLong(any).putLong(hi == null ? any : hi.address())
                    .putLong(scaleCodes == null ? any : scaleCodes.address())
                    .putLong(offsetCodes == null ? any : offsetCodes.address()).putLong(scales.address())
                    .putLong(offsets == null ? any : offsets.address());
        }

        /** @return the storages a product reads */
        VkContext.Slot[] reads(VkStorage... more) {
            List<VkContext.Slot> slots = new ArrayList<>();
            for (VkStorage s : more) {
                slots.add(s.slot());
            }
            for (VkStorage s : new VkStorage[] {levels, hi, scaleCodes, offsetCodes, scales, offsets}) {
                if (s != null) {
                    slots.add(s.slot());
                }
            }
            return slots.toArray(VkContext.Slot[]::new);
        }
    }

    /** Rows up to which a product reads packed levels directly; more unpack first. */
    private static final int PACKED_ROWS = 8;
    /** Work groups a split attention aims for, and the fewest keys a piece takes. */
    private static final int ATTENTION_GROUPS = Integer.getInteger("brainiac.vulkan.attentionGroups", 256);
    private static final int ATTENTION_PIECE = Integer.getInteger("brainiac.vulkan.attentionPiece", 32);
    /** Products by packed weights split their rows into pieces until about this many work groups run. */
    private static final int PACKED_GROUPS = Integer.getInteger("brainiac.vulkan.packedGroups", 256);

    /** Whether 5- and 6-bit levels are kept as nibbles plus high-bit planes ({@code -Dbrainiac.vulkan.packedPlanes=false}: a byte each). */
    private static final boolean PACKED_PLANES = !"false".equals(System.getProperty("brainiac.vulkan.packedPlanes"));
    /** Whether levels of at most 16 distinct values go as 4-bit indices into a table ({@code -Dbrainiac.vulkan.packedCodebook=false}: never). */
    private static final boolean PACKED_CODEBOOK = !"false".equals(System.getProperty("brainiac.vulkan.packedCodebook"));
    /** Whether factored scales stay factored on the device ({@code -Dbrainiac.vulkan.packedFactors=false}: float32 per group). */
    private static final boolean PACKED_FACTORS = !"false".equals(System.getProperty("brainiac.vulkan.packedFactors"));

    /**
     * Packs a matrix for this backend. It uploads in the background, while the caller goes on
     * (decoding a model's next weights), or on its first use if that comes first. A matrix
     * unpacked to the host before it uploaded, as {@code Llama.fuseProjections} joins q, k
     * and v and gate and up, waits for its first use instead: joined parts then never upload.
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

    /** Uploads in the background, unless it was unpacked to the host first and {@code always} is false. */
    private void fetch(Deferred d, boolean always) {
        prefetcher.execute(() -> {
            if (always || !d.skip) {
                try {
                    packed(d);
                } catch (RuntimeException e) {
                    // out of memory or such: the first use tries again, and fails there
                }
            }
        });
    }

    /** Whether packed matrices upload in the background ({@code -Dbrainiac.vulkan.prefetch=false}: on first use). */
    private static final boolean PREFETCH = !"false".equals(System.getProperty("brainiac.vulkan.prefetch"));
    /** Uploads packed matrices in the background, one at a time; its thread starts with the first. */
    private final java.util.concurrent.ExecutorService prefetcher = java.util.concurrent.Executors
            .newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "brainiac-vulkan-upload");
                t.setDaemon(true);
                return t;
            });

    /** A matrix packed for a backend, uploaded in the background or on its first use. */
    private static final class Deferred implements PackedStorage {
        final VulkanBackend owner;
        final PackedMatrix host;
        /** The uploaded matrix, once uploaded. */
        Packed done;
        /** Whether the background upload should leave it (it was unpacked to the host first). */
        volatile boolean skip;

        Deferred(VulkanBackend owner, PackedMatrix host) {
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
            return done != null ? done.bytes() : host.data().length;
        }
    }

    /** Uploads a packed matrix. */
    private synchronized Packed uploadPacked(PackedMatrix matrix) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        int bits = matrix.bits();
        int flags = matrix.nibbles() ? Packed.NIBBLES : 0;
        VkStorage levels;
        VkStorage hi = null;
        byte[] table;
        if (PACKED_PLANES && !matrix.nibbles() && (bits == 5 || bits == 6) && rows % 8 == 0) {
            int planeBits = bits - 4;
            int perByte = 8 / planeBits;
            byte[] data = matrix.data();
            byte[] low = new byte[rows / 2 * cols];
            byte[] high = new byte[rows / perByte * cols];
            int half = 1 << (bits - 1);
            // in parallel over runs of 8 rows, which write their own bytes of both planes
            java.util.stream.IntStream.range(0, rows / 8).parallel().forEach(run -> {
                for (int r = run * 8; r < run * 8 + 8; r++) {
                    int lowAt = r / 2 * cols;
                    int highAt = r / perByte * cols;
                    int lowShift = (r & 1) * 4;
                    int highShift = (r % perByte) * planeBits;
                    for (int c = 0, at = r * cols; c < cols; c++, at++) {
                        int u = data[at] + half;
                        low[lowAt + c] |= (byte) ((u & 15) << lowShift);
                        high[highAt + c] |= (byte) ((u >> 4) << highShift);
                    }
                }
            });
            levels = uploadBytes(low);
            hi = uploadBytes(high);
        } else if (PACKED_CODEBOOK && !matrix.nibbles() && rows % 2 == 0
                && (table = codebook(matrix.data())) != null) {
            // at most 16 distinct levels (IQ4_NL, IQ4_XS, MXFP4): a 4-bit index each into a table
            int[] index = new int[256];
            for (int i = 0; i < table.length; i++) {
                index[table[i] & 0xFF] = i;
            }
            byte[] data = matrix.data();
            byte[] packed = new byte[rows / 2 * cols];
            // in parallel over row pairs, each filling its own bytes
            java.util.stream.IntStream.range(0, rows / 2).parallel().forEach(pair -> {
                int even = 2 * pair * cols;
                int odd = even + cols;
                int at = pair * cols;
                for (int c = 0; c < cols; c++) {
                    packed[at + c] = (byte) (index[data[even + c] & 0xFF] | index[data[odd + c] & 0xFF] << 4);
                }
            });
            levels = uploadBytes(packed);
            hi = uploadBytes(java.util.Arrays.copyOf(table, 16));
            flags |= Packed.CODEBOOK;
        } else {
            levels = uploadBytes(matrix.data());
        }
        PackedMatrix.Factors f = matrix.factors();
        if (PACKED_FACTORS && f != null && halves(f.superScales()) && (f.superOffsets() == null || halves(f.superOffsets()))) {
            flags |= Packed.HALF | (f.scaleCodes() != null ? Packed.SCALE_CODES : 0)
                    | (f.superOffsets() != null ? Packed.OFFSETS : 0) | (f.offsetCodes() != null ? Packed.OFFSET_CODES : 0);
            return new Packed(rows, cols, bits, matrix.group(), flags, f.superGroup() / matrix.group(), f.shift(), matrix,
                    levels, hi, f.scaleCodes() == null ? null : uploadBytes(f.scaleCodes()),
                    f.offsetCodes() == null ? null : uploadBytes(f.offsetCodes()), uploadHalves(f.superScales()),
                    f.superOffsets() == null ? null : uploadHalves(f.superOffsets()));
        }
        flags |= matrix.offsets() != null ? Packed.OFFSETS : 0;
        if (PACKED_FACTORS && halves(matrix.scales()) && (matrix.offsets() == null || halves(matrix.offsets()))) {
            // float16 values per group, as IQ4_NL and MXFP4 store them
            return new Packed(rows, cols, bits, matrix.group(), flags | Packed.HALF, 1, 0, matrix, levels, hi, null, null,
                    uploadHalves(matrix.scales()), matrix.offsets() == null ? null : uploadHalves(matrix.offsets()));
        }
        return new Packed(rows, cols, bits, matrix.group(), flags, 1, 0, matrix, levels, hi, null, null,
                vs(upload(matrix.scales())), matrix.offsets() == null ? null : vs(upload(matrix.offsets())));
    }

    /** @return the distinct levels in increasing order if there are at most 16, else null */
    private static byte[] codebook(byte[] levels) {
        // the levels seen, as a 256-bit set, gathered in parallel over chunks; a chunk with more
        // than 16 on its own (an 8-bit format) stops the others early
        int chunk = 1 << 20;
        java.util.concurrent.atomic.AtomicBoolean tooMany = new java.util.concurrent.atomic.AtomicBoolean();
        long[] seen = java.util.stream.IntStream.range(0, (int) Math.ceilDiv((long) levels.length, chunk)).parallel()
                .mapToObj(part -> {
                    long[] bits = new long[4];
                    int count = 0;
                    for (int i = part * chunk, end = Math.min(levels.length, i + chunk); i < end; i++) {
                        int v = levels[i] & 0xFF;
                        long bit = 1L << v;
                        if ((bits[v >> 6] & bit) == 0) {
                            bits[v >> 6] |= bit;
                            if (++count > 16 || tooMany.get()) {
                                tooMany.set(true);
                                break;
                            }
                        }
                    }
                    return bits;
                })
                .reduce(new long[4], (a, b) -> new long[] {a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]});
        int count = Long.bitCount(seen[0]) + Long.bitCount(seen[1]) + Long.bitCount(seen[2]) + Long.bitCount(seen[3]);
        if (tooMany.get() || count > 16) {
            return null;
        }
        byte[] table = new byte[count];
        int at = 0;
        for (int v = -128; v < 128; v++) {
            if ((seen[(v & 0xFF) >> 6] & 1L << v) != 0) {
                table[at++] = (byte) v;
            }
        }
        return table;
    }

    /** Whether every value is exactly a float16. */
    private static boolean halves(float[] values) {
        for (float v : values) {
            if (Float.float16ToFloat(Float.floatToFloat16(v)) != v) {
                return false;
            }
        }
        return true;
    }

    private VkStorage uploadBytes(byte[] data) {
        VkStorage s = raw(Math.max(1, (data.length + 3) / 4), null);
        vk.upload(s.slot(), java.lang.foreign.MemorySegment.ofArray(data));
        return s;
    }

    private VkStorage uploadHalves(float[] values) {
        short[] h = new short[values.length];
        for (int i = 0; i < h.length; i++) {
            h[i] = Float.floatToFloat16(values[i]);
        }
        VkStorage s = raw(Math.max(1, (h.length + 1) / 2), null);
        if (java.nio.ByteOrder.nativeOrder() == java.nio.ByteOrder.LITTLE_ENDIAN) {
            vk.upload(s.slot(), java.lang.foreign.MemorySegment.ofArray(h));
        } else {
            vk.upload(s.slot(), 2L * h.length, b -> b.asShortBuffer().put(h));
        }
        return s;
    }

    @Override
    public synchronized PackedStorage quantize(Storage values, int rows, int cols, int bits, int group) {
        VkStorage x = vs(values);
        if (bits <= 4 && rows % group == 0 && group % 2 == 0 && x.dtype == DType.FLOAT32
                && x.size == (long) rows * cols) {
            // levels on the host, two per byte on the device; no host copy is kept, so a large
            // model quantized at load takes no Java heap
            PackedMatrix m = PackedMatrix.quantize(download(values), rows, cols, bits, group);
            VkStorage levels = uploadBytes(m.data());
            VkStorage scales = vs(upload(m.scales()));
            return new Packed(rows, cols, bits, group, Packed.NIBBLES, 1, 0, null, levels, null, null, null, scales,
                    null);
        }
        if (bits != 8 || cols % 4 != 0 || rows % group != 0 || x.dtype != DType.FLOAT32 || x.size != (long) rows * cols) {
            return Backend.super.quantize(values, rows, cols, bits, group);
        }
        eager(x);
        VkStorage levels = raw((long) rows * cols / 4, null);
        VkStorage scales = raw((long) rows / group * cols, DType.FLOAT32, null);
        long work = (long) rows / group * (cols / 4);
        vk.push().putLong(x.address()).putLong(levels.address()).putLong(scales.address()).putInt(cols / 4)
                .putInt(group).putInt((int) work);
        vk.dispatch(shader("quantize8", DType.FLOAT32), groups(work), new VkContext.Slot[] {levels.slot(), scales.slot()},
                x.slot());
        return new Packed(rows, cols, bits, group, 0, 1, 0, null, levels, null, null, null, scales, null);
    }

    @Override
    public PackedMatrix unpackToHost(PackedStorage packed) {
        if (packed instanceof Deferred d) {
            d.skip = true;
            return d.host;
        }
        if (packed instanceof Packed p) {
            if (p.host() != null) {
                return p.host();
            }
            float[] scales = download(p.scales());
            float[] offsets = p.offsets() == null ? null : download(p.offsets());
            byte[] levels = new byte[(int) p.size()];
            if ((p.flags() & Packed.NIBBLES) != 0) {
                byte[] data = new byte[p.rows() / 2 * p.cols()];
                synchronized (this) {
                    eager(p.levels());
                    vk.download(p.levels().slot(), p.levels().bytes()).get(data);
                }
                for (int r = 0; r < p.rows(); r++) {
                    for (int c = 0; c < p.cols(); c++) {
                        int b = data[r / 2 * p.cols() + c];
                        levels[r * p.cols() + c] = (byte) ((r & 1) == 0 ? (b << 28) >> 28 : (b << 24) >> 28);
                    }
                }
            } else {
                synchronized (this) {
                    eager(p.levels());
                    vk.download(p.levels().slot(), p.levels().bytes()).get(levels);
                }
            }
            return PackedMatrix.of(levels, scales, offsets, p.rows(), p.cols(), p.bits(), p.group());
        }
        return Backend.super.unpackToHost(packed);
    }

    private synchronized Packed packed(PackedStorage s) {
        if (s instanceof Deferred d && d.owner == this) {
            if (d.done == null) {
                d.done = uploadPacked(d.host);
            }
            return d.done;
        }
        if (s instanceof Packed p && p.levels().owner == this) {
            return p;
        }
        return uploadPacked(unpackToHost(s));   // packed for another backend: bring it here
    }

    @Override
    public synchronized void unpack(PackedStorage packed, Storage out) {
        if (packed instanceof QRows r) {
            unpackRows(r, vs(out));
            return;
        }
        if (packed instanceof QuantizedRows) {
            Backend.super.unpack(packed, out);
            return;
        }
        Packed p = packed(packed);
        VkStorage z = vs(out);
        eager(z);
        java.nio.ByteBuffer b = vk.push();
        p.push(b);
        b.putLong(z.address()).putInt(p.rows()).putInt(p.cols()).putInt(p.group()).putInt(p.layout())
                .putInt(p.perSuper()).putFloat(p.shift());
        vk.dispatch(shader("unpack_packed", z.dtype), groups(p.size()), z.slot(), p.reads());
    }

    @Override
    public synchronized void matmulPacked(Storage a, PackedStorage b, int m, int k, int n, Storage out) {
        if (m > PACKED_ROWS && !(b instanceof QuantizedRows) && k > 0 && n > 0 && m <= PACKED_TILE_ROWS) {
            Packed p = packed(b);
            VkStorage z = vs(out);
            eager(vs(a), z);
            VkStorage x = as(vs(a), z.dtype);
            matmulPackedTiled(x, p, m, k, n, z);
            done(x, vs(a));
            return;
        }
        if (m > PACKED_ROWS && !(b instanceof QuantizedRows) && k > 0) {
            // unpacked into a temporary, freed as soon as the product is recorded: its slot
            // goes back to the pool, and the hazard tracking orders any reuse after this read.
            // Left to the garbage collector, a prefill held one per layer at once.
            VkStorage w = raw((long) k * n, vs(out).dtype, null);
            unpack(b, w);
            matmulBatched(a, w, 1, m, k, n, out);
            w.release();
            return;
        }
        if (m > PACKED_ROWS || m == 0 || k == 0) {
            Backend.super.matmulPacked(a, b, m, k, n, out);
            return;
        }
        Packed p = packed(b);
        VkStorage z = vs(out);
        eager(vs(a), z);
        VkStorage x = as(vs(a), z.dtype);
        // one row, sixteen columns per invocation from 16-byte loads, where the tuning chose it
        boolean wide = m == 1 && n % 16 == 0 && ((p.layout() >> 8) & 15) == 0 && tuning.get("packedWide") != 0;
        int columnGroups = ceil(n, wide ? 512 : 128);
        // enough pieces to fill the device, of at least PACKED_PIECE rows each, as matvec does:
        // with 256 a 896-wide model ran only 28 work groups and decoded at half f16's speed
        int pieces = Math.max(1, Math.min(ceil(k, (int) tuning.get("packedPiece")),
                (int) tuning.get("packedGroups") / columnGroups));
        int chunk = ceil(ceil(k, pieces), 2) * 2;
        pieces = ceil(k, chunk);
        int rows = m == 1 ? 1 : m == 2 ? 2 : m <= 4 ? 4 : 8;
        Pieces split = pieces(pieces, m, n, columnGroups, z.dtype);
        java.nio.ByteBuffer push = vk.push().putLong(x.address());
        p.push(push);
        push.putLong(z.address()).putLong(split.partsAddress()).putLong(counters.address()).putInt(m).putInt(k)
                .putInt(n).putInt(p.group()).putInt(p.layout()).putInt(chunk).putInt(pieces).putInt(split.counter())
                .putInt(p.perSuper()).putFloat(p.shift());
        vk.dispatch(shader(wide ? "matmul_packed_wide" : "matmul_packed" + rows, z.dtype), columnGroups, 1, pieces,
                split.writes(z), p.reads(x));
        split.release();
        done(x, vs(a));
    }

    /**
     * Whether products of more rows than {@link #PACKED_ROWS} stage a packed matrix's values in
     * the tiles from its levels ({@code -Dbrainiac.vulkan.packedTiles=false}: unpack it first).
     */
    private static final boolean PACKED_TILES =
            !"false".equalsIgnoreCase(System.getProperty("brainiac.vulkan.packedTiles"));
    /** The most rows for which they do ({@code -Dbrainiac.vulkan.packedTileRows}); more unpack first. */
    private static final int PACKED_TILE_ROWS =
            PACKED_TILES ? Integer.getInteger("brainiac.vulkan.packedTileRows", Integer.MAX_VALUE) : 0;

    /** matmul's tiles over a packed matrix: no unpacked copy of it, read as levels. */
    private void matmulPackedTiled(VkStorage x, Packed p, int m, int k, int n, VkStorage z) {
        boolean coop = coop(z.dtype, m);
        Tuning.CoopShape shape = coop ? coopShape() : null;
        int tileM = coop ? shape.rows() : (long) ceil(m, 64) * ceil(n, 64) >= bigTiles() ? 64 : 32;
        int tileN = coop ? shape.cols() : tileM;
        long tiles = (long) ceil(m, tileM) * ceil(n, tileN);
        int splits = 1;
        int chunk = k;
        if (tiles < splitBelow() && k >= 1024) {
            splits = (int) Math.min(ceil(k, 256), Math.max(2, 2 * splitBelow() / tiles));
            // whole steps of the tiles (up to 64 values of k): their scales stage once per step
            chunk = ceil(ceil(k, splits), 64) * 64;
            splits = ceil(k, chunk);
        }
        long mn = (long) m * n;
        VkStorage target = splits == 1 ? z : raw(splits * mn, z.dtype, null);
        java.nio.ByteBuffer push = vk.push().putLong(x.address());
        p.push(push);
        push.putLong(target.address()).putInt(m).putInt(k).putInt(n).putInt((int) mn).putInt(splits).putInt(chunk)
                .putInt(p.group()).putInt(p.layout()).putInt(p.perSuper()).putFloat(p.shift());
        String name = coop ? "matmul_coop_packed" + shape.suffix() : tileM == 64 ? "matmul64p" : "matmul32p";
        vk.dispatch(shader(name, z.dtype), ceil(n, tileN), ceil(m, tileM), splits, target.slot(), p.reads(x));
        if (splits > 1) {
            reduce(ReduceOp.SUM.ordinal(), target, z, mn, splits, mn);
            target.release();
        }
    }

    // ------------------------------------------------------------------ windows and indices

    @Override
    public synchronized void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        VkStorage z = vs(out);
        eager(vs(in), z);
        VkStorage x = as(vs(in), z.dtype);
        Tables t = tables(geometry);
        int ks = geometry.kernelSize();
        long total = (long) geometry.batch() * geometry.positionCount() * geometry.channels() * ks;
        vk.push().putLong(x.address()).putLong(t.im2col.address()).putLong(z.address()).putInt(geometry.channels())
                .putInt(geometry.positionCount()).putInt(ks).putInt(geometry.imageSize()).putFloat(padValue)
                .putInt((int) total);
        vk.dispatch(shader("im2col", z.dtype), groups(total), z.slot(), x.slot(), t.im2col.slot());
        done(x, vs(in));
    }

    @Override
    public synchronized void col2im(Storage cols, ConvGeometry geometry, Storage out) {
        VkStorage z = vs(out);
        eager(vs(cols), z);
        VkStorage c = as(vs(cols), z.dtype);
        Tables t = tables(geometry);
        long total = (long) geometry.batch() * geometry.channels() * geometry.imageSize();
        vk.push().putLong(c.address()).putLong(t.offsets.address()).putLong(t.entries.address())
                .putLong(z.address()).putInt(geometry.channels()).putInt(geometry.positionCount())
                .putInt(geometry.kernelSize()).putInt(geometry.imageSize()).putInt((int) total);
        vk.dispatch(shader("col2im", z.dtype), groups(total), z.slot(), c.slot(), t.offsets.slot(), t.entries.slot());
        done(c, vs(cols));
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
        // evicted tables are freed by the cleaner once no recorded command can still need them:
        // a slot is only reused by commands recorded later, which run after
        t = new Tables(uploadInts(table), uploadInts(offsets), uploadInts(entries));
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

    /** The indices as whole numbers, checked; read from the upload when it was kept. */
    private int[] indices(VkStorage storage, int count, long length) {
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
        VkStorage z = vs(out);
        eager(vs(in), vs(indices), z);
        VkStorage x = as(vs(in), z.dtype);
        long bound = vs(indices).indexBound;
        if (bound <= 0 || bound > length) {
            indices(vs(indices), count, length);   // made on the device in range: no download
        }
        VkStorage ind = as(vs(indices), DType.FLOAT32);
        long total = outer * count * inner;
        vk.push().putLong(x.address()).putLong(ind.address()).putLong(z.address()).putInt((int) length)
                .putInt((int) inner).putInt(count).putInt((int) total);
        vk.dispatch(shader("gather", z.dtype), groups(total), z.slot(), x.slot(), ind.slot());
        done(x, vs(in));
        done(ind, vs(indices));
    }

    /**
     * A bitonic sort on the device: each row is copied into a power-of-two wide work row, a
     * compare-exchange pass runs per (stage, distance), and the first {@code length} entries
     * are copied out, so the whole sort is {@code log2(width) * (log2(width) + 1) / 2 + 2}
     * dispatches with no host round trip.
     */
    @Override
    public synchronized void sort(Storage in, long rows, int length, boolean descending, Storage values,
            Storage indices) {
        VkStorage v = vs(values);
        VkStorage ix = vs(indices);
        eager(vs(in), v, ix);
        int width = Integer.highestOneBit(Math.max(1, length - 1)) << 1;
        if (length == 1) {
            width = 1;
        }
        long total = rows * width;
        if (total > Integer.MAX_VALUE || rows * (long) length == 0) {
            Backend.super.sort(in, rows, length, descending, values, indices);
            return;
        }
        VkStorage x = as(vs(in), v.dtype);
        VkStorage key = raw(total, v.dtype, null);
        VkStorage idx = raw(total, DType.FLOAT32, null);
        vk.push().putLong(x.address()).putLong(key.address()).putLong(idx.address()).putInt((int) total)
                .putInt(length).putInt(width);
        vk.dispatch(shader("sort_init", v.dtype), groups(total), new VkContext.Slot[] {key.slot(), idx.slot()},
                x.slot());
        long pairs = total / 2;
        for (int k = 2; k <= width; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                vk.push().putLong(key.address()).putLong(idx.address()).putInt((int) pairs).putInt(width).putInt(j)
                        .putInt(k).putInt(length).putInt(descending ? 1 : 0);
                vk.dispatch(shader("sort_step", v.dtype), groups(pairs), new VkContext.Slot[] {key.slot(), idx.slot()});
            }
        }
        long count = rows * length;
        vk.push().putLong(key.address()).putLong(idx.address()).putLong(v.address()).putLong(ix.address())
                .putInt((int) count).putInt(length).putInt(width);
        vk.dispatch(shader("sort_out", v.dtype), groups(count), new VkContext.Slot[] {v.slot(), ix.slot()},
                key.slot(), idx.slot());
        key.release();
        idx.release();
        done(x, vs(in));
    }

    /** Pipelines of user kernels, by kernel, number format and source. */
    private final Map<String, String> customShaders = new HashMap<>();

    /**
     * Runs a user kernel's shader, compiled on first use for the output's format; kernels
     * without one, or with shader compilation unavailable, run their Java version.
     */
    @Override
    public synchronized void custom(io.github.zzuegg.brainiac.tensor.Kernel kernel, Storage[] inputs, Storage out,
            int[] ints, float[] floats) {
        if (kernel.glsl() == null || compiler == null) {
            Backend.super.custom(kernel, inputs, out, ints, floats);
            return;
        }
        VkStorage y = vs(out);
        VkStorage[] originals = new VkStorage[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            originals[i] = vs(inputs[i]);
        }
        VkStorage[] all = java.util.Arrays.copyOf(originals, inputs.length + 1);
        all[inputs.length] = y;
        eager(all);
        String key = kernel.name() + "\n" + y.dtype + "\n" + kernel.inputs() + "\n" + kernel.glsl();
        String name = customShaders.get(key);
        if (name == null) {
            byte[] code = compiler.compile(compiler.source(kernel.header() + kernel.glsl(), y.dtype));
            name = "custom#" + kernel.name() + "#" + customShaders.size();
            vk.register(name, code);
            customShaders.put(key, name);
        }
        VkStorage[] xs = new VkStorage[inputs.length];
        VkContext.Slot[] reads = new VkContext.Slot[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            xs[i] = as(originals[i], y.dtype);
            reads[i] = xs[i].slot();
        }
        ByteBuffer p = vk.push().putLong(y.address());
        for (VkStorage x : xs) {
            p.putLong(x.address());
        }
        p.putInt((int) y.size);
        for (int i = 0; i < io.github.zzuegg.brainiac.tensor.Kernel.MAX_INTS; i++) {
            p.putInt(i < ints.length ? ints[i] : 0);
        }
        for (int i = 0; i < io.github.zzuegg.brainiac.tensor.Kernel.MAX_FLOATS; i++) {
            p.putFloat(i < floats.length ? floats[i] : 0);
        }
        vk.dispatch(name, groups(y.size), y.slot(), reads);
        for (int i = 0; i < inputs.length; i++) {
            done(xs[i], originals[i]);
        }
    }

    /** Most slices an index list is split into for {@link #scatterAdd} ({@code -Dbrainiac.vulkan.scatterSlices}; 1: none). */
    private static final int SCATTER_SLICES = Integer.getInteger("brainiac.vulkan.scatterSlices", 32);
    /** Work groups {@link #scatterAdd} aims for when it splits the index lists. */
    private static final long SCATTER_GROUPS = 2048;

    @Override
    public synchronized void scatterAdd(Storage src, long outer, int count, long inner, Storage indices, long length,
            Storage out) {
        VkStorage z = vs(out);
        eager(vs(src), vs(indices), z);
        VkStorage s = as(vs(src), z.dtype);
        int[] idx = indices(vs(indices), count, length);
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
        VkStorage off = uploadInts(offsets);
        VkStorage ent = uploadInts(entries);
        long total = outer * length * inner;
        // Few targets with many entries each (a small vocabulary) leave most of the device idle
        // while each invocation adds up a long list: then slices of the lists are added up
        // side by side and summed afterwards, always in the same order.
        int most = 0;
        for (int l = 0; l < len; l++) {
            most = Math.max(most, offsets[l + 1] - offsets[l]);
        }
        int slices = (int) Math.max(1, Math.min(Math.min(SCATTER_SLICES, most / 8),
                SCATTER_GROUPS / Math.max(1, groups(total))));
        VkStorage target = slices == 1 ? z : raw(total * slices, z.dtype, null);
        vk.push().putLong(s.address()).putLong(off.address()).putLong(ent.address()).putLong(target.address())
                .putInt(len).putInt((int) inner).putInt(count).putInt((int) total).putInt(slices);
        vk.dispatch(shader("scatter_add", z.dtype), groups(total), slices, target.slot(), s.slot(), off.slot(),
                ent.slot());
        if (slices > 1) {
            reduce(ReduceOp.SUM.ordinal(), target, z, total, slices, total);
            target.release();
        }
        done(s, vs(src));
        off.release();
        ent.release();
    }
}
