package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.PackedStorage;
import io.github.zzuegg.brainiac.tensor.spi.RawValues;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
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
                vk = new VkContext();
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
                if (Boolean.getBoolean("brainiac.vulkan.statistics")) {
                    Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                        if (vk.dispatches > 0) {
                            System.err.println(name() + ": " + statistics());
                        }
                    }, "vulkan-statistics"));
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
                + (compiler != null ? "on" : FUSION ? "unavailable (" + fusionUnavailable + ")" : "off") + "; devices "
                + vk.deviceList;
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
        return vk.dispatches + " dispatches (" + fusedDispatches + " fused, " + fusedShaders.size()
                + " shaders generated in " + compileNanos / 1_000_000 + " ms), " + vk.barriers + " barriers, " + vk.submits + " submits, " + vk.waits
                + " waits, " + vk.reclaims + " reclaims, " + vk.collections + " collections, " + (vk.allocated() >> 20) + " MB allocated, " + (vk.peakInUse() >> 20) + " MB peak in use";
    }

    /** Commands recorded so far, fused or not; for tests. */
    synchronized long dispatchCount() {
        return vk == null ? 0 : vk.dispatches;
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

    /** Whether an element-wise result can be recorded instead of computed now. */
    private boolean fusable(VkStorage out, Storage... inputs) {
        if (compiler == null || out.size == 0 || (out.dtype != DType.FLOAT32 && out.dtype != DType.FLOAT64)) {
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
                && v.strides()[0] == 1) {
            // a contiguous range of stored values
            int bytes = s.dtype.bytes();
            vk.copy(v.storage().slot(), (long) v.offset() * bytes, s.slot(), 0, s.size * bytes);
            return;
        }
        Fusion.Kernel k = Fusion.generate(e);
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
        String key = dtype == DType.FLOAT64 ? "d" + body : body;
        String name = fusedShaders.get(key);
        if (name == null) {
            long start = System.nanoTime();
            byte[] code = compiler.compile(compiler.source(body, dtype));
            name = "fused#" + fusedShaders.size();
            vk.register(name, code);
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
        if (storage instanceof Packed) {
            return;   // weights: freed with the tensor that holds them
        }
        VkStorage s = vs(storage);
        if (s.pending == null) {
            s.release();   // pending results may still be inlined elsewhere: left to the collector
        }
    }

    @Override
    public synchronized Storage upload(float[] data) {
        VkStorage s = raw(data.length, data.length <= 4096 ? data.clone() : null);
        vk.upload(s.slot(), s.bytes(), b -> b.asFloatBuffer().put(data));
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
        vk.upload(s.slot(), data.byteSize(), b -> java.lang.foreign.MemorySegment.ofBuffer(b)
                .copyFrom(data));
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
        eager(vs(in), z);
        VkStorage x = as(vs(in), z.dtype);
        copy(x, inOffset, inStrides, shape, z, outOffset, outStrides, n);
        done(x, vs(in));
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
            vk.push().putLong(x.address()).putLong(y.address()).putInt((int) rows).putInt((int) length)
                    .putInt((int) inner);
            vk.dispatch(shader("argmax", y.dtype), groups(rows), y.slot(), x.slot());
        } else {
            reduce(op.ordinal(), x, y, rows, length, inner);
        }
        done(x, vs(in));
    }

    private void reduce(int op, VkStorage x, VkStorage y, long rows, long length, long inner) {
        vk.push().putLong(x.address()).putLong(y.address()).putInt((int) rows).putInt((int) length)
                .putInt((int) inner).putInt(op);
        if (rows < 1024 && length > 256) {
            vk.dispatch(shader("reduce_group", y.dtype), rows, y.slot(), x.slot());
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
    private static final long BIG_TILES = Long.getLong("brainiac.vulkan.bigTileGroups", 128);
    /** Products with fewer tiles than this, and a long k, split k into pieces summed afterwards. */
    private static final long SPLIT_BELOW = 64;

    @Override
    public synchronized void matmulBatched(Storage a, Storage b, int batch, int m, int k, int n, Storage out) {
        VkStorage z = vs(out);
        if ((long) batch * m * n == 0) {
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

    private void matmul(VkStorage x, VkStorage y, int batch, int m, int k, int n, VkStorage z) {
        long big = (long) batch * ceil(m, 64) * ceil(n, 64);
        int tile = big >= BIG_TILES ? 64 : 32;
        long tiles = (long) batch * ceil(m, tile) * ceil(n, tile);
        int splits = 1;
        int chunk = k;
        if (tiles < SPLIT_BELOW && k >= 1024) {
            splits = (int) Math.min(ceil(k, 256), Math.max(2, 2 * SPLIT_BELOW / tiles));
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
            vk.push().putLong(x.address(firstBatch * m * k)).putLong(y.address(firstBatch * k * n))
                    .putLong(target.address(first * mn)).putInt(m).putInt(k).putInt(n).putInt(m * k).putInt(k * n)
                    .putInt((int) mn).putInt(splits).putInt(chunk);
            vk.dispatch(shader(tile == 64 ? "matmul64" : "matmul32", z.dtype), ceil(n, tile), ceil(m, tile), count,
                    target.slot(), x.slot(), y.slot());
        }
        if (splits > 1) {
            reduce(ReduceOp.SUM.ordinal(), target, z, (long) batch * mn, splits, mn);
            target.release();
        }
    }

    private static int ceil(long a, int b) {
        return (int) ((a + b - 1) / b);
    }

    // ------------------------------------------------------------------ packed weights

    /** A packed matrix on the device: levels as raw bytes (read as 32-bit words) and scales. */
    private record Packed(PackedMatrix host, VkStorage levels, VkStorage scales) implements PackedStorage {

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
            return host.size();
        }

        @Override
        public long bytes() {
            return levels.bytes() + scales.bytes();
        }
    }

    /** Rows up to which a product reads packed levels directly; more unpack first. */
    private static final int PACKED_ROWS = 8;
    /** Products by packed weights split their rows into pieces until about this many work groups run. */
    private static final int PACKED_GROUPS = Integer.getInteger("brainiac.vulkan.packedGroups", 256);

    @Override
    public synchronized PackedStorage pack(PackedMatrix matrix) {
        byte[] data = matrix.data();
        VkStorage levels = raw(Math.max(1, (data.length + 3) / 4), null);
        vk.upload(levels.slot(), levels.bytes(), b -> b.put(data));
        return new Packed(matrix, levels, vs(upload(matrix.scales())));
    }

    @Override
    public PackedMatrix unpackToHost(PackedStorage packed) {
        if (packed instanceof Packed p) {
            return p.host();
        }
        return Backend.super.unpackToHost(packed);
    }

    private Packed packed(PackedStorage s) {
        if (s instanceof Packed p && p.levels().owner == this) {
            return p;
        }
        return (Packed) pack(unpackToHost(s));   // packed for another backend: bring it here
    }

    @Override
    public synchronized void unpack(PackedStorage packed, Storage out) {
        Packed p = packed(packed);
        VkStorage z = vs(out);
        eager(z);
        PackedMatrix h = p.host();
        vk.push().putLong(p.levels().address()).putLong(p.scales().address()).putLong(z.address())
                .putInt(h.rows()).putInt(h.cols()).putInt(h.group()).putInt(h.nibbles() ? 1 : 0);
        vk.dispatch(shader("unpack_packed", z.dtype), groups(h.size()), z.slot(), p.levels().slot(),
                p.scales().slot());
    }

    @Override
    public synchronized void matmulPacked(Storage a, PackedStorage b, int m, int k, int n, Storage out) {
        if (m > PACKED_ROWS || m == 0 || k == 0) {
            Backend.super.matmulPacked(a, b, m, k, n, out);
            return;
        }
        Packed p = packed(b);
        VkStorage z = vs(out);
        eager(vs(a), z);
        VkStorage x = as(vs(a), z.dtype);
        int columnGroups = ceil(n, 128);
        // enough pieces to fill the device, of at least 256 rows each: every piece costs a pass
        // that adds them up, which small products would mostly spend waiting
        int pieces = Math.max(1, Math.min(ceil(k, 256), PACKED_GROUPS / columnGroups));
        int chunk = ceil(ceil(k, pieces), 2) * 2;
        pieces = ceil(k, chunk);
        long mn = (long) m * n;
        VkStorage target = pieces == 1 ? z : raw(pieces * mn, z.dtype, null);
        int rows = m == 1 ? 1 : m == 2 ? 2 : m <= 4 ? 4 : 8;
        vk.push().putLong(x.address()).putLong(p.levels().address()).putLong(p.scales().address())
                .putLong(target.address()).putInt(m).putInt(k).putInt(n).putInt(p.host().group())
                .putInt(p.host().nibbles() ? 1 : 0).putInt(chunk);
        vk.dispatch(shader("matmul_packed" + rows, z.dtype), columnGroups, 1, pieces, target.slot(), x.slot(),
                p.levels().slot(), p.scales().slot());
        if (pieces > 1) {
            reduce(ReduceOp.SUM.ordinal(), target, z, mn, pieces, mn);
            target.release();
        }
        done(x, vs(a));
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
        indices(vs(indices), count, length);
        VkStorage ind = as(vs(indices), DType.FLOAT32);
        long total = outer * count * inner;
        vk.push().putLong(x.address()).putLong(ind.address()).putLong(z.address()).putInt((int) length)
                .putInt((int) inner).putInt(count).putInt((int) total);
        vk.dispatch(shader("gather", z.dtype), groups(total), z.slot(), x.slot(), ind.slot());
        done(x, vs(in));
        done(ind, vs(indices));
    }

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
        vk.push().putLong(s.address()).putLong(off.address()).putLong(ent.address()).putLong(z.address())
                .putInt(len).putInt((int) inner).putInt(count).putInt((int) total);
        vk.dispatch(shader("scatter_add", z.dtype), groups(total), z.slot(), s.slot(), off.slot(), ent.slot());
        done(s, vs(src));
        off.release();
        ent.release();
    }
}
