package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A GPU backend on Vulkan compute shaders, called through LWJGL. Works with any Vulkan 1.2
 * driver that supports buffer device addresses and 64-bit shader integers, which covers
 * current AMD, NVIDIA and Intel GPUs.
 *
 * <p>Stores float32 tensors. Operations are recorded into command buffers and submitted in
 * batches, so the host rarely waits for the device: only when values are read back.
 *
 * <p>System properties: {@code brainiac.vulkan.device} picks a device by index (the list is
 * in {@link #status()}); {@code brainiac.vulkan.devices=all} also allows CPU drivers such as
 * lavapipe; {@code brainiac.vulkan.memory} sets the memory budget in MB;
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
                synchronized (VulkanBackend.class) {
                    if (first == null) {
                        first = this;
                    }
                }
                if (Boolean.getBoolean("brainiac.vulkan.statistics")) {
                    Runtime.getRuntime().addShutdownHook(new Thread(
                            () -> System.err.println(name() + ": " + statistics()), "vulkan-statistics"));
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
        return vk.deviceName + ", " + (vk.deviceMemory >> 20) + " MB, " + (vk.budget >> 20) + " MB budget; devices "
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
        return vk.dispatches + " dispatches, " + vk.barriers + " barriers, " + vk.submits + " submits, " + vk.waits
                + " waits, " + vk.reclaims + " reclaims, " + vk.collections + " collections, " + (vk.allocated() >> 20) + " MB allocated, " + (vk.peakInUse() >> 20) + " MB peak in use";
    }

    /** Waits until all recorded work is done; for timing. */
    public synchronized void finish() {
        if (vk != null) {
            vk.finish();
        }
    }

    // ------------------------------------------------------------------ storage

    private VkStorage raw(long size, float[] host) {
        if (size > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("the vulkan backend holds at most 2^31 values per tensor");
        }
        isAvailable();
        return new VkStorage(this, vk.allocate(size * Float.BYTES), size, host, CLEANER, vk);
    }

    private VkStorage vs(Storage s) {
        if (s instanceof VkStorage v && v.owner == this) {
            return v;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to " + name());
    }

    @Override
    public boolean supports(DType dtype) {
        return dtype == DType.FLOAT32;
    }

    @Override
    public synchronized Storage allocate(long size) {
        VkStorage s = raw(size, null);
        vk.fill(s.slot, 0, s.bytes(), 0);
        return s;
    }

    @Override
    public synchronized Storage allocateUninitialized(long size, DType dtype) {
        if (dtype != DType.FLOAT32) {
            return allocate(size, dtype);
        }
        return raw(size, null);
    }

    @Override
    public synchronized void release(Storage storage) {
        vs(storage).release();
    }

    @Override
    public synchronized Storage upload(float[] data) {
        VkStorage s = raw(data.length, data.length <= 4096 ? data.clone() : null);
        vk.upload(s.slot, s.bytes(), b -> b.asFloatBuffer().put(data));
        return s;
    }

    private VkStorage uploadInts(int[] data) {
        VkStorage s = raw(Math.max(1, data.length), null);
        vk.upload(s.slot, (long) data.length * Integer.BYTES, b -> b.asIntBuffer().put(data));
        return s;
    }

    @Override
    public synchronized float[] download(Storage src) {
        VkStorage s = vs(src);
        float[] out = new float[Math.toIntExact(s.size)];
        ByteBuffer b = vk.download(s.slot, s.bytes());
        b.asFloatBuffer().get(out);
        return out;
    }

    @Override
    public synchronized void fill(Storage dst, float value) {
        VkStorage s = vs(dst);
        vk.fill(s.slot, 0, s.bytes(), Float.floatToRawIntBits(value));
    }

    @Override
    public synchronized Storage random(long size, long seed) {
        VkStorage s = raw(size, null);
        vk.push().putLong(s.address()).putLong(seed).putInt((int) size);
        vk.dispatch("random", groups(size), s.slot);
        return s;
    }

    private static long groups(long n) {
        return (n + GROUP - 1) / GROUP;
    }

    // ------------------------------------------------------------------ element-wise

    @Override
    public synchronized void unary(UnaryOp op, Storage in, Storage out) {
        VkStorage x = vs(in);
        VkStorage y = vs(out);
        vk.push().putLong(x.address()).putLong(y.address()).putInt((int) y.size).putInt(op.ordinal());
        vk.dispatch("unary", groups(y.size), y.slot, x.slot);
    }

    @Override
    public synchronized void activation(ActivationOp op, float param, Storage in, Storage out) {
        VkStorage x = vs(in);
        VkStorage y = vs(out);
        vk.push().putLong(x.address()).putLong(y.address()).putInt((int) y.size).putInt(op.ordinal()).putFloat(param);
        vk.dispatch("activation", groups(y.size), y.slot, x.slot);
    }

    @Override
    public synchronized void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out) {
        VkStorage x = vs(in);
        VkStorage g = vs(grad);
        VkStorage y = vs(out);
        vk.push().putLong(x.address()).putLong(g.address()).putLong(y.address()).putInt((int) y.size)
                .putInt(op.ordinal()).putFloat(param);
        vk.dispatch("activation_backward", groups(y.size), y.slot, x.slot, g.slot);
    }

    @Override
    public synchronized void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape,
            Storage out) {
        VkStorage x = vs(a);
        VkStorage y = vs(b);
        VkStorage z = vs(out);
        long n = z.size;
        if (n == 0) {
            return;
        }
        long[] contiguous = contiguous(shape);
        if (Arrays.equals(aStrides, contiguous) && Arrays.equals(bStrides, contiguous)) {
            vk.push().putLong(x.address()).putLong(y.address()).putLong(z.address()).putInt((int) n)
                    .putInt(op.ordinal());
            vk.dispatch("binary_contiguous", groups(n), z.slot, x.slot, y.slot);
            return;
        }
        Axes ax = Axes.merge(shape, aStrides, bStrides, contiguous);
        ax.forEachBlock((offsets, inner, count) -> {
            ByteBuffer p = vk.push().putLong(x.address(offsets[0])).putLong(y.address(offsets[1]))
                    .putLong(z.address(offsets[2])).putInt((int) count).putInt(op.ordinal()).putInt(inner)
                    .putInt(0).putInt(0);
            ax.put(p, inner);
            vk.dispatch("binary", groups(count), z.slot, x.slot, y.slot);
        });
    }

    @Override
    public void copy(Storage in, long[] strides, int[] shape, Storage out) {
        copy(in, 0, strides, shape, out, 0, contiguous(shape));
    }

    @Override
    public synchronized void copy(Storage in, long inOffset, long[] inStrides, int[] shape, Storage out,
            long outOffset, long[] outStrides) {
        VkStorage x = vs(in);
        VkStorage z = vs(out);
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n == 0) {
            return;
        }
        Axes ax = Axes.merge(shape, inStrides, outStrides, null);
        if (ax.rank() == 1 && ax.stride(0, 0) == 1 && ax.stride(1, 0) == 1) {
            vk.copy(x.slot, inOffset * Float.BYTES, z.slot, outOffset * Float.BYTES, n * Float.BYTES);
            return;
        }
        ax.forEachBlock((offsets, inner, count) -> {
            ByteBuffer p = vk.push().putLong(x.address(inOffset + offsets[0]))
                    .putLong(z.address(outOffset + offsets[1])).putInt((int) count).putInt(inner).putInt(0).putInt(0);
            ax.put(p, inner);
            vk.dispatch("copy", groups(count), z.slot, x.slot);
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
        VkStorage x = vs(in);
        VkStorage y = vs(out);
        long rows = outer * inner;
        if (rows == 0) {
            return;
        }
        if (op == ReduceOp.ARGMAX) {
            vk.push().putLong(x.address()).putLong(y.address()).putInt((int) rows).putInt((int) length)
                    .putInt((int) inner);
            vk.dispatch("argmax", groups(rows), y.slot, x.slot);
            return;
        }
        reduce(op.ordinal(), x, y, rows, length, inner);
    }

    private void reduce(int op, VkStorage x, VkStorage y, long rows, long length, long inner) {
        vk.push().putLong(x.address()).putLong(y.address()).putInt((int) rows).putInt((int) length)
                .putInt((int) inner).putInt(op);
        if (rows < 1024 && length > 256) {
            vk.dispatch("reduce_group", rows, y.slot, x.slot);
        } else {
            vk.dispatch("reduce", groups(rows), y.slot, x.slot);
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
        VkStorage x = vs(a);
        VkStorage y = vs(b);
        VkStorage z = vs(out);
        if ((long) batch * m * n == 0) {
            return;
        }
        if (k == 0) {
            vk.fill(z.slot, 0, z.bytes(), 0);
            return;
        }
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
        VkStorage target = splits == 1 ? z : raw((long) batch * splits * mn, null);
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
            vk.dispatch(tile == 64 ? "matmul64" : "matmul32", ceil(n, tile), ceil(m, tile), count, target.slot,
                    x.slot, y.slot);
        }
        if (splits > 1) {
            reduce(ReduceOp.SUM.ordinal(), target, z, (long) batch * mn, splits, mn);
            target.release();
        }
    }

    private static int ceil(long a, int b) {
        return (int) ((a + b - 1) / b);
    }

    // ------------------------------------------------------------------ windows and indices

    @Override
    public synchronized void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        VkStorage x = vs(in);
        VkStorage z = vs(out);
        Tables t = tables(geometry);
        int ks = geometry.kernelSize();
        long total = (long) geometry.batch() * geometry.positionCount() * geometry.channels() * ks;
        vk.push().putLong(x.address()).putLong(t.im2col.address()).putLong(z.address()).putInt(geometry.channels())
                .putInt(geometry.positionCount()).putInt(ks).putInt(geometry.imageSize()).putFloat(padValue)
                .putInt((int) total);
        vk.dispatch("im2col", groups(total), z.slot, x.slot, t.im2col.slot);
    }

    @Override
    public synchronized void col2im(Storage cols, ConvGeometry geometry, Storage out) {
        VkStorage c = vs(cols);
        VkStorage z = vs(out);
        Tables t = tables(geometry);
        long total = (long) geometry.batch() * geometry.channels() * geometry.imageSize();
        vk.push().putLong(c.address()).putLong(t.offsets.address()).putLong(t.entries.address())
                .putLong(z.address()).putInt(geometry.channels()).putInt(geometry.positionCount())
                .putInt(geometry.kernelSize()).putInt(geometry.imageSize()).putInt((int) total);
        vk.dispatch("col2im", groups(total), z.slot, c.slot, t.offsets.slot, t.entries.slot);
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
        float[] values = storage.host != null ? storage.host : download(storage);
        int[] idx = new int[count];
        for (int i = 0; i < count; i++) {
            float v = values[i];
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
        VkStorage x = vs(in);
        VkStorage ind = vs(indices);
        VkStorage z = vs(out);
        indices(ind, count, length);
        long total = outer * count * inner;
        vk.push().putLong(x.address()).putLong(ind.address()).putLong(z.address()).putInt((int) length)
                .putInt((int) inner).putInt(count).putInt((int) total);
        vk.dispatch("gather", groups(total), z.slot, x.slot, ind.slot);
    }

    @Override
    public synchronized void scatterAdd(Storage src, long outer, int count, long inner, Storage indices, long length,
            Storage out) {
        VkStorage s = vs(src);
        VkStorage z = vs(out);
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
        vk.dispatch("scatter_add", groups(total), z.slot, s.slot, off.slot, ent.slot);
        off.release();
        ent.release();
    }
}
