package io.github.zzuegg.brainiac.tensor.spi;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A weight matrix kept exactly as a GGUF file stores it: block-quantized rows, read in place
 * from the file's memory mapping. Nothing is copied or expanded, so a matrix takes its
 * file's size in memory (2 to 8.5 bits per weight), and the operating system pages it in
 * from disk when it is first read and can drop it again under memory pressure. RAM then
 * works as a cache over the file, and a model larger than RAM still runs, at the speed
 * the disk can deliver the weights that are not cached.
 *
 * <p>The file stores {@code out} rows of {@code in} values each, blocks along the inputs.
 * As every {@link PackedStorage}, this one is seen as the transposed {@code [in, out]}
 * matrix that {@code x.matmul(w)} multiplies by: {@link #rows()} is {@code in} and
 * {@link #cols()} is {@code out}.
 *
 * <p>Backends that have no kernel for it see the values through {@link #dequantize}. The
 * CPU backend multiplies by it directly, decoding one stored row at a time.
 */
public final class GgmlBlocks implements PackedStorage {

    /** Decodes whole blocks of one of GGML's formats into floats. */
    @FunctionalInterface
    public interface Decoder {
        /**
         * Decodes {@code count} values, whole blocks, from {@code blocks} into {@code out}.
         *
         * @param blocks the blocks, as stored
         * @param count the values, a multiple of the block size
         * @param out where to write them
         * @param at the first index written
         */
        void decode(MemorySegment blocks, int count, float[] out, int at);
    }

    private final MemorySegment data;
    private final String format;
    private final int in;
    private final int out;
    private final int blockValues;
    private final int blockBytes;
    private final long rowBytes;
    private final Decoder decoder;
    /**
     * {@link #READ} when this matrix was last {@linkplain #load loaded}, or -1. A mapped file
     * larger than memory keeps only what was read last in memory: once more than
     * {@link #RESIDENT} bytes were read after it, a matrix may have left memory and is loaded
     * again (which costs little where its pages stayed).
     */
    private volatile long warmAt = -1;

    /** The bytes of mapped matrices {@linkplain #load loaded} so far, by all matrices. */
    private static final AtomicLong READ = new AtomicLong();

    /**
     * The bytes of mapped files that stay in memory: {@code brainiac.mappedResidentBytes}, or
     * by default half of the machine's memory, leaving the rest to the program and the system.
     */
    private static final long RESIDENT = Long.getLong("brainiac.mappedResidentBytes", physicalMemory() / 2);

    /** Bytes per read when a matrix is loaded: many reads at once keep an SSD's queue full. */
    private static final long CHUNK = 1L << 20;

    private static long physicalMemory() {
        try {
            if (java.lang.management.ManagementFactory.getOperatingSystemMXBean()
                    instanceof com.sun.management.OperatingSystemMXBean os) {
                return os.getTotalMemorySize();
            }
        } catch (RuntimeException | LinkageError e) {
            // no management module: assume a modest machine
        }
        return 16L << 30;
    }

    /**
     * Wraps stored blocks.
     *
     * @param data the {@code out * in / blockValues * blockBytes} bytes, as stored
     * @param format the format's name, such as {@code "Q4_K"}
     * @param out the stored rows, the outputs of a product
     * @param in the values per stored row, the inputs
     * @param blockValues the values per block
     * @param blockBytes the bytes per block
     * @param decoder decodes the blocks
     */
    public GgmlBlocks(MemorySegment data, String format, int out, int in, int blockValues, int blockBytes,
            Decoder decoder) {
        this.data = Objects.requireNonNull(data, "data");
        this.format = Objects.requireNonNull(format, "format");
        this.decoder = Objects.requireNonNull(decoder, "decoder");
        if (out < 1 || in < 1 || blockValues < 1 || blockBytes < 1 || in % blockValues != 0) {
            throw new IllegalArgumentException("bad " + format + " matrix: " + out + " rows of " + in + " values, blocks of "
                    + blockValues);
        }
        this.out = out;
        this.in = in;
        this.blockValues = blockValues;
        this.blockBytes = blockBytes;
        this.rowBytes = (long) in / blockValues * blockBytes;
        if (data.byteSize() != rowBytes * out) {
            throw new IllegalArgumentException(format + " " + out + "x" + in + " takes " + rowBytes * out + " bytes, not "
                    + data.byteSize());
        }
    }

    /** @return the format's name, such as {@code "MXFP4"} */
    public String format() {
        return format;
    }

    /** @return the inputs {@code k}: the values per stored row */
    @Override
    public int rows() {
        return in;
    }

    /** @return the outputs {@code n}: the stored rows */
    @Override
    public int cols() {
        return out;
    }

    @Override
    public long size() {
        return (long) in * out;
    }

    @Override
    public long bytes() {
        return data.byteSize();
    }

    /** @return the values per block */
    public int blockValues() {
        return blockValues;
    }

    /** @return the bytes per block */
    public int blockBytes() {
        return blockBytes;
    }

    /** @return the bytes of one stored row */
    public long rowBytes() {
        return rowBytes;
    }

    /** @return the stored bytes, valid while the file they map is open */
    public MemorySegment data() {
        return data;
    }

    /**
     * Decodes stored row {@code row}, the weights of output {@code row} for every input.
     *
     * @param row the output, 0 to {@link #cols()}
     * @param dst at least {@link #rows()} floats
     */
    public void decodeRow(int row, float[] dst) {
        Objects.checkIndex(row, out);
        decoder.decode(data.asSlice(row * rowBytes, rowBytes), in, dst, 0);
    }

    /**
     * Computes the values of the {@code [in, out]} matrix this stands for, row-major.
     *
     * @param dst {@link #size()} floats
     */
    public void dequantize(float[] dst) {
        if (dst.length < size()) {
            throw new IllegalArgumentException("needs " + size() + " floats, got " + dst.length);
        }
        float[] row = new float[in];
        for (int c = 0; c < out; c++) {
            decodeRow(c, row);
            for (int r = 0; r < in; r++) {
                dst[r * out + c] = row[r];
            }
        }
    }

    /**
     * Asks for the whole matrix to be read into memory now, if it comes from a mapped file,
     * so a product that follows does not wait on the disk page by page. Called from several
     * threads at once for several matrices, it keeps a fast SSD busy with many reads.
     */
    public void load() {
        if (data.isMapped() && !loaded()) {
            data.load();
            warmAt = READ.addAndGet(data.byteSize());
        }
    }

    /**
     * {@linkplain #load Loads} the matrices that are not all in memory, in parallel and in
     * pieces of about a megabyte: many reads at once keep a fast SSD busy, where a product
     * alone would wait on one page after another. For the weights a step is about to use,
     * such as the experts a router chose.
     *
     * @param matrices the matrices
     */
    public static void loadAll(Collection<GgmlBlocks> matrices) {
        List<Pass> passes = new ArrayList<>();
        int pieces = 0;
        for (GgmlBlocks b : matrices) {
            Pass p = b.pass();
            if (p != null) {
                passes.add(p);
                pieces += p.pieces;
            }
        }
        if (pieces == 0) {
            return;
        }
        if (pieces == 1) {
            passes.getFirst().readOne();
            return;
        }
        // one task per piece, each reading the next piece no one took yet: pieces a read-ahead
        // already took are not read twice (measured: most of the reading as a step began was
        // of experts a read-ahead was still reading)
        Pass[] each = new Pass[pieces];
        for (int i = 0, at = 0; i < passes.size(); i++) {
            for (int j = 0; j < passes.get(i).pieces; j++) {
                each[at++] = passes.get(i);
            }
        }
        java.util.Arrays.stream(each).parallel().forEach(Pass::readOne);
    }

    /**
     * Starts {@linkplain #loadAll loading} the matrices that are not all in memory on
     * background threads and returns at once: for weights a later step will probably use,
     * such as the experts a router is likely to choose next, so the disk reads them while
     * this step computes. A matrix counts as loaded once all its pieces are; a step that
     * needs it sooner loads the pieces still left itself.
     *
     * @param matrices the matrices
     * @return the bytes asked for
     */
    public static long readAhead(Collection<GgmlBlocks> matrices) {
        long asked = 0;
        for (GgmlBlocks b : matrices) {
            Pass p = b.pass();
            if (p == null || !p.ahead.compareAndSet(false, true)) {
                continue;   // in memory, or on its way
            }
            asked += b.data.byteSize();
            for (int i = 0; i < p.pieces; i++) {
                READERS.execute(p::readOne);
            }
        }
        return asked;
    }

    /**
     * A load of this matrix under way, or a new one; null if it is in memory. Every loader of
     * the matrix takes its pieces from the same pass.
     */
    private Pass pass() {
        if (loaded()) {
            return null;
        }
        Pass p = pass;
        if (p != null) {
            return p;
        }
        Pass fresh = new Pass(this);
        return PASS.compareAndExchange(this, null, fresh) instanceof Pass other ? other : fresh;
    }

    /** One load of a matrix's pieces, shared by whoever reads them. */
    private static final class Pass {
        final GgmlBlocks of;
        final int pieces;
        final java.util.concurrent.atomic.AtomicInteger next = new java.util.concurrent.atomic.AtomicInteger();
        final java.util.concurrent.atomic.AtomicInteger left;
        /** Whether a read-ahead already asked for its pieces. */
        final java.util.concurrent.atomic.AtomicBoolean ahead = new java.util.concurrent.atomic.AtomicBoolean();

        Pass(GgmlBlocks of) {
            this.of = of;
            this.pieces = (int) Math.max(1, (of.data.byteSize() + CHUNK - 1) / CHUNK);
            this.left = new java.util.concurrent.atomic.AtomicInteger(pieces);
        }

        /** Reads the next piece no one took, if any; the last one marks the matrix loaded. */
        void readOne() {
            int i = next.getAndIncrement();
            if (i >= pieces) {
                return;
            }
            long size = of.data.byteSize();
            long at = (long) i * CHUNK;
            try {
                of.data.asSlice(at, Math.min(CHUNK, size - at)).load();
            } finally {
                if (left.decrementAndGet() == 0) {
                    of.warmAt = READ.addAndGet(size);
                    of.pass = null;
                }
            }
        }
    }

    /** The load under way, or null. */
    private volatile Pass pass;
    private static final java.lang.invoke.VarHandle PASS;

    static {
        try {
            PASS = java.lang.invoke.MethodHandles.lookup().findVarHandle(GgmlBlocks.class, "pass", Pass.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * The threads of {@link #readAhead}: they mostly wait on the disk, so there are more of
     * them than a fast SSD needs reads at once, and they leave the cores to the computation.
     */
    private static final java.util.concurrent.ExecutorService READERS = java.util.concurrent.Executors
            .newFixedThreadPool(Integer.getInteger("brainiac.readAheadThreads", 16), r -> {
                Thread t = new Thread(r, "brainiac-read-ahead");
                t.setDaemon(true);
                return t;
            });

    /**
     * Returns whether the matrix is in memory: not mapped, or {@linkplain #load loaded} with
     * fewer than {@code brainiac.mappedResidentBytes} read since. Asking the operating system
     * page by page on every use costs about as much as reading, and on some systems it cannot
     * tell at all.
     *
     * @return whether it need not be loaded
     */
    public boolean loaded() {
        long at = warmAt;
        return !data.isMapped() || at >= 0 && READ.get() - at < RESIDENT;
    }

    @Override
    public String toString() {
        return "GgmlBlocks[" + format + ", " + in + "x" + out + "]";
    }
}
