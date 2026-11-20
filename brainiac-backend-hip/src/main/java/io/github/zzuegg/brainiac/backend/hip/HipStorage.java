package io.github.zzuegg.brainiac.backend.hip;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.lang.ref.Cleaner;

/**
 * Values in device memory. The memory is taken when the values are first written, so a result
 * that is only ever inlined into generated kernels ({@link HipFusion}) never takes any, and goes
 * back to the context's pool when the storage is released, or when the garbage collector finds
 * it unreachable.
 */
final class HipStorage implements Storage {

    /** The memory, or 0 until it is needed; shared with the cleaner action. */
    private static final class Holder {
        volatile long pointer;
    }

    final HipBackend owner;
    final long size;
    final DType dtype;
    /** The values as uploaded, kept for small uploads so indices can be checked without a download. */
    final float[] host;
    /**
     * Above every value when they are indices made on the device, such as an argmax over rows
     * of this length, or 0: a gather over at least this many entries need not download them to check.
     */
    long indexBound;
    /** The element-wise expression not yet computed into memory, or null once it is. */
    HipFusion.Expr pending;
    /** The size of {@link #pending} in nodes, inlined inputs included. */
    int pendingNodes;
    /** The backend's fusion generation when {@link #pending} was recorded. */
    long pendingGeneration;
    /**
     * Storages whose pending expressions read this one directly, or did when they were recorded
     * (null if none ever did): while any still does, this one is not freed early, and a write
     * to it computes them first.
     */
    java.util.ArrayList<java.lang.ref.WeakReference<HipStorage>> readers;
    private final HipContext context;
    private final Holder holder = new Holder();
    private final Cleaner.Cleanable cleanable;
    private boolean released;

    HipStorage(HipBackend owner, HipContext context, long size, DType dtype, float[] host, Cleaner cleaner) {
        this.owner = owner;
        this.size = size;
        this.dtype = dtype;
        this.host = host;
        this.context = context;
        this.cleanable = cleaner.register(this, new Free(context, holder, Math.max(1, size) * dtype.bytes()));
    }

    /** Runs on the collector's thread (or on release): hands the memory back for the next allocation. */
    private record Free(HipContext context, Holder holder, long bytes) implements Runnable {
        @Override
        public void run() {
            long p = holder.pointer;
            if (p != 0) {
                context.freeLater(p, bytes);
            }
        }
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    /** The device address of element {@code index}; the memory is taken on first use. */
    long address(long index) {
        if (released) {
            throw new IllegalStateException("a released tensor was used");
        }
        if (pending != null) {
            throw new IllegalStateException("a pending result was read before it was computed");
        }
        long p = holder.pointer;
        if (p == 0) {
            p = context.allocate(Math.max(1, size) * dtype.bytes());
            holder.pointer = p;
        }
        return p + index * dtype.bytes();
    }

    long address() {
        return address(0);
    }

    long bytes() {
        return size * dtype.bytes();
    }

    void release() {
        if (!released) {
            released = true;
            cleanable.clean();
        }
    }

    @Override
    public String toString() {
        return "HipStorage[" + size + " " + dtype + "]";
    }
}
