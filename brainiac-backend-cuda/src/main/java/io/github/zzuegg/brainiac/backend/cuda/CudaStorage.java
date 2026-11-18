package io.github.zzuegg.brainiac.backend.cuda;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.lang.ref.Cleaner;

/**
 * Values in device memory. The memory goes back to the context's pool when the storage is
 * released, or when the garbage collector finds it unreachable.
 */
final class CudaStorage implements Storage {

    final CudaBackend owner;
    final long size;
    final DType dtype;
    /** The values as uploaded, kept for small uploads so indices can be checked without a download. */
    final float[] host;
    private final long pointer;
    private final Cleaner.Cleanable cleanable;
    private boolean released;

    CudaStorage(CudaBackend owner, CudaContext context, long size, DType dtype, float[] host, Cleaner cleaner) {
        this.owner = owner;
        this.size = size;
        this.dtype = dtype;
        this.host = host;
        long bytes = Math.max(1, size) * dtype.bytes();
        this.pointer = context.allocate(bytes);
        this.cleanable = cleaner.register(this, new Free(context, pointer, bytes));
    }

    /** Runs on the collector's thread (or on release): hands the memory back for the next allocation. */
    private record Free(CudaContext context, long pointer, long bytes) implements Runnable {
        @Override
        public void run() {
            context.freeLater(pointer, bytes);
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

    /** The device address of element {@code index}. */
    long address(long index) {
        if (released) {
            throw new IllegalStateException("a released tensor was used");
        }
        return pointer + index * dtype.bytes();
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
        return "CudaStorage[" + size + " " + dtype + "]";
    }
}
