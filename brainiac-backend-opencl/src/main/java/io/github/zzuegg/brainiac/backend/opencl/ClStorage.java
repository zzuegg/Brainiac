package io.github.zzuegg.brainiac.backend.opencl;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.lang.foreign.MemorySegment;

/**
 * Values in OpenCL device memory. The buffer is created when it is first written, so results
 * that are only ever inlined into a fused kernel never take one; it goes back to the pool
 * when this object is garbage collected.
 */
final class ClStorage implements Storage {

    /** The device buffer, or null until it is needed; shared with the cleaner action. */
    static final class Buffer {
        volatile MemorySegment mem;
        /** The size {@link #mem} was created with. */
        long bytes;
    }

    final OpenClBackend owner;
    final Buffer buffer;
    final long size;
    final DType dtype;
    /** The element-wise expression not yet computed into {@link #mem}, or null once it is. */
    Fused.Expr pending;
    /** The size of {@link #pending} in nodes, inlined inputs included. */
    int pendingNodes;
    /** The backend's sync generation when {@link #pending} was recorded. */
    long pendingGeneration;
    /** Whether its buffer went back to the pool through {@link OpenClBackend#release}. */
    boolean released;

    ClStorage(OpenClBackend owner, Buffer buffer, long size, DType dtype) {
        this.owner = owner;
        this.buffer = buffer;
        this.size = size;
        this.dtype = dtype;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    /** The device buffer, created (or taken from the pool) on first use. */
    MemorySegment mem() {
        if (buffer.mem == null) {
            if (released) {
                throw new IllegalStateException("a released tensor was used");
            }
            buffer.bytes = OpenClBackend.bucket(size * dtype.bytes());
            buffer.mem = owner.buffer(buffer.bytes);
        }
        return buffer.mem;
    }

    long bytes() {
        return size * dtype.bytes();
    }

    @Override
    public String toString() {
        return "ClStorage[" + size + " " + dtype + "]";
    }
}
