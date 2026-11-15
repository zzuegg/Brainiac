package io.github.zzuegg.brainiac.backend.opencl;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.lang.foreign.MemorySegment;

/** A buffer in OpenCL device memory, released when this object is garbage collected. */
final class ClStorage implements Storage {

    final OpenClBackend owner;
    final MemorySegment mem;
    final long size;
    final DType dtype;

    ClStorage(OpenClBackend owner, MemorySegment mem, long size, DType dtype) {
        this.owner = owner;
        this.mem = mem;
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

    long bytes() {
        return size * dtype.bytes();
    }

    @Override
    public String toString() {
        return "ClStorage[" + size + " " + dtype + "]";
    }
}
