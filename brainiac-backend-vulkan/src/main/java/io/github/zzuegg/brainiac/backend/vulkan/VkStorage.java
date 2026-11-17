package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.lang.ref.Cleaner;

/**
 * Values of one {@link DType} in Vulkan device memory: a slot carved out of a larger memory block,
 * taken when the values are first written and returned to the pool when this object is released
 * or garbage collected. A result that is only ever inlined into fused shaders never takes one.
 */
final class VkStorage implements Storage {

    /** The slot, or null until it is needed; shared with the cleaner action. */
    private static final class Holder {
        volatile VkContext.Slot slot;
    }

    final VulkanBackend owner;
    final long size;
    final DType dtype;
    /** The values as uploaded, kept for small uploads so indices can be checked without a download. */
    final float[] host;
    /** The element-wise expression not yet computed into the slot, or null once it is. */
    Fusion.Expr pending;
    /** The size of {@link #pending} in nodes, inlined inputs included. */
    int pendingNodes;
    /** The backend's fusion generation when {@link #pending} was recorded. */
    long pendingGeneration;
    private final VkContext context;
    private final Holder holder = new Holder();
    private final Cleaner.Cleanable cleanable;
    private boolean released;

    VkStorage(VulkanBackend owner, long size, DType dtype, float[] host, Cleaner cleaner, VkContext context) {
        this.owner = owner;
        this.size = size;
        this.dtype = dtype;
        this.host = host;
        this.context = context;
        this.cleanable = cleaner.register(this, new Free(context, holder));
    }

    /** The cleaner action; must not reference the storage itself. */
    private record Free(VkContext context, Holder holder) implements Runnable {
        @Override
        public void run() {
            VkContext.Slot s = holder.slot;
            if (s != null) {
                context.free(s);
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

    /** The device memory, taken from the pool on first use. */
    VkContext.Slot slot() {
        if (released) {
            throw new IllegalStateException("a released tensor was used");
        }
        VkContext.Slot s = holder.slot;
        if (s == null) {
            s = context.allocate(bytes());
            holder.slot = s;
        }
        return s;
    }

    /** The device address of element {@code index}. */
    long address(long index) {
        return slot().address() + index * dtype.bytes();
    }

    /** The device address of the first element. */
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
        return "VkStorage[" + size + " " + dtype + "]";
    }
}
