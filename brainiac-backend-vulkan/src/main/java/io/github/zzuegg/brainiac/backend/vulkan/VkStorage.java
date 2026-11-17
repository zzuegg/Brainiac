package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.lang.ref.Cleaner;

/**
 * Values of one {@link DType} in Vulkan device memory: a slot carved out of a larger memory block,
 * returned to the pool when this object is released or garbage collected.
 */
final class VkStorage implements Storage {

    final VulkanBackend owner;
    final VkContext.Slot slot;
    final long size;
    final DType dtype;
    /** The values as uploaded, kept for small uploads so indices can be checked without a download. */
    final float[] host;
    private final Cleaner.Cleanable cleanable;
    private boolean released;

    VkStorage(VulkanBackend owner, VkContext.Slot slot, long size, DType dtype, float[] host, Cleaner cleaner,
            VkContext context) {
        this.owner = owner;
        this.slot = slot;
        this.size = size;
        this.dtype = dtype;
        this.host = host;
        this.cleanable = cleaner.register(this, new Free(context, slot));
    }

    /** The cleaner action; must not reference the storage itself. */
    private record Free(VkContext context, VkContext.Slot slot) implements Runnable {
        @Override
        public void run() {
            context.free(slot);
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
        return slot.address() + index * dtype.bytes();
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
