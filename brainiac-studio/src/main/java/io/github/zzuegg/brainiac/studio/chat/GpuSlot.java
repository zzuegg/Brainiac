package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.BooleanSupplier;

/**
 * The one model of a kind that this PC keeps loaded, as {@link GpuResidency} sees it. Its owner
 * holds {@link #lock()} while it loads or uses the model, so a model that is working is never
 * unloaded under it; other kinds unload it only when it is free ({@link #tryUnload()}).
 */
final class GpuSlot {

    /** Loads a model. */
    @FunctionalInterface
    interface Loader<T> {
        T load() throws IOException;
    }

    private final String kind;
    private final GpuResidency residency;
    private final java.util.concurrent.locks.ReentrantLock lock = new java.util.concurrent.locks.ReentrantLock();
    private volatile String model;
    private volatile String key;
    private volatile long bytes;
    private volatile long lastUsed;
    private volatile Runnable unloader;

    GpuSlot(String kind, GpuResidency residency) {
        this.kind = kind;
        this.residency = residency;
        residency.register(this);
    }

    String kind() {
        return kind;
    }

    /** Waits for the slot: hold it while loading or using its model. */
    void lock() {
        lock.lock();
    }

    void unlock() {
        lastUsed = System.currentTimeMillis();
        lock.unlock();
    }

    /** @return whether someone holds the slot: loading, using or waiting for room */
    boolean busy() {
        return lock.isLocked();
    }

    /** @return the loaded model's name, or null */
    String loadedModel() {
        return model;
    }

    long bytes() {
        return bytes;
    }

    long lastUsed() {
        return lastUsed;
    }

    /**
     * Says whether the model loaded is this one on this device.
     *
     * @param path the model's file or folder
     * @param device the device
     * @return whether it is loaded
     */
    boolean holds(Path path, Device device) {
        return key(path, device).equals(key);
    }

    private static String key(Path path, Device device) {
        return path.toAbsolutePath().normalize() + "@" + device.name();
    }

    /**
     * Loads a model while holding the slot, after the owner let go of the old one: room is made
     * first as the policy says, and what the model takes is measured.
     *
     * @param name the model's name, for messages
     * @param path its file or folder, to guess its size
     * @param device where it loads
     * @param cancelled says when to stop waiting for room
     * @param loader loads it
     * @param unload lets it go, when another kind needs the room or it was idle too long
     * @return what was loaded
     * @throws GpuResidency.NoRoom if it does not fit and the policy says not to make room here
     * @throws IOException if it cannot be loaded
     */
    <T> T load(String name, Path path, Device device, BooleanSupplier cancelled, Loader<T> loader, Runnable unload)
            throws IOException {
        if (!lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("hold the slot to load into it");
        }
        forget();
        String k = key(path, device);
        residency.admit(this, name, device, residency.estimate(k, path), cancelled);
        long before = residency.inUse(device);
        T t = loader.load();
        long after = residency.inUse(device);
        long used = after > before ? after - before : GpuResidency.sizeOnDisk(path);
        model = name;
        key = k;
        bytes = used;
        unloader = unload;
        residency.measuredAs(k, after > before ? used : 0);
        residency.keepMeasured();
        return t;
    }

    /** Lets go of the model, holding the slot. */
    void forget() {
        Runnable u = unloader;
        unloader = null;
        model = null;
        key = null;
        bytes = 0;
        if (u != null) {
            u.run();
        }
    }

    /**
     * Unloads the model if nobody holds the slot.
     *
     * @return whether it was unloaded (or nothing was loaded)
     */
    boolean tryUnload() {
        if (!lock.tryLock()) {
            return false;
        }
        try {
            forget();
            return true;
        } finally {
            lock.unlock();
        }
    }
}
