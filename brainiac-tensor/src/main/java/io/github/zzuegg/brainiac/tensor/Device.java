package io.github.zzuegg.brainiac.tensor;

import io.github.zzuegg.brainiac.tensor.spi.Backend;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;

/**
 * Where tensors live and their operations run, such as the CPU or a GPU.
 *
 * <p>Devices come from backend modules on the module path (or class path), for example
 * {@code brainiac-backend-cpu}. An operation never mixes devices; move data explicitly
 * with {@link Tensor#to(Device)}.
 *
 * <pre>{@code
 * Device.available();               // [cpu]
 * Device gpu = Device.gpu();        // fails clearly when no GPU backend is installed
 * Device.setDefault(Device.best()); // GPU if there is one, else CPU
 * }</pre>
 *
 * <p>With several GPU backends (Vulkan and CUDA on an NVIDIA card), {@link #gpu()} and
 * {@link #best()} take Vulkan unless {@code -Dbrainiac.gpu=cuda} asks for another.
 */
public final class Device {

    /** The kind of hardware a device runs on. */
    public enum Kind {
        /** the processor running the JVM */
        CPU,
        /** a graphics card or other accelerator */
        GPU
    }

    private static volatile Device defaultDevice;

    private final Backend backend;

    private Device(Backend backend) {
        this.backend = backend;
    }

    /**
     * Returns every usable device, CPUs first.
     *
     * @return the available devices
     */
    public static List<Device> available() {
        return Registry.DEVICES;
    }

    /**
     * Returns the CPU device.
     *
     * @return the CPU device
     * @throws IllegalStateException if no CPU backend is installed
     */
    public static Device cpu() {
        return first(Kind.CPU).orElseThrow(() -> new IllegalStateException(
                "No CPU backend found. Add the brainiac-backend-cpu module (or depend on brainiac)."));
    }

    /**
     * Returns the first GPU device.
     *
     * @return a GPU device
     * @throws IllegalStateException if no usable GPU backend is installed
     */
    public static Device gpu() {
        return first(Kind.GPU).orElseThrow(() -> new IllegalStateException(
                "No GPU backend available. Available devices: " + available()));
    }

    /**
     * Returns a GPU if one is available, otherwise the CPU.
     *
     * @return the fastest available device
     */
    public static Device best() {
        return first(Kind.GPU).orElseGet(Device::cpu);
    }

    /**
     * Returns the device that new tensors use when none is given. Initially the CPU.
     *
     * @return the default device
     */
    public static Device getDefault() {
        Device d = defaultDevice;
        if (d == null) {
            d = cpu();
            defaultDevice = d;
        }
        return d;
    }

    /**
     * Changes the device that new tensors use when none is given.
     *
     * @param device the new default
     */
    public static void setDefault(Device device) {
        defaultDevice = Objects.requireNonNull(device, "device");
    }

    private static Optional<Device> first(Kind kind) {
        return Registry.DEVICES.stream().filter(d -> d.kind() == kind).findFirst();
    }

    /**
     * Returns the device name, such as {@code "cpu"}.
     *
     * @return the name
     */
    public String name() {
        return backend.name();
    }

    /**
     * Returns the kind of hardware.
     *
     * @return the kind
     */
    public Kind kind() {
        return backend.kind();
    }

    /**
     * Returns whether tensors of a type can live on this device.
     *
     * @param dtype the type
     * @return {@code true} if they can
     */
    public boolean supports(DType dtype) {
        return backend().supports(dtype);
    }

    /**
     * How much device memory tensors use, for finding what a model or a training step needs.
     *
     * @param inUse bytes held by live tensors now
     * @param peak the most bytes held at once since the start or {@link #resetPeakMemory()}
     * @param reserved bytes the backend has taken from the device, in use or pooled for reuse
     * @param budget the most bytes the backend means to take
     */
    public record Memory(long inUse, long peak, long reserved, long budget) {
    }

    /**
     * Device memory use. Freed tensors count until they are released (see
     * {@link Tensor#release()}) or garbage collected.
     *
     * @return the use, or empty for devices that do not count it, such as the CPU
     */
    public Optional<Memory> memory() {
        return Optional.ofNullable(backend.memory());
    }

    /**
     * Returns whether {@code bytes} take more than {@code share} of the memory this device's
     * backend means to take ({@link Memory#budget()}), for deciding where a large model runs.
     *
     * @param bytes the bytes to hold
     * @param share the part of the budget they may take, such as 0.8 to leave room for
     *     activations and a cache
     * @return {@code true} if they do; {@code false} also for a device that does not count
     *     its memory, such as the CPU
     */
    public boolean tooSmallFor(long bytes, double share) {
        Memory m = backend.memory();
        return m != null && m.budget() > 0 && bytes > share * m.budget();
    }

    /**
     * Sends the work recorded so far to the device without waiting for it. GPUs that drive a
     * display are reset by their driver when one submission runs for about two seconds;
     * computations with long parts (a large diffusion model's blocks) call this between them.
     */
    public void flush() {
        backend.flush();
    }

    /** Starts measuring the peak in {@link #memory()} again from the current use. */
    public void resetPeakMemory() {
        backend.resetPeakMemory();
    }

    Backend backend() {
        return backend;
    }

    @Override
    public String toString() {
        return name();
    }

    private static final class Registry {
        static final List<Device> DEVICES = load();

        private static List<Device> load() {
            List<Device> devices = new ArrayList<>();
            for (Backend b : ServiceLoader.load(Backend.class)) {
                if (b.isAvailable()) {
                    devices.add(new Device(b));
                    for (Backend sibling : b.siblings()) {
                        if (sibling.isAvailable()) {
                            devices.add(new Device(sibling));
                        }
                    }
                }
            }
            // CPUs first; among GPUs the backend named by -Dbrainiac.gpu (such as "cuda"), then
            // vulkan, then the rest in the order the service loader found them
            String preferred = System.getProperty("brainiac.gpu", "vulkan").toLowerCase(java.util.Locale.ROOT);
            devices.sort(Comparator.comparing(Device::kind).thenComparingInt(d -> d.kind() == Kind.GPU
                    ? rank(d.name().toLowerCase(java.util.Locale.ROOT), preferred) : 0));
            return List.copyOf(devices);
        }

        private static int rank(String name, String preferred) {
            return name.startsWith(preferred) ? 0 : name.startsWith("vulkan") ? 1 : 2;
        }
    }
}
