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
                }
            }
            devices.sort(Comparator.comparing(Device::kind));
            return List.copyOf(devices);
        }
    }
}
