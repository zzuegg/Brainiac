package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Where this PC's models run: the best device found (a GPU if there is one), or one the user
 * chose by name, such as the processor while the GPU trains. The choice is kept in a file.
 *
 * <p>It also keeps a limit on the GPU memory the studio takes, in MB: the core's
 * {@code brainiac.vulkan.memory}, which caps device memory and lets the rest spill to system
 * memory. The core reads it once, when it first opens the GPU, so a kept limit is applied when
 * the studio starts.
 */
public final class LocalDevice {

    /** The choice that takes a GPU if there is one, otherwise the processor. */
    public static final String BEST = "best";

    private static final System.Logger LOG = System.getLogger(LocalDevice.class.getName());

    /** The core's property for the GPU memory limit, in MB. */
    public static final String MEMORY_PROPERTY = "brainiac.vulkan.memory";
    /** The file a GPU memory limit is kept in, beside the studio's other settings. */
    public static final String MEMORY_FILE = "local-gpu-memory.txt";

    /** The file the GPU scheduling choices are kept in, beside the studio's other settings. */
    public static final String SCHEDULING_FILE = "gpu-scheduling.json";

    private final Path file;
    private final GpuResidency residency;
    private volatile String choice = BEST;
    private volatile int memory;

    /**
     * Applies the kept GPU memory limit, before anything opens the GPU. A limit given on the
     * command line ({@code -Dbrainiac.vulkan.memory=MB}) stays as it is.
     *
     * @param file where the limit is kept
     */
    public static void applyMemoryLimit(Path file) {
        int kept = readMemory(file);
        if (kept > 0 && System.getProperty(MEMORY_PROPERTY) == null) {
            System.setProperty(MEMORY_PROPERTY, String.valueOf(kept));
        }
    }

    private static int readMemory(Path file) {
        if (file != null && Files.isRegularFile(file)) {
            try {
                return Math.max(0, Integer.parseInt(Files.readString(file).strip()));
            } catch (IOException | NumberFormatException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot read " + file, e);
            }
        }
        return 0;
    }

    private Path memoryFile() {
        return file == null ? null : file.resolveSibling(MEMORY_FILE);
    }

    /**
     * Returns the GPU memory limit the user set.
     *
     * @return it in MB, or 0 for none
     */
    public int memory() {
        return memory;
    }

    /**
     * Returns the GPU memory limit in effect since the studio started.
     *
     * @return it in MB, or 0 for none
     */
    public static int memoryInEffect() {
        Integer mb = Integer.getInteger(MEMORY_PROPERTY);
        return mb == null ? 0 : Math.max(0, mb);
    }

    /**
     * Sets the GPU memory limit. It is kept, and applies from the next start of the studio.
     *
     * @param mb the limit in MB, or 0 for none
     * @throws IllegalArgumentException if it is negative, or too small to hold a model
     * @throws IOException if it cannot be kept
     */
    public void limitMemory(int mb) throws IOException {
        if (mb < 0 || (mb > 0 && mb < 256)) {
            throw new IllegalArgumentException("a GPU memory limit is at least 256 MB, or 0 for none");
        }
        memory = mb;
        Path f = memoryFile();
        if (f != null) {
            if (mb == 0) {
                Files.deleteIfExists(f);
            } else {
                Files.createDirectories(f.toAbsolutePath().getParent());
                Files.writeString(f, mb + "\n");
            }
        }
    }

    /**
     * Reads the choice.
     *
     * @param file where it is kept, or null to keep it only while running
     */
    LocalDevice(Path file) {
        this.file = file;
        this.residency = new GpuResidency(file == null ? null : file.resolveSibling(SCHEDULING_FILE));
        if (file != null && Files.isRegularFile(file)) {
            try {
                String kept = Files.readString(file).strip();
                if (!kept.isEmpty()) {
                    choice = kept;
                }
            } catch (IOException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot read " + file, e);
            }
        }
        memory = readMemory(memoryFile());
    }

    /**
     * Returns what decides which models stay in GPU memory, shared by the chat, image and speech models.
     *
     * @return the scheduler
     */
    public GpuResidency residency() {
        return residency;
    }

    /**
     * Returns the choice.
     *
     * @return {@value #BEST} or a device's name
     */
    public String choice() {
        return choice;
    }

    /**
     * Returns the device to load a model on: the one chosen, or the best one when none was chosen
     * or the chosen one is no longer there.
     *
     * @return the device
     */
    public Device device() {
        String c = choice;
        if (!c.equals(BEST)) {
            for (Device d : devices()) {
                if (d.name().equals(c)) {
                    return d;
                }
            }
        }
        return Device.best();
    }

    /**
     * Chooses where models run from now on; a model already loaded elsewhere is loaded again the
     * next time it is used.
     *
     * @param name {@value #BEST} or the name of one of this PC's devices
     * @throws IllegalArgumentException if there is no such device
     * @throws IOException if the choice cannot be kept
     */
    public void choose(String name) throws IOException {
        if (!BEST.equals(name) && devices().stream().noneMatch(d -> d.name().equals(name))) {
            throw new IllegalArgumentException("no device named " + name + " on this PC");
        }
        choice = name;
        if (file != null) {
            Files.createDirectories(file.toAbsolutePath().getParent());
            Files.writeString(file, name + "\n");
        }
    }

    private static List<Device> devices() {
        try {
            return Device.available();
        } catch (RuntimeException | LinkageError e) {
            return List.of();
        }
    }
}
