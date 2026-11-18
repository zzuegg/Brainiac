package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Where this PC's models run: the best device found (a GPU if there is one), or one the user
 * chose by name, such as the processor while the GPU trains. The choice is kept in a file.
 */
public final class LocalDevice {

    /** The choice that takes a GPU if there is one, otherwise the processor. */
    public static final String BEST = "best";

    private static final System.Logger LOG = System.getLogger(LocalDevice.class.getName());

    private final Path file;
    private volatile String choice = BEST;

    /**
     * Reads the choice.
     *
     * @param file where it is kept, or null to keep it only while running
     */
    LocalDevice(Path file) {
        this.file = file;
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
