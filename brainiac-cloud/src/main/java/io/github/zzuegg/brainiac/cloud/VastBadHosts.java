package io.github.zzuegg.brainiac.cloud;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.util.HashSet;
import java.util.Set;

/**
 * Host machines that failed {@code setup.sh check} (no usable NVIDIA Vulkan driver), kept in a
 * file so that later searches skip their offers. One line per machine: its id, then a note.
 */
public final class VastBadHosts {

    /** The default file, {@code ~/.brainiac/vast/bad-machines.txt}. */
    public static final Path DEFAULT_FILE = Path.of(System.getProperty("user.home"), ".brainiac", "vast", "bad-machines.txt");

    private final Path file;
    private final Set<Long> machines = new HashSet<>();

    /**
     * Reads the list.
     *
     * @param file the file; it need not exist yet
     */
    public VastBadHosts(Path file) {
        this.file = file;
        try {
            if (Files.exists(file)) {
                for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
                    String id = line.strip().split("\\s+", 2)[0];
                    if (id.matches("\\d+")) {
                        machines.add(Long.parseLong(id));
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("could not read " + file + ": " + e.getMessage());
        }
    }

    /**
     * Reads the default list.
     *
     * @return the list in {@link #DEFAULT_FILE}
     */
    public static VastBadHosts load() {
        return new VastBadHosts(DEFAULT_FILE);
    }

    /**
     * Tells whether an offer is on a machine that failed before.
     *
     * @param offer the offer
     * @return true to skip it
     */
    public boolean contains(VastOffer offer) {
        return offer.machineId() != 0 && machines.contains(offer.machineId());
    }

    /**
     * Records a machine that failed.
     *
     * @param offer the offer that was rented on it
     * @param reason a short note, such as {@code no NVIDIA Vulkan driver}
     */
    public void add(VastOffer offer, String reason) {
        if (offer.machineId() == 0 || !machines.add(offer.machineId())) {
            return;
        }
        try {
            Files.createDirectories(file.getParent());
            Files.writeString(file, offer.machineId() + "  " + LocalDate.now() + " " + offer.gpuName() + ", driver "
                    + offer.driverVersion() + ", " + offer.location() + ": " + reason.replaceAll("\\s+", " ") + "\n",
                    StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.err.println("could not write " + file + ": " + e.getMessage());
        }
    }
}
