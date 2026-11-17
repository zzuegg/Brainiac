package io.github.zzuegg.brainiac.backend.vulkan;

import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Per-device choices of kernel shapes and thresholds: measured once per device and driver by
 * {@link Tuner} and kept in a file under {@code ~/.brainiac/vulkan-tuning} (or
 * {@code -Dbrainiac.vulkan.tuningDir}). A system property {@code brainiac.vulkan.<key>} fixes
 * a value: it wins over the file and is not tuned.
 */
final class Tuning {

    /** A tile shape of the cooperative-matrix product: the shader's suffix and its output tile. */
    record CoopShape(String suffix, int rows, int cols) {
    }

    /** The compiled shapes (see matmul_coop.comp and the build's variants). */
    static final List<CoopShape> COOP_SHAPES = List.of(new CoopShape("", 128, 128), new CoopShape("k64", 128, 128),
            new CoopShape("m64", 64, 128), new CoopShape("n256", 128, 256), new CoopShape("m256", 256, 128),
            new CoopShape("t44", 128, 128), new CoopShape("v", 128, 128), new CoopShape("t44v", 128, 128),
            new CoopShape("k64v", 128, 128));

    /** The values used without a tuning: those measured on an RX 9070 XT. */
    static final Map<String, Long> DEFAULTS = Map.of(
            "coopShape", 0L,          // index into COOP_SHAPES; -1: no cooperative matrices
            "coopRows", 64L,          // fewest rows of a float16 product on cooperative matrices
            "bigTileGroups", 128L,    // 64 x 64 tiles from this many of them, else 32 x 32
            "splitBelow", 256L,       // products of fewer tiles split k into pieces
            "packedGroups", 256L,     // work groups a few-row packed product aims for
            "packedPiece", 64L);      // the fewest rows of k per piece of it

    private final Map<String, Long> values = new LinkedHashMap<>();

    long get(String key) {
        String fixed = System.getProperty("brainiac.vulkan." + key);
        if (fixed != null) {
            return Long.parseLong(fixed.trim());
        }
        Long v = values.get(key);
        return v != null ? v : DEFAULTS.get(key);
    }

    /** Whether a system property fixes the value. */
    static boolean fixed(String key) {
        return System.getProperty("brainiac.vulkan." + key) != null;
    }

    void set(String key, long value) {
        values.put(key, value);
    }

    Map<String, Long> values() {
        return values;
    }

    static Path directory() {
        String dir = System.getProperty("brainiac.vulkan.tuningDir");
        return dir != null ? Path.of(dir) : Path.of(System.getProperty("user.home"), ".brainiac", "vulkan-tuning");
    }

    static Path file(String deviceKey) {
        return directory().resolve(deviceKey + ".properties");
    }

    /** The tuning kept for a device, or null if there is none or it cannot be read. */
    static Tuning load(Path file) {
        if (!Files.isRegularFile(file)) {
            return null;
        }
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file)) {
            p.load(r);
        } catch (IOException | IllegalArgumentException e) {
            return null;
        }
        Tuning t = new Tuning();
        for (String key : DEFAULTS.keySet()) {
            String v = p.getProperty(key);
            if (v != null) {
                try {
                    t.values.put(key, Long.parseLong(v.trim()));
                } catch (NumberFormatException e) {
                    // left at the default
                }
            }
        }
        long shape = t.get("coopShape");
        if (shape >= COOP_SHAPES.size() || shape < -1) {
            t.values.remove("coopShape");
        }
        return t;
    }

    void save(Path file, String comment) {
        Properties p = new Properties();
        values.forEach((k, v) -> p.setProperty(k, Long.toString(v)));
        try {
            Files.createDirectories(file.getParent());
            Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
            try (Writer w = Files.newBufferedWriter(tmp)) {
                p.store(w, comment);
            }
            Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public String toString() {
        StringBuilder b = new StringBuilder();
        for (String key : DEFAULTS.keySet()) {
            long v = get(key);
            b.append(b.isEmpty() ? "" : ", ").append(key).append(' ')
                    .append(key.equals("coopShape") ? (v < 0 ? "off" : COOP_SHAPES.get((int) v).suffix().isEmpty()
                            ? "128x128" : COOP_SHAPES.get((int) v).suffix()) : Long.toString(v));
        }
        return b.toString();
    }
}
