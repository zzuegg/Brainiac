package io.github.zzuegg.brainiac.studio.items;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Which project each item belongs to, for the kinds whose own files keep no project
 * (conversations and graphs). Kept in one file, apart from the items, so a client that rewrites
 * an item's file whole cannot lose it. An item with no entry belongs to the default project.
 */
public final class Owners {

    private static final System.Logger LOG = System.getLogger(Owners.class.getName());
    private static final String DEFAULT = "default";

    private final Path file;
    private final Map<String, String> owners = new LinkedHashMap<>();

    /**
     * Reads the owners.
     *
     * @param file where they are kept, or null to keep them only while running
     */
    public Owners(Path file) {
        this.file = file;
        if (file != null && Files.isRegularFile(file)) {
            try {
                Json.object(Json.read(file)).forEach((k, v) -> {
                    if (v instanceof String s) {
                        owners.put(k, s);
                    }
                });
            } catch (IOException | RuntimeException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot read " + file, e);
            }
        }
    }

    /**
     * Returns the project an item belongs to.
     *
     * @param kind its kind
     * @param id its id
     * @return the project's id, {@code default} when none was recorded
     */
    public synchronized String of(String kind, String id) {
        return owners.getOrDefault(kind + "/" + id, DEFAULT);
    }

    /**
     * Records the project an item belongs to.
     *
     * @param kind its kind
     * @param id its id
     * @param project the project's id
     * @throws IOException if the change cannot be kept
     */
    public synchronized void set(String kind, String id, String project) throws IOException {
        if (project == null || DEFAULT.equals(project)) {
            owners.remove(kind + "/" + id);
        } else {
            owners.put(kind + "/" + id, project);
        }
        write();
    }

    /**
     * Records the project of an item seen for the first time; one already recorded keeps its project.
     * Errors are logged, not thrown, so saving the item itself never fails because of this.
     *
     * @param kind its kind
     * @param id its id
     * @param project the project the call that made it was in, or null
     */
    public synchronized void claim(String kind, String id, String project) {
        String key = kind + "/" + id;
        if (project == null || DEFAULT.equals(project) || owners.containsKey(key)) {
            return;
        }
        owners.put(key, project);
        try {
            write();
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot keep the project of " + key, e);
        }
    }

    /**
     * Moves an item's entry to a new id, after a rename that changes it.
     *
     * @param kind its kind
     * @param from the old id
     * @param to the new id
     * @throws IOException if the change cannot be kept
     */
    public synchronized void rekey(String kind, String from, String to) throws IOException {
        String p = owners.remove(kind + "/" + from);
        if (p != null) {
            owners.put(kind + "/" + to, p);
            write();
        }
    }

    /**
     * Forgets an item that is gone for good.
     *
     * @param kind its kind
     * @param id its id
     */
    public synchronized void forget(String kind, String id) {
        if (owners.remove(kind + "/" + id) != null) {
            try {
                write();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    private void write() throws IOException {
        if (file == null) {
            return;
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.writeString(tmp, Json.write(owners));
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }
}
