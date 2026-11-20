package io.github.zzuegg.brainiac.studio.items;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.LongSupplier;

/**
 * Deleted items, kept for {@linkplain #KEEP 30 days} so a delete can be undone. Each entry is a
 * folder in {@code trash} holding {@code entry.json} (what was deleted and what its kind needs
 * to put it back) and the item's files under {@code files}.
 */
public final class Trash {

    /** How long a deleted item is kept. */
    public static final Duration KEEP = Duration.ofDays(30);

    private static final System.Logger LOG = System.getLogger(Trash.class.getName());

    private final Path folder;
    private final LongSupplier clock;
    private long counter;

    /**
     * One deleted item.
     *
     * @param id the entry's id in the trash
     * @param item the item as it was listed when deleted
     * @param deleted when, in milliseconds since the epoch
     * @param kept what its kind needs to put it back
     */
    public record Entry(String id, Item item, long deleted, Map<String, Object> kept) {

        /**
         * Describes the entry for a page.
         *
         * @return {@code {id, deleted, expires, item}}
         */
        public Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("deleted", deleted);
            m.put("expires", deleted + KEEP.toMillis());
            m.put("item", item.toJson());
            return m;
        }
    }

    /**
     * Opens the trash.
     *
     * @param folder where deleted items are kept
     */
    public Trash(Path folder) {
        this(folder, System::currentTimeMillis);
    }

    Trash(Path folder, LongSupplier clock) {
        this.folder = folder;
        this.clock = clock;
    }

    /**
     * Moves an item to the trash.
     *
     * @param item the item as listed
     * @param kind its kind
     * @return the entry
     * @throws IOException if it cannot be moved
     */
    public synchronized Entry put(Item item, ItemKind kind) throws IOException {
        long now = clock.getAsLong();
        String id = now + "-" + (++counter);
        Path dir = folder.resolve(id);
        Files.createDirectories(dir.resolve("files"));
        Map<String, Object> kept;
        try {
            kept = kind.detach(item.id(), dir.resolve("files"));
        } catch (IOException | RuntimeException e) {
            deleteTree(dir);
            throw e;
        }
        Entry e = new Entry(id, item, now, kept == null ? Map.of() : kept);
        Map<String, Object> m = new LinkedHashMap<>(e.toJson());
        m.put("kept", e.kept());
        Files.writeString(dir.resolve("entry.json"), Json.write(m), StandardCharsets.UTF_8);
        return e;
    }

    /**
     * Lists what is in the trash, the most recently deleted first.
     *
     * @return the entries
     */
    public synchronized List<Entry> list() {
        List<Entry> out = new ArrayList<>();
        if (!Files.isDirectory(folder)) {
            return out;
        }
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(folder, Files::isDirectory)) {
            for (Path dir : ds) {
                read(dir).ifPresent(out::add);
            }
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot list " + folder, e);
        }
        out.sort(Comparator.comparingLong(Entry::deleted).reversed());
        return out;
    }

    /**
     * Finds an entry.
     *
     * @param id the entry's id
     * @return it, or empty
     */
    public synchronized Optional<Entry> get(String id) {
        if (id == null || !id.matches("\\d+-\\d+")) {
            return Optional.empty();
        }
        return read(folder.resolve(id));
    }

    /**
     * Puts an item back where it was.
     *
     * @param id the entry's id
     * @param kind the item's kind
     * @return the entry restored
     * @throws IllegalArgumentException if there is no such entry
     * @throws IOException if it cannot be put back
     */
    public synchronized Entry restore(String id, ItemKind kind) throws IOException {
        Entry e = get(id).orElseThrow(() -> new IllegalArgumentException("nothing in the trash is called " + id));
        Path dir = folder.resolve(id);
        kind.attach(e.item().id(), dir.resolve("files"), e.kept());
        deleteTree(dir);
        return e;
    }

    /**
     * Deletes an entry for good.
     *
     * @param id the entry's id
     * @param kind the item's kind, which forgets what it kept elsewhere, or null
     * @return whether there was one
     * @throws IOException if it cannot be deleted
     */
    public synchronized boolean purge(String id, ItemKind kind) throws IOException {
        Optional<Entry> e = get(id);
        if (e.isEmpty()) {
            return false;
        }
        if (kind != null) {
            kind.purge(e.get().item().id(), e.get().kept());
        }
        deleteTree(folder.resolve(id));
        return true;
    }

    /**
     * Returns the entries older than {@link #KEEP}, which {@link Items} purges.
     *
     * @return them
     */
    public synchronized List<Entry> expired() {
        long cutoff = clock.getAsLong() - KEEP.toMillis();
        return list().stream().filter(e -> e.deleted() < cutoff).toList();
    }

    private Optional<Entry> read(Path dir) {
        Path f = dir.resolve("entry.json");
        if (!Files.isRegularFile(f)) {
            return Optional.empty();
        }
        try {
            Map<String, Object> m = Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8)));
            Map<String, Object> i = Json.object(m.get("item"));
            Map<String, Object> extra = new LinkedHashMap<>(i);
            List.of("kind", "id", "title", "project", "updated").forEach(extra::remove);
            Item item = new Item(Json.string(i, "kind", ""), Json.string(i, "id", ""), Json.string(i, "title", ""),
                    Json.string(i, "project", null), Json.integer(i, "updated", 0), extra);
            return Optional.of(new Entry(Json.string(m, "id", dir.getFileName().toString()), item,
                    Json.integer(m, "deleted", 0), Json.object(m.getOrDefault("kept", Map.of()))));
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "skipping " + f + ": " + e.getMessage());
            return Optional.empty();
        }
    }

    static void deleteTree(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        Files.walkFileTree(dir, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path d, IOException exc) throws IOException {
                if (exc != null) {
                    throw exc;
                }
                Files.delete(d);
                return FileVisitResult.CONTINUE;
            }
        });
    }
}
