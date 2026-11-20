package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * The chat conversations, one JSON file each in a folder next to the keys, so they outlive the
 * browser and show up in any browser on this PC. The page owns their shape; the studio checks
 * only what it needs to keep the files safe and listable.
 */
public final class Conversations {

    private static final System.Logger LOG = System.getLogger(Conversations.class.getName());
    private static final Pattern ID = Pattern.compile("[a-z0-9]{4,40}");
    /** At most this many are kept; the ones saved longest ago go first. */
    static final int KEEP = 1000;

    private final Path folder;
    private final int keep;

    /**
     * Opens the folder; it is made on the first save.
     *
     * @param folder where the conversations go
     */
    public Conversations(Path folder) {
        this(folder, KEEP);
    }

    Conversations(Path folder, int keep) {
        this.folder = folder;
        this.keep = keep;
    }

    /**
     * Reads every conversation, the most recently updated first. A file that cannot be read is
     * skipped and reported, never deleted.
     *
     * @return the conversations
     */
    public synchronized List<Map<String, Object>> list() {
        List<Map<String, Object>> out = new ArrayList<>();
        if (!Files.isDirectory(folder)) {
            return out;
        }
        try (DirectoryStream<Path> files = Files.newDirectoryStream(folder, "*.json")) {
            for (Path f : files) {
                try {
                    out.add(Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8))));
                } catch (IOException | IllegalArgumentException bad) {
                    LOG.log(System.Logger.Level.WARNING, "skipping " + f + ": " + bad.getMessage());
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot list " + folder, e);
        }
        out.sort(Comparator.comparingLong((Map<String, Object> c) -> Json.integer(c, "updated", 0)).reversed());
        return out;
    }

    /**
     * Saves a conversation, replacing the one with its id.
     *
     * @param c the conversation: {@code id}, {@code title}, {@code messages} and anything else the page keeps
     * @throws IOException if it cannot be written
     */
    public synchronized void save(Map<String, Object> c) throws IOException {
        String id = id(c.get("id"));
        if (!(c.get("title") instanceof String title) || title.length() > 500) {
            throw new IllegalArgumentException("a conversation needs a title of at most 500 characters");
        }
        if (!(c.get("messages") instanceof List<?>)) {
            throw new IllegalArgumentException("a conversation needs a list of messages");
        }
        Object status = c.get("status");
        if (status != null && !"open".equals(status) && !"finished".equals(status)) {
            throw new IllegalArgumentException("status is open or finished");
        }
        Files.createDirectories(folder);
        Path f = folder.resolve(id + ".json");
        Path tmp = folder.resolve(id + ".json.tmp");
        Files.writeString(tmp, Json.write(c), StandardCharsets.UTF_8);
        Files.move(tmp, f, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        trim();
    }

    /**
     * Returns the folder the conversations are kept in.
     *
     * @return the folder
     */
    public Path folder() {
        return folder;
    }

    /**
     * Returns a conversation's file.
     *
     * @param id its id
     * @return the file, which may not exist
     * @throws IllegalArgumentException if the id is not one a conversation can have
     */
    public Path file(Object id) {
        return folder.resolve(id(id) + ".json");
    }

    /**
     * Reads one conversation.
     *
     * @param id its id
     * @return it, or empty
     * @throws IOException if it cannot be read
     */
    public synchronized java.util.Optional<Map<String, Object>> get(Object id) throws IOException {
        Path f = file(id);
        if (!Files.isRegularFile(f)) {
            return java.util.Optional.empty();
        }
        return java.util.Optional.of(Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8))));
    }

    /**
     * Deletes a conversation.
     *
     * @param id its id
     * @return true if there was one
     * @throws IOException if it cannot be deleted
     */
    public synchronized boolean delete(Object id) throws IOException {
        return Files.deleteIfExists(folder.resolve(id(id) + ".json"));
    }

    private static String id(Object id) {
        if (!(id instanceof String s) || !ID.matcher(s).matches()) {
            throw new IllegalArgumentException("a conversation id is 4 to 40 lowercase letters and digits");
        }
        return s;
    }

    private void trim() throws IOException {
        List<Path> files = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(folder, "*.json")) {
            ds.forEach(files::add);
        }
        if (files.size() <= keep) {
            return;
        }
        files.sort(Comparator.comparing(f -> {
            try {
                return Files.getLastModifiedTime(f);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }));
        for (Path f : files.subList(0, files.size() - keep)) {
            Files.deleteIfExists(f);
        }
    }
}
