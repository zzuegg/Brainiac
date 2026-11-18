package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Keeps what a file was before an agent changed it, so a run can be undone as a whole. Each run
 * has a folder with a copy per change and a list of the changes in order.
 */
public final class Checkpoints {

    private final Path folder;

    /**
     * Keeps checkpoints in a folder.
     *
     * @param folder one subfolder per run goes here
     */
    public Checkpoints(Path folder) {
        this.folder = folder;
    }

    /**
     * Remembers a file as it is, before a run changes it.
     *
     * @param run the run
     * @param file the file, which may not exist yet
     * @throws IOException if the copy cannot be made
     */
    public synchronized void before(String run, Path file) throws IOException {
        Path dir = folder.resolve(run);
        Files.createDirectories(dir);
        Path list = dir.resolve("changes.jsonl");
        int seq = Files.exists(list) ? Files.readAllLines(list, StandardCharsets.UTF_8).size() : 0;
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("seq", seq);
        m.put("path", file.toAbsolutePath().normalize().toString());
        m.put("existed", Files.isRegularFile(file));
        m.put("at", System.currentTimeMillis());
        if (Files.isRegularFile(file)) {
            Files.copy(file, dir.resolve(seq + ".bin"), StandardCopyOption.REPLACE_EXISTING);
        }
        Files.writeString(list, Json.write(m) + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                StandardOpenOption.APPEND);
    }

    /**
     * Lists what a run changed.
     *
     * @param run the run
     * @return {@code {path, existed, at}} per change, oldest first
     * @throws IOException if the list cannot be read
     */
    public synchronized List<Map<String, Object>> changes(String run) throws IOException {
        Path list = folder.resolve(run).resolve("changes.jsonl");
        List<Map<String, Object>> out = new ArrayList<>();
        if (!Files.exists(list)) {
            return out;
        }
        for (String line : Files.readAllLines(list, StandardCharsets.UTF_8)) {
            if (!line.isBlank()) {
                out.add(Json.object(Json.parse(line)));
            }
        }
        return out;
    }

    /**
     * Puts every file a run changed back as it was, newest change first, and forgets the
     * checkpoints.
     *
     * @param run the run
     * @return the files put back
     * @throws IOException if a file cannot be restored
     */
    public synchronized List<String> undo(String run) throws IOException {
        List<Map<String, Object>> changes = changes(run);
        List<String> restored = new ArrayList<>();
        Path dir = folder.resolve(run);
        for (Map<String, Object> c : changes.reversed()) {
            Path file = Path.of(Json.string(c, "path", ""));
            if (Boolean.TRUE.equals(c.get("existed"))) {
                Files.createDirectories(file.toAbsolutePath().getParent());
                Files.copy(dir.resolve(Json.integer(c, "seq", 0) + ".bin"), file, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.deleteIfExists(file);
            }
            if (!restored.contains(file.toString())) {
                restored.add(file.toString());
            }
        }
        forget(run);
        return restored;
    }

    /**
     * Drops a run's checkpoints.
     *
     * @param run the run
     * @throws IOException if they cannot be deleted
     */
    public synchronized void forget(String run) throws IOException {
        Path dir = folder.resolve(run);
        if (!Files.exists(dir)) {
            return;
        }
        try (Stream<Path> s = Files.walk(dir)) {
            for (Path p : s.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(p);
            }
        }
    }
}
