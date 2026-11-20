package io.github.zzuegg.brainiac.studio.builder;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * The builder's saved graphs, one JSON file each in {@code graphs}, named by the graph's name.
 * A graph that was built also has a model file of the same name in {@code models}.
 */
public final class Graphs {

    private static final System.Logger LOG = System.getLogger(Graphs.class.getName());
    private static final Pattern NAME = Pattern.compile("[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}");

    private final Path graphs;
    private final Path models;

    /**
     * Opens the graphs of a data folder; the folders are made on the first save.
     *
     * @param folder the studio's data folder
     */
    public Graphs(Path folder) {
        this.graphs = folder.resolve("graphs");
        this.models = folder.resolve("models");
    }

    /**
     * Checks a graph's name.
     *
     * @param v the name
     * @return it
     * @throws IllegalArgumentException if it cannot name a file
     */
    public static String name(Object v) {
        if (!(v instanceof String s) || !NAME.matcher(s).matches() || s.contains("..") || s.endsWith(".") || s.endsWith(" ")) {
            throw new IllegalArgumentException("a model name is 1 to 64 letters, digits, spaces, dots, dashes or underscores");
        }
        return s;
    }

    /** @return the folder built models go to */
    public Path models() {
        return models;
    }

    /**
     * Returns a graph's file.
     *
     * @param name its name
     * @return the file, which may not exist
     */
    public Path file(String name) {
        return graphs.resolve(name(name) + ".json");
    }

    /**
     * Returns the model built from a graph.
     *
     * @param name the graph's name
     * @return the file, which may not exist
     */
    public Path model(String name) {
        return models.resolve(name(name) + ".brainiac");
    }

    /**
     * Saves a graph, replacing the one with its name.
     *
     * @param name its name
     * @param graph the graph, with at least a list of {@code nodes}
     * @throws IOException if it cannot be written
     */
    public synchronized void save(String name, Map<String, Object> graph) throws IOException {
        Json.array(graph.get("nodes"));   // must at least have a list of blocks
        Files.createDirectories(graphs);
        Map<String, Object> doc = new LinkedHashMap<>(graph);
        doc.put("name", name(name));
        doc.put("updated", System.currentTimeMillis());
        Path tmp = graphs.resolve(name + ".json.tmp");
        Files.writeString(tmp, Json.write(doc), StandardCharsets.UTF_8);
        Files.move(tmp, file(name), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /**
     * Reads one graph.
     *
     * @param name its name
     * @return the graph, or empty
     * @throws IOException if it cannot be read
     */
    public synchronized Optional<Map<String, Object>> get(String name) throws IOException {
        Path f = file(name);
        if (!Files.isRegularFile(f)) {
            return Optional.empty();
        }
        Map<String, Object> g = Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8)));
        g.put("name", name);
        g.put("built", Files.exists(model(name)));
        return Optional.of(g);
    }

    /**
     * Lists the graphs, the most recently saved first. A file that cannot be read is skipped.
     *
     * @return the graphs
     * @throws IOException if the folder cannot be listed
     */
    public synchronized List<Map<String, Object>> list() throws IOException {
        List<Map<String, Object>> out = new ArrayList<>();
        if (!Files.isDirectory(graphs)) {
            return out;
        }
        try (DirectoryStream<Path> files = Files.newDirectoryStream(graphs, "*.json")) {
            for (Path f : files) {
                String name = f.getFileName().toString().replaceFirst("\\.json$", "");
                try {
                    get(name).ifPresent(out::add);
                } catch (IllegalArgumentException | IOException bad) {
                    LOG.log(System.Logger.Level.WARNING, "skipping " + f + ": " + bad.getMessage());
                }
            }
        }
        out.sort(Comparator.comparingLong((Map<String, Object> g) -> Json.integer(g, "updated", 0)).reversed());
        return out;
    }

    /**
     * Deletes a graph; the model built from it stays.
     *
     * @param name its name
     * @return whether there was one
     * @throws IOException if it cannot be deleted
     */
    public synchronized boolean delete(String name) throws IOException {
        return Files.deleteIfExists(file(name));
    }

    /**
     * Renames a graph, and the model built from it.
     *
     * @param from its name
     * @param to the new name
     * @throws IllegalArgumentException if there is no such graph or the new name is taken
     * @throws IOException if the files cannot be moved
     */
    public synchronized void rename(String from, String to) throws IOException {
        if (from.equals(name(to))) {
            return;
        }
        if (!Files.isRegularFile(file(from))) {
            throw new IllegalArgumentException("there is no graph " + from);
        }
        if (Files.exists(file(to)) || Files.exists(model(to))) {
            throw new IllegalArgumentException("there is already a graph or model named " + to);
        }
        Map<String, Object> g = Json.object(Json.parse(Files.readString(file(from), StandardCharsets.UTF_8)));
        g.put("name", to);
        Files.writeString(file(to), Json.write(g), StandardCharsets.UTF_8);
        Files.delete(file(from));
        if (Files.exists(model(from))) {
            Files.move(model(from), model(to));
        }
    }

    /**
     * Finds a free name like a given one: the name itself, or it with " 2", " 3" and so on.
     *
     * @param base the name wanted
     * @return a name no graph or model has
     */
    public synchronized String free(String base) {
        String b = base.length() > 58 ? base.substring(0, 58).strip() : base;
        String n = name(b);
        for (int i = 2; Files.exists(file(n)) || Files.exists(model(n)); i++) {
            n = b + " " + i;
        }
        return n;
    }
}
