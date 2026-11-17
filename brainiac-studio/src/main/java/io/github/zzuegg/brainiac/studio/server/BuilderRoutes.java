package io.github.zzuegg.brainiac.studio.server;

import com.sun.net.httpserver.HttpExchange;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog;
import io.github.zzuegg.brainiac.studio.builder.ModelGraph;
import io.github.zzuegg.brainiac.tensor.Shape;
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
import java.util.regex.Pattern;

/**
 * The model builder API.
 *
 * <ul>
 *   <li>{@code GET /api/builder/layers} lists every layer the installed modules offer and the
 *       options of each, found from their public API.
 *   <li>{@code POST /api/builder/check} with {@code {"graph": {...}}} works out every block's
 *       output shape with the core's own rules, and says why a block cannot take its input.
 *   <li>{@code POST /api/builder/build} with {@code {"name": "resnet-s", "graph": {...}, "seed": 42}}
 *       builds the model with fresh weights, saves it as {@code models/<name>.brainiac} in the data
 *       folder with the graph beside it in {@code graphs}, and answers with its parameter count
 *       and summary.
 *   <li>{@code GET /api/builder/graphs} lists the saved graphs; {@code POST /api/builder/graphs}
 *       with {@code {"name": "...", "graph": {...}}} saves one, and with {@code "delete": true}
 *       deletes it.
 * </ul>
 */
final class BuilderRoutes {

    private static final Pattern NAME = Pattern.compile("[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}");
    private static final int MAX_BODY = 8 << 20;

    private final Path models;
    private final Path graphs;
    private volatile LayerCatalog catalog;
    private volatile ModelGraph checker;

    BuilderRoutes(Path folder) {
        this.models = folder.resolve("models");
        this.graphs = folder.resolve("graphs");
    }

    void register(StudioServer server) {
        server.get("/api/builder/layers", ex -> Exchanges.json(ex, 200, catalog().toJson()));
        server.post("/api/builder/check", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, checker().check(Json.object(body.get("graph"))).toJson());
        });
        server.post("/api/builder/build", this::build);
        server.get("/api/builder/graphs", ex -> Exchanges.json(ex, 200, Map.of("graphs", listGraphs())));
        server.post("/api/builder/graphs", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            String name = name(body.get("name"));
            if (Boolean.TRUE.equals(body.get("delete"))) {
                Files.deleteIfExists(graphs.resolve(name + ".json"));
            } else {
                saveGraph(name, Json.object(body.get("graph")));
            }
            Exchanges.json(ex, 200, Map.of("ok", true));
        });
    }

    private LayerCatalog catalog() {
        LayerCatalog c = catalog;
        if (c == null) {
            synchronized (this) {
                if (catalog == null) {
                    catalog = LayerCatalog.discover();
                    checker = new ModelGraph(catalog);
                }
                c = catalog;
            }
        }
        return c;
    }

    private ModelGraph checker() {
        catalog();
        return checker;
    }

    private void build(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
        String name = name(body.get("name"));
        Map<String, Object> graph = Json.object(body.get("graph"));
        long seed = Json.integer(body, "seed", 42);
        Model model = checker().build(graph, seed);
        Files.createDirectories(models);
        Path file = models.resolve(name + ".brainiac");
        Path tmp = models.resolve(name + ".brainiac.tmp");
        String saved;
        String note = null;
        try {
            model.save(tmp);
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            saved = file.toString();
        } catch (IllegalStateException | UnsupportedOperationException cannot) {
            Files.deleteIfExists(tmp);
            saved = null;
            note = "built, but not saved: " + cannot.getMessage();
        }
        saveGraph(name, graph);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("name", name);
        out.put("parameters", model.parameterCount());
        out.put("inputShape", dims(model.inputShape()));
        out.put("outputShape", dims(model.outputShape()));
        out.put("summary", model.summary());
        if (saved != null) {
            out.put("file", saved);
        }
        if (note != null) {
            out.put("note", note);
        }
        Exchanges.json(ex, 200, out);
    }

    private synchronized void saveGraph(String name, Map<String, Object> graph) throws IOException {
        Json.array(graph.get("nodes"));   // must at least have a list of blocks
        Files.createDirectories(graphs);
        Map<String, Object> doc = new LinkedHashMap<>(graph);
        doc.put("name", name);
        doc.put("updated", System.currentTimeMillis());
        Path tmp = graphs.resolve(name + ".json.tmp");
        Files.writeString(tmp, Json.write(doc), StandardCharsets.UTF_8);
        Files.move(tmp, graphs.resolve(name + ".json"), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private synchronized List<Map<String, Object>> listGraphs() throws IOException {
        List<Map<String, Object>> out = new ArrayList<>();
        if (!Files.isDirectory(graphs)) {
            return out;
        }
        try (DirectoryStream<Path> files = Files.newDirectoryStream(graphs, "*.json")) {
            for (Path f : files) {
                try {
                    Map<String, Object> g = Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8)));
                    String name = f.getFileName().toString().replaceFirst("\\.json$", "");
                    g.put("name", name);
                    g.put("built", Files.exists(models.resolve(name + ".brainiac")));
                    out.add(g);
                } catch (IllegalArgumentException | IOException bad) {
                    System.getLogger(BuilderRoutes.class.getName()).log(System.Logger.Level.WARNING,
                            "skipping " + f + ": " + bad.getMessage());
                }
            }
        }
        out.sort(Comparator.comparingLong((Map<String, Object> g) -> Json.integer(g, "updated", 0)).reversed());
        return out;
    }

    private static String name(Object v) {
        if (!(v instanceof String s) || !NAME.matcher(s).matches() || s.contains("..") || s.endsWith(".") || s.endsWith(" ")) {
            throw new IllegalArgumentException("a model name is 1 to 64 letters, digits, spaces, dots, dashes or underscores");
        }
        return s;
    }

    private static List<Integer> dims(Shape s) {
        List<Integer> out = new ArrayList<>();
        for (int d : s.toArray()) {
            out.add(d);
        }
        return out;
    }
}
