package io.github.zzuegg.brainiac.studio.server;

import com.sun.net.httpserver.HttpExchange;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.studio.builder.Graphs;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog;
import io.github.zzuegg.brainiac.studio.builder.ModelGraph;
import io.github.zzuegg.brainiac.tensor.Shape;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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

    private static final int MAX_BODY = 8 << 20;

    private final Path models;
    private final Graphs graphs;
    private final io.github.zzuegg.brainiac.studio.items.Owners owners;
    private volatile LayerCatalog catalog;
    private volatile ModelGraph checker;

    BuilderRoutes(Graphs graphs, io.github.zzuegg.brainiac.studio.items.Owners owners) {
        this.graphs = graphs;
        this.models = graphs.models();
        this.owners = owners;
    }

    void register(StudioServer server) {
        server.get("/api/builder/layers", ex -> Exchanges.json(ex, 200, catalog().toJson()));
        server.post("/api/builder/check", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, checker().check(Json.object(body.get("graph"))).toJson());
        });
        server.post("/api/builder/build", this::build);
        server.get("/api/builder/graphs", ex -> Exchanges.json(ex, 200, Map.of("graphs", graphs.list())));
        server.post("/api/builder/graphs", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            String name = name(body.get("name"));
            if (Boolean.TRUE.equals(body.get("delete"))) {
                graphs.delete(name);
            } else {
                saveGraph(name, Json.object(body.get("graph")));
            }
            Map<String, Object> out = new LinkedHashMap<>(Map.of("ok", true));
            graphs.get(name).ifPresent(g -> out.put("updated", g.get("updated")));
            Exchanges.json(ex, 200, out);
        });
        server.post("/api/builder/graphs/get", ex -> {
            var g = graphs.get(name(Exchanges.body(ex, MAX_BODY).get("name")));
            if (g.isPresent()) {
                Exchanges.json(ex, 200, Map.of("graph", g.get()));
            } else {
                Exchanges.error(ex, 404, "there is no such network");
            }
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

    // a new graph belongs to the project the call is made in
    private void saveGraph(String name, Map<String, Object> graph) throws IOException {
        graphs.save(name, graph);
        owners.claim(io.github.zzuegg.brainiac.studio.items.ItemKind.GRAPH, name, ItemRoutes.callerProject());
    }

    private static String name(Object v) {
        return Graphs.name(v);
    }

    private static List<Integer> dims(Shape s) {
        List<Integer> out = new ArrayList<>();
        for (int d : s.toArray()) {
            out.add(d);
        }
        return out;
    }
}
