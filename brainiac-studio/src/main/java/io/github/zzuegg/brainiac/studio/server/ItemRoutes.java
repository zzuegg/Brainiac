package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.items.Item;
import io.github.zzuegg.brainiac.studio.items.Items;
import io.github.zzuegg.brainiac.studio.items.Trash;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The same actions for every kind of item ({@code chat}, {@code agent}, {@code graph},
 * {@code project}).
 *
 * <ul>
 *   <li>{@code POST /api/items {project?, kinds?}}: {@code {project, items}}, the items of the
 *       project named (else the call's project), newest first; {@code "project": "*"} lists every
 *       project's. Projects are listed only when asked for in {@code kinds}.
 *   <li>{@code POST /api/items/rename {kind, id, title}}, {@code POST /api/items/move {kind, id,
 *       project}} and {@code POST /api/items/duplicate {kind, id}}: {@code {item}} afterwards
 *       (a renamed graph has a new id).
 *   <li>{@code POST /api/items/delete {kind, id}}: moves the item to the trash and answers
 *       {@code {trash: {id, deleted, expires, item}}}; that id undoes it.
 *   <li>{@code GET /api/trash}: {@code {entries}}, the most recently deleted first, kept 30 days.
 *   <li>{@code POST /api/trash/restore {id}}: {@code {item}} put back.
 *   <li>{@code POST /api/trash/empty {id?}}: deletes one entry for good, or all of them;
 *       answers {@code {entries}} left.
 * </ul>
 */
final class ItemRoutes {

    private static final int MAX_BODY = 1 << 16;

    private final Items items;

    ItemRoutes(Items items) {
        this.items = items;
    }

    /**
     * Returns the project the current API call belongs to.
     *
     * @return its id, or null outside a call
     */
    static String callerProject() {
        return ChatService.SCOPE.isBound() && ChatService.SCOPE.get() instanceof Project p ? p.id() : null;
    }

    void register(StudioServer server) {
        server.post("/api/items", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            String project = Json.string(body, "project", null);
            if (project == null || project.isBlank()) {
                project = callerProject() == null ? Items.DEFAULT : callerProject();
            }
            List<String> kinds = body.get("kinds") == null ? null
                    : Json.array(body.get("kinds")).stream().map(String::valueOf).toList();
            List<Item> list = items.list("*".equals(project) ? null : project, kinds);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("project", project);
            out.put("items", list.stream().map(Item::toJson).toList());
            Exchanges.json(ex, 200, out);
        });
        server.post("/api/items/rename", ex -> {
            Map<String, Object> b = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("item", items.rename(kind(b), id(b), Json.string(b, "title", "")).toJson()));
        });
        server.post("/api/items/move", ex -> {
            Map<String, Object> b = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("item", items.move(kind(b), id(b), Json.string(b, "project", "")).toJson()));
        });
        server.post("/api/items/duplicate", ex -> {
            Map<String, Object> b = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("item", items.duplicate(kind(b), id(b)).toJson()));
        });
        server.post("/api/items/delete", ex -> {
            Map<String, Object> b = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("trash", items.delete(kind(b), id(b)).toJson()));
        });
        server.get("/api/trash", ex -> Exchanges.json(ex, 200, entries()));
        server.post("/api/trash/restore", ex -> {
            Map<String, Object> b = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("item", items.restore(Json.string(b, "id", "")).toJson()));
        });
        server.post("/api/trash/empty", ex -> {
            Map<String, Object> b = Exchanges.body(ex, MAX_BODY);
            String id = Json.string(b, "id", null);
            if (id != null) {
                items.purge(id);
            } else {
                for (Trash.Entry e : items.trash()) {
                    items.purge(e.id());
                }
            }
            Exchanges.json(ex, 200, entries());
        });
    }

    private Map<String, Object> entries() {
        return Map.of("entries", items.trash().stream().map(Trash.Entry::toJson).toList());
    }

    private static String kind(Map<String, Object> b) {
        return Json.string(b, "kind", "");
    }

    private static String id(Map<String, Object> b) {
        String id = Json.string(b, "id", "");
        if (id.isBlank()) {
            throw new IllegalArgumentException("say which item: {kind, id}");
        }
        return id;
    }
}
