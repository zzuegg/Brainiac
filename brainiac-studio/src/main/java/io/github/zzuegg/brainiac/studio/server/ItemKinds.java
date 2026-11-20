package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.agent.AgentRuns;
import io.github.zzuegg.brainiac.studio.builder.Graphs;
import io.github.zzuegg.brainiac.studio.chat.Conversations;
import io.github.zzuegg.brainiac.studio.items.Item;
import io.github.zzuegg.brainiac.studio.items.ItemKind;
import io.github.zzuegg.brainiac.studio.project.Project;
import io.github.zzuegg.brainiac.studio.project.Projects;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** The studio's stores as {@link ItemKind}s. */
final class ItemKinds {

    private ItemKinds() {
    }

    /** Conversations, one file each; their project is kept by the owners. */
    static final class Chats implements ItemKind {

        private static final SecureRandom RANDOM = new SecureRandom();
        private final Conversations store;

        Chats(Conversations store) {
            this.store = store;
        }

        @Override
        public String kind() {
            return CHAT;
        }

        // what each file said when it last changed: the list is asked for every few seconds and
        // a conversation file holds every message, so only changed files are read again
        private final Map<Path, Map.Entry<Long, Item>> seen = new java.util.concurrent.ConcurrentHashMap<>();

        @Override
        public List<Item> list() throws IOException {
            Path folder = store.folder();
            if (!Files.isDirectory(folder)) {
                return List.of();
            }
            List<Item> out = new ArrayList<>();
            java.util.Set<Path> present = new java.util.HashSet<>();
            try (java.nio.file.DirectoryStream<Path> ds = Files.newDirectoryStream(folder, "*.json")) {
                for (Path f : ds) {
                    present.add(f);
                    try {
                        long modified = Files.getLastModifiedTime(f).toMillis() ^ Files.size(f);
                        Map.Entry<Long, Item> e = seen.get(f);
                        if (e == null || e.getKey() != modified) {
                            e = Map.entry(modified, item(Json.object(Json.parse(Files.readString(f)))));
                            seen.put(f, e);
                        }
                        out.add(e.getValue());
                    } catch (IOException | IllegalArgumentException bad) {
                        seen.remove(f);   // being written, or not a conversation: skipped, as the store does
                    }
                }
            }
            seen.keySet().retainAll(present);
            return out;
        }

        @Override
        public Optional<Item> get(String id) throws IOException {
            return store.get(id).map(Chats::item);
        }

        private static Item item(Map<String, Object> c) {
            Map<String, Object> extra = new LinkedHashMap<>();
            extra.put("status", Json.string(c, "status", "open"));
            extra.put("messages", c.get("messages") instanceof List<?> l ? l.size() : 0);
            return new Item(CHAT, Json.string(c, "id", ""), Json.string(c, "title", "Untitled"), null,
                    Json.integer(c, "updated", 0), extra);
        }

        @Override
        public String rename(String id, String title) throws IOException {
            Map<String, Object> c = store.get(id).orElseThrow(() -> new IllegalArgumentException("there is no chat " + id));
            c.put("title", title);
            store.save(c);
            return id;
        }

        @Override
        public String duplicate(String id) throws IOException {
            Map<String, Object> c = store.get(id).orElseThrow(() -> new IllegalArgumentException("there is no chat " + id));
            String copy;
            do {
                copy = Long.toString(System.currentTimeMillis(), 36) + Integer.toString(RANDOM.nextInt(1 << 20), 36);
            } while (store.get(copy).isPresent());
            c.put("id", copy);
            String title = Json.string(c, "title", "Untitled") + " (copy)";
            c.put("title", title.length() > 500 ? title.substring(0, 500) : title);
            c.put("updated", System.currentTimeMillis());
            store.save(c);
            return copy;
        }

        @Override
        public Map<String, Object> detach(String id, Path into) throws IOException {
            Path f = store.file(id);
            if (!Files.isRegularFile(f)) {
                throw new IllegalArgumentException("there is no chat " + id);
            }
            Files.move(f, into.resolve(f.getFileName()));
            return Map.of();
        }

        @Override
        public void attach(String id, Path from, Map<String, Object> kept) throws IOException {
            Path f = store.file(id);
            if (Files.exists(f)) {
                throw new IllegalArgumentException("a chat with this id is back already");
            }
            Files.createDirectories(f.getParent());
            Files.move(from.resolve(f.getFileName()), f);
        }
    }

    /** The builder's graphs, named by their file; their project is kept by the owners. */
    static final class GraphKind implements ItemKind {

        private final Graphs store;

        GraphKind(Graphs store) {
            this.store = store;
        }

        @Override
        public String kind() {
            return GRAPH;
        }

        @Override
        public List<Item> list() throws IOException {
            return store.list().stream().map(GraphKind::item).toList();
        }

        @Override
        public Optional<Item> get(String id) throws IOException {
            return store.get(id).map(GraphKind::item);
        }

        private static Item item(Map<String, Object> g) {
            String name = Json.string(g, "name", "");
            Map<String, Object> extra = new LinkedHashMap<>();
            extra.put("built", Boolean.TRUE.equals(g.get("built")));
            extra.put("nodes", g.get("nodes") instanceof List<?> l ? l.size() : 0);
            return new Item(GRAPH, name, name, null, Json.integer(g, "updated", 0), extra);
        }

        @Override
        public String rename(String id, String title) throws IOException {
            store.rename(id, Graphs.name(title));
            return title;
        }

        @Override
        public String duplicate(String id) throws IOException {
            Map<String, Object> g = store.get(id).orElseThrow(() -> new IllegalArgumentException("there is no graph " + id));
            String copy = store.free(id + " copy");
            g.remove("built");
            store.save(copy, g);
            return copy;
        }

        @Override
        public Map<String, Object> detach(String id, Path into) throws IOException {
            Path f = store.file(id);
            if (!Files.isRegularFile(f)) {
                throw new IllegalArgumentException("there is no graph " + id);
            }
            Files.move(f, into.resolve("graph.json"));
            return Map.of();
        }

        @Override
        public void attach(String id, Path from, Map<String, Object> kept) throws IOException {
            Path f = store.file(id);
            if (Files.exists(f)) {
                throw new IllegalArgumentException("there is a graph named " + id + " again; rename it first");
            }
            Files.createDirectories(f.getParent());
            Files.move(from.resolve("graph.json"), f);
        }
    }

    /** Agent runs, which keep their own project. A run's sub-agents go with it. */
    static final class Agents implements ItemKind {

        private final AgentRuns runs;

        Agents(AgentRuns runs) {
            this.runs = runs;
        }

        @Override
        public String kind() {
            return AGENT;
        }

        @Override
        public boolean keepsProject() {
            return true;
        }

        @Override
        public List<Item> list() {
            // sub-agents show inside the run that started them
            return runs.list().stream().filter(r -> r.get("parent") == null).map(Agents::item).toList();
        }

        @Override
        public Optional<Item> get(String id) {
            return runs.get(id).map(Agents::item);
        }

        private static Item item(Map<String, Object> r) {
            String goal = Json.string(r, "goal", "");
            String title = Json.string(r, "title", null);
            if (title == null) {
                String first = goal.strip().lines().findFirst().orElse("Agent");
                title = first.length() > 80 ? first.substring(0, 77) + "..." : first;
            }
            Map<String, Object> extra = new LinkedHashMap<>();
            extra.put("state", Json.string(r, "state", "stopped"));
            extra.put("role", Json.string(r, "role", ""));
            extra.put("goal", goal);
            if (r.get("pending") != null) {
                extra.put("waiting", true);
            }
            long updated = Math.max(Json.integer(r, "ended", 0), Json.integer(r, "created", 0));
            return new Item(AGENT, Json.string(r, "id", ""), title, Json.string(r, "project", Project.DEFAULT), updated, extra);
        }

        @Override
        public String rename(String id, String title) {
            runs.rename(id, title);
            return id;
        }

        @Override
        public void move(String id, String project) {
            runs.move(id, project);
        }

        @Override
        public Map<String, Object> detach(String id, Path into) throws IOException {
            return Map.of("runs", runs.detach(id, into));
        }

        @Override
        public void attach(String id, Path from, Map<String, Object> kept) throws IOException {
            runs.attach(ids(kept, id), from);
        }

        @Override
        public void purge(String id, Map<String, Object> kept) throws IOException {
            runs.forget(ids(kept, id));
        }

        private static List<String> ids(Map<String, Object> kept, String id) {
            List<String> out = new ArrayList<>();
            for (Object o : Json.array(kept.getOrDefault("runs", List.of(id)))) {
                out.add(String.valueOf(o));
            }
            return out;
        }
    }

    /** Projects; deleting one keeps its items, which come back with it. */
    static final class ProjectKind implements ItemKind {

        private final Projects store;

        ProjectKind(Projects store) {
            this.store = store;
        }

        @Override
        public String kind() {
            return PROJECT;
        }

        @Override
        public List<Item> list() {
            return store.list().stream().map(ProjectKind::item).toList();
        }

        private static Item item(Project p) {
            Map<String, Object> extra = new LinkedHashMap<>();
            extra.put("description", p.description());
            if (p.folder() != null) {
                extra.put("folder", p.folder().toString());
            }
            return new Item(PROJECT, p.id(), p.name(), null, 0, extra);
        }

        @Override
        public String rename(String id, String title) throws IOException {
            Project p = store.get(id).orElseThrow(() -> new IllegalArgumentException("there is no project " + id));
            store.save(new Project(p.id(), title, p.description(), p.folder(), p.tools(), p.models(), p.routes(),
                    p.runners()), false);
            return id;
        }

        @Override
        public Map<String, Object> detach(String id, Path into) throws IOException {
            Project p = store.get(id).orElseThrow(() -> new IllegalArgumentException("there is no project " + id));
            if (Project.DEFAULT.equals(id)) {
                throw new IllegalArgumentException("the default project stays");
            }
            Map<String, Object> kept = Map.of("project", p.toJson());
            store.delete(id);
            return kept;
        }

        @Override
        public void attach(String id, Path from, Map<String, Object> kept) throws IOException {
            if (store.get(id).isPresent()) {
                throw new IllegalArgumentException("there is a project " + id + " again");
            }
            store.save(Project.fromJson(Json.object(kept.get("project"))), false);
        }
    }
}
