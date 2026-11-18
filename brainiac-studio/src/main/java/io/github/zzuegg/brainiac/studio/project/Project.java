package io.github.zzuegg.brainiac.studio.project;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.GpuResidency;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * A project: the agents, tools, models and files that belong together, and how its tasks run.
 *
 * @param id the project's id: lowercase letters, digits and {@code -}
 * @param name its name
 * @param description what it is for, given to its agents
 * @param folder the folder its agents may read, or null for none
 * @param tools the tools its agents may call, or null for all of them
 * @param models the models it may use, as {@code provider/id} keys or {@code provider/} prefixes,
 *     or null for all of them
 * @param routes how each kind of task runs, by medium ({@code text}, {@code image},
 *     {@code speech}, {@code transcribe}, {@code video})
 */
public record Project(String id, String name, String description, Path folder, List<String> tools,
        List<String> models, Map<String, ChatService.Route> routes) implements ChatService.Scope {

    /** The project everything belongs to unless another is chosen: it limits nothing. */
    public static final String DEFAULT = "default";

    /** The kinds of task a project routes. */
    public static final List<String> MEDIA = List.of("text", "image", "speech", "transcribe", "video");

    /** Copies the lists, so a project never changes. */
    public Project {
        if (id == null || !id.matches("[a-z0-9][a-z0-9-]{0,39}")) {
            throw new IllegalArgumentException("a project's id is up to 40 lowercase letters, digits and -");
        }
        name = name == null || name.isBlank() ? id : name.strip();
        description = description == null ? "" : description.strip();
        tools = tools == null ? null : List.copyOf(tools);
        models = models == null ? null : List.copyOf(models);
        Map<String, ChatService.Route> rs = new LinkedHashMap<>();
        if (routes != null) {
            for (Map.Entry<String, ChatService.Route> e : routes.entrySet()) {
                if (!MEDIA.contains(e.getKey())) {
                    throw new IllegalArgumentException("no kind of task " + e.getKey() + ": use " + String.join(", ", MEDIA));
                }
                if (e.getValue() != null) {
                    rs.put(e.getKey(), e.getValue());
                }
            }
        }
        routes = java.util.Collections.unmodifiableMap(rs);
    }

    /**
     * Returns the project that limits nothing.
     *
     * @return it
     */
    public static Project unlimited() {
        return new Project(DEFAULT, "Default", "", null, null, null, Map.of());
    }

    @Override
    public ChatService.Route route(String medium) {
        return routes.get(medium);
    }

    @Override
    public boolean allows(String key) {
        if (models == null) {
            return true;
        }
        for (String m : models) {
            if (m.equals(key) || (m.endsWith("/") && key.startsWith(m))) {
                return true;
            }
        }
        // where a route sends a task is allowed by naming it
        return routes.values().stream().anyMatch(r -> key.equals(r.model()) || key.equals(r.fallback()));
    }

    /**
     * Says whether its agents may call a tool.
     *
     * @param tool the tool's name
     * @return whether it is allowed
     */
    public boolean allowsTool(String tool) {
        return tools == null || tools.contains(tool);
    }

    /**
     * Resolves a file its agents asked for inside its folder.
     *
     * @param relative the path inside the folder; empty for the folder itself
     * @return the file
     * @throws IllegalArgumentException if the project has no folder or the path leaves it
     */
    public Path inFolder(String relative) {
        if (folder == null) {
            throw new IllegalArgumentException("the project " + name + " has no folder");
        }
        Path root = folder.toAbsolutePath().normalize();
        Path p = root.resolve(relative == null ? "" : relative.replace('\\', '/')).normalize();
        if (!p.startsWith(root)) {
            throw new IllegalArgumentException(relative + " is outside the project's folder");
        }
        try {
            // a link inside the folder must not lead out of it
            if (Files.exists(p) && !p.toRealPath().startsWith(root.toRealPath())) {
                throw new IllegalArgumentException(relative + " leads outside the project's folder");
            }
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException("cannot read " + relative + ": " + e.getMessage());
        }
        return p;
    }

    /**
     * Describes it for the page and the file it is kept in.
     *
     * @return {@code {id, name, description, folder, tools, models, routes}}
     */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("name", name);
        m.put("description", description);
        m.put("folder", folder == null ? null : folder.toString());
        m.put("tools", tools);
        m.put("models", models);
        Map<String, Object> rs = new LinkedHashMap<>();
        for (Map.Entry<String, ChatService.Route> e : routes.entrySet()) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("model", e.getValue().model());
            r.put("policy", e.getValue().policy() == null ? null : e.getValue().policy().key());
            r.put("fallback", e.getValue().fallback());
            rs.put(e.getKey(), r);
        }
        m.put("routes", rs);
        return m;
    }

    /**
     * Reads one from what {@link #toJson} wrote, or from the page.
     *
     * @param m the JSON object
     * @return the project
     * @throws IllegalArgumentException if a field is wrong
     */
    public static Project fromJson(Map<String, Object> m) {
        String id = Json.string(m, "id", "");
        if (id.isBlank()) {
            id = slug(Json.string(m, "name", ""));
        }
        String folder = Json.string(m, "folder", "");
        Map<String, ChatService.Route> routes = new LinkedHashMap<>();
        if (m.get("routes") != null) {
            for (Map.Entry<String, Object> e : Json.object(m.get("routes")).entrySet()) {
                if (e.getValue() == null) {
                    continue;
                }
                Map<String, Object> r = Json.object(e.getValue());
                String policy = Json.string(r, "policy", "");
                String model = Json.string(r, "model", "");
                String fallback = Json.string(r, "fallback", "");
                if (model.isBlank() && policy.isBlank() && fallback.isBlank()) {
                    continue;
                }
                routes.put(e.getKey(), new ChatService.Route(model.isBlank() ? null : model,
                        policy.isBlank() ? null : GpuResidency.Policy.of(policy), fallback.isBlank() ? null : fallback));
            }
        }
        return new Project(id, Json.string(m, "name", id), Json.string(m, "description", ""),
                folder.isBlank() ? null : Path.of(folder), strings(m.get("tools")), strings(m.get("models")), routes);
    }

    private static List<String> strings(Object o) {
        if (o == null) {
            return null;
        }
        List<String> out = new ArrayList<>();
        for (Object x : Json.array(o)) {
            String s = String.valueOf(x).strip();
            if (!s.isEmpty() && !out.contains(s)) {
                out.add(s);
            }
        }
        return out;
    }

    /**
     * Makes an id from a name.
     *
     * @param name the name
     * @return lowercase letters, digits and {@code -}
     */
    static String slug(String name) {
        String s = name.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-").replaceAll("^-+|-+$", "");
        if (s.length() > 40) {
            s = s.substring(0, 40).replaceAll("-+$", "");
        }
        return s.isEmpty() ? "project" : s;
    }

    /**
     * Returns the route's model for a kind of task, if it names one.
     *
     * @param medium the kind
     * @return the model key
     */
    public Optional<String> model(String medium) {
        ChatService.Route r = routes.get(medium);
        return r == null || r.model() == null ? Optional.empty() : Optional.of(r.model());
    }
}
