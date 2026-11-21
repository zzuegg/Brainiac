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
 * @param runners machines a route may rent, by name
 * @param parents the projects this one is part of; it may be part of several, and it can never
 *     do more than every one of them allows (see {@link Projects#get})
 * @param requests who answers its subprojects' requests: {@value #BY_USER} or {@value #BY_AGENTS}
 */
public record Project(String id, String name, String description, Path folder, List<String> tools,
        List<String> models, Map<String, ChatService.Route> routes, Map<String, Map<String, Object>> runners,
        List<String> parents, String requests)
        implements ChatService.Scope {

    /** Requests from subprojects wait for the user. */
    public static final String BY_USER = "user";

    /** Requests from subprojects may be answered by this project's agents, within their own limits. */
    public static final String BY_AGENTS = "agents";

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
        Map<String, Map<String, Object>> rn = new LinkedHashMap<>();
        if (runners != null) {
            for (Map.Entry<String, Map<String, Object>> e : runners.entrySet()) {
                if (!e.getKey().matches("[A-Za-z0-9_-]{1,40}")) {
                    throw new IllegalArgumentException("a runner's name is up to 40 letters, digits, _ and -");
                }
                if (e.getValue() != null) {
                    // a field left empty on the page is just not set
                    Map<String, Object> r = new LinkedHashMap<>(e.getValue());
                    r.values().removeIf(v -> v == null || "".equals(v));
                    rn.put(e.getKey(), java.util.Collections.unmodifiableMap(r));
                }
            }
        }
        runners = java.util.Collections.unmodifiableMap(rn);
        List<String> ps = new ArrayList<>();
        for (String parent : parents == null ? List.<String>of() : parents) {
            if (parent != null && !parent.isBlank() && !ps.contains(parent.strip())) {
                ps.add(parent.strip());
            }
        }
        if (ps.contains(id)) {
            throw new IllegalArgumentException("a project cannot be part of itself");
        }
        if (!ps.isEmpty() && DEFAULT.equals(id)) {
            throw new IllegalArgumentException("the default project is part of no other");
        }
        if (ps.contains(DEFAULT)) {
            throw new IllegalArgumentException("the default project holds no subprojects");
        }
        parents = List.copyOf(ps);
        requests = BY_AGENTS.equals(requests) ? BY_AGENTS : BY_USER;
    }

    /**
     * Makes a project that is part of no other.
     *
     * @param id its id
     * @param name its name
     * @param description what it is for
     * @param folder the folder its agents may read, or null
     * @param tools the tools allowed, or null for all
     * @param models the models allowed, or null for all
     * @param routes the routes by medium
     * @param runners machines a route may rent
     */
    public Project(String id, String name, String description, Path folder, List<String> tools, List<String> models,
            Map<String, ChatService.Route> routes, Map<String, Map<String, Object>> runners) {
        this(id, name, description, folder, tools, models, routes, runners, List.of(), BY_USER);
    }

    /** @return this project with another id */
    public Project withId(String newId) {
        return new Project(newId, name, description, folder, tools, models, routes, runners, parents, requests);
    }

    /** @return this project with another name */
    public Project withName(String newName) {
        return new Project(id, newName, description, folder, tools, models, routes, runners, parents, requests);
    }

    /** @return this project with other routes */
    public Project withRoutes(Map<String, ChatService.Route> newRoutes) {
        return new Project(id, name, description, folder, tools, models, newRoutes, runners, parents, requests);
    }

    /** @return this project as part of other projects */
    public Project withParents(List<String> newParents) {
        return new Project(id, name, description, folder, tools, models, routes, runners, newParents, requests);
    }

    /** @return this project with other tools allowed, null for all */
    public Project withTools(List<String> newTools) {
        return new Project(id, name, description, folder, newTools, models, routes, runners, parents, requests);
    }

    /** @return this project with other models allowed, null for all */
    public Project withModels(List<String> newModels) {
        return new Project(id, name, description, folder, tools, newModels, routes, runners, parents, requests);
    }

    /**
     * Makes a project without runners.
     *
     * @param id its id
     * @param name its name
     * @param description what it is for
     * @param folder the folder its agents may read, or null
     * @param tools the tools allowed, or null for all
     * @param models the models allowed, or null for all
     * @param routes the routes by medium
     */
    public Project(String id, String name, String description, Path folder, List<String> tools, List<String> models,
            Map<String, ChatService.Route> routes) {
        this(id, name, description, folder, tools, models, routes, Map.of());
    }

    @Override
    public Map<String, Object> runner(String name) {
        return runners.get(name);
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
        m.put("runners", runners);
        m.put("parents", parents);
        m.put("requests", requests);
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
        Map<String, Map<String, Object>> runners = new LinkedHashMap<>();
        if (m.get("runners") != null) {
            for (Map.Entry<String, Object> e : Json.object(m.get("runners")).entrySet()) {
                if (e.getValue() != null) {
                    runners.put(e.getKey(), Json.object(e.getValue()));
                }
            }
        }
        return new Project(id, Json.string(m, "name", id), Json.string(m, "description", ""),
                folder.isBlank() ? null : Path.of(folder), strings(m.get("tools")), strings(m.get("models")), routes,
                runners, strings(m.get("parents")), Json.string(m, "requests", BY_USER));
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
