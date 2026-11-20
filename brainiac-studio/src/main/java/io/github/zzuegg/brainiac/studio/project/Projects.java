package io.github.zzuegg.brainiac.studio.project;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The user's projects, kept in one file in the data folder. There is always a
 * {@linkplain Project#DEFAULT default} project, and one project is the active one: what the page
 * and agents use when they name none.
 */
public final class Projects {

    private static final System.Logger LOG = System.getLogger(Projects.class.getName());

    private final Path file;
    private final Map<String, Project> projects = new LinkedHashMap<>();
    private String active = Project.DEFAULT;

    /**
     * Reads the projects.
     *
     * @param file where they are kept, or null to keep them only while running
     */
    public Projects(Path file) {
        this.file = file;
        projects.put(Project.DEFAULT, Project.unlimited());
        if (file != null && Files.isRegularFile(file)) {
            try {
                Map<String, Object> m = Json.object(Json.read(file));
                for (Object o : Json.array(m.getOrDefault("projects", List.of()))) {
                    try {
                        Project p = Project.fromJson(Json.object(o));
                        projects.put(p.id(), p);
                    } catch (RuntimeException e) {
                        LOG.log(System.Logger.Level.WARNING, "skipping a project in " + file + ": " + e.getMessage());
                    }
                }
                String a = Json.string(m, "active", Project.DEFAULT);
                active = projects.containsKey(a) ? a : Project.DEFAULT;
            } catch (IOException | RuntimeException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot read " + file, e);
            }
        }
    }

    /**
     * Lists the projects, the default one first.
     *
     * @return them
     */
    public synchronized List<Project> list() {
        return List.copyOf(projects.values());
    }

    /**
     * Finds a project as it works: with the limits of every project it is part of. A
     * subproject can never do more than each of its parents allows, so what it may use is what
     * it allows itself and every parent allows too (see {@link #effective}). Use {@link #own} to
     * change a project's settings.
     *
     * @param id its id
     * @return it, if there is one
     */
    public synchronized Optional<Project> get(String id) {
        Project p = id == null ? null : projects.get(id);
        return Optional.ofNullable(p == null ? null : effective(p, new java.util.HashSet<>()));
    }

    /**
     * Finds a project with only its own settings, as it is kept.
     *
     * @param id its id
     * @return it, if there is one
     */
    public synchronized Optional<Project> own(String id) {
        return Optional.ofNullable(id == null ? null : projects.get(id));
    }

    // the project combined with what each parent allows; seen guards against a broken file
    private Project effective(Project p, java.util.Set<String> seen) {
        if (p.parents().isEmpty() || !seen.add(p.id())) {
            return p;
        }
        List<Project> ups = new ArrayList<>();
        for (String id : p.parents()) {
            Project up = projects.get(id);
            if (up != null) {
                ups.add(effective(up, seen));
            }
        }
        seen.remove(p.id());
        if (ups.isEmpty()) {
            return p;
        }
        List<String> tools = p.tools();
        List<String> models = p.models();
        java.nio.file.Path folder = p.folder();
        Map<String, Map<String, Object>> runners = new LinkedHashMap<>();
        for (Project up : ups) {
            tools = Limits.tools(tools, up.tools());
            models = Limits.models(models, up.models());
            if (folder == null) {
                folder = up.folder();
            }
            up.runners().forEach(runners::putIfAbsent);
        }
        runners.putAll(p.runners());
        // a route of its own holds only where every parent allows its model; others come from a parent
        Map<String, io.github.zzuegg.brainiac.studio.chat.ChatService.Route> routes = new LinkedHashMap<>();
        for (String medium : Project.MEDIA) {
            io.github.zzuegg.brainiac.studio.chat.ChatService.Route r = p.routes().get(medium);
            if (r != null && (r.model() == null || ups.stream().allMatch(up -> up.allows(r.model())))) {
                routes.put(medium, r);
                continue;
            }
            for (Project up : ups) {
                if (up.routes().get(medium) != null) {
                    routes.put(medium, up.routes().get(medium));
                    break;
                }
            }
        }
        return new Project(p.id(), p.name(), p.description(), folder, tools, models, routes, runners, p.parents(),
                p.requests());
    }

    /**
     * Lists the projects a project is part of, directly or further up, nearest first.
     *
     * @param id the project
     * @return their ids
     */
    public synchronized List<String> ancestors(String id) {
        List<String> out = new ArrayList<>();
        java.util.ArrayDeque<String> todo = new java.util.ArrayDeque<>();
        Project p = projects.get(id);
        if (p != null) {
            todo.addAll(p.parents());
        }
        while (!todo.isEmpty()) {
            String x = todo.poll();
            if (out.contains(x) || x.equals(id)) {
                continue;
            }
            out.add(x);
            Project q = projects.get(x);
            if (q != null) {
                todo.addAll(q.parents());
            }
        }
        return out;
    }

    /**
     * Lists the projects that are part of a project, directly or further down, nearest first.
     *
     * @param id the project
     * @return their ids
     */
    public synchronized List<String> descendants(String id) {
        List<String> out = new ArrayList<>();
        java.util.ArrayDeque<String> todo = new java.util.ArrayDeque<>(children(id));
        while (!todo.isEmpty()) {
            String x = todo.poll();
            if (out.contains(x) || x.equals(id)) {
                continue;
            }
            out.add(x);
            todo.addAll(children(x));
        }
        return out;
    }

    /**
     * Lists the projects directly part of a project.
     *
     * @param id the project
     * @return their ids
     */
    public synchronized List<String> children(String id) {
        return projects.values().stream().filter(p -> p.parents().contains(id)).map(Project::id).toList();
    }

    /**
     * Says whether a project is another or lies below it, which is where that project's agents may act.
     *
     * @param target the project acted on
     * @param from the project acting
     * @return whether it is the same project or one of its subprojects
     */
    public synchronized boolean within(String target, String from) {
        return target != null && (target.equals(from) || ancestors(target).contains(from));
    }

    /**
     * Returns a project, or the active one when none is named.
     *
     * @param id its id, or null or blank
     * @return the project
     * @throws IllegalArgumentException if there is no such project
     */
    public synchronized Project pick(String id) {
        if (id == null || id.isBlank()) {
            return projects.get(active);
        }
        Project p = projects.get(id);
        if (p == null) {
            throw new IllegalArgumentException("there is no project " + id);
        }
        return p;
    }

    /** @return the active project */
    public synchronized Project active() {
        return projects.get(active);
    }

    /**
     * Makes a project the active one.
     *
     * @param id its id
     * @throws IllegalArgumentException if there is no such project
     * @throws IOException if the choice cannot be kept
     */
    public synchronized void activate(String id) throws IOException {
        pick(id);
        active = id;
        write();
    }

    /**
     * Adds a project, or changes the one with the same id.
     *
     * @param p the project
     * @param create true to refuse an id in use (a new project with a taken name gets a number)
     * @return the project kept
     * @throws IOException if it cannot be kept
     */
    public synchronized Project save(Project p, boolean create) throws IOException {
        if (create) {
            String id = p.id();
            for (int n = 2; projects.containsKey(id); n++) {
                id = p.id() + "-" + n;
            }
            p = p.withId(id);
        }
        if (p.folder() != null && !Files.isDirectory(p.folder())) {
            throw new IllegalArgumentException("there is no folder " + p.folder());
        }
        for (String parent : p.parents()) {
            if (!projects.containsKey(parent)) {
                throw new IllegalArgumentException("there is no project " + parent + " to be part of");
            }
            if (parent.equals(p.id()) || ancestors(parent).contains(p.id())) {
                throw new IllegalArgumentException(name(parent) + " is already part of " + p.name()
                        + ", so " + p.name() + " cannot be part of it");
            }
        }
        projects.put(p.id(), p);
        write();
        return p;
    }

    /**
     * What deleting a project took away: the projects removed (it, and subprojects that were
     * part of nothing else) and the links cut from subprojects that are also part of others.
     *
     * @param removed the projects removed, as they were kept
     * @param unlinked {@code [child, parent]} pairs cut
     */
    public record Removal(List<Project> removed, List<List<String>> unlinked) {

        /** @return it as JSON, to keep in the trash */
        public Map<String, Object> toJson() {
            return Map.of("removed", removed.stream().map(Project::toJson).toList(), "unlinked", unlinked);
        }

        /**
         * Reads what {@link #toJson} wrote.
         *
         * @param m the JSON
         * @return the removal
         */
        public static Removal fromJson(Map<String, Object> m) {
            List<Project> removed = Json.array(m.getOrDefault("removed", List.of())).stream()
                    .map(o -> Project.fromJson(Json.object(o))).toList();
            List<List<String>> unlinked = Json.array(m.getOrDefault("unlinked", List.of())).stream()
                    .map(o -> Json.array(o).stream().map(String::valueOf).toList()).toList();
            return new Removal(removed, unlinked);
        }
    }

    /**
     * Deletes a project. Its subprojects that are part of no other project go with it; those
     * that are part of others only stop being part of it. Items stay, and belong to the default
     * project until their project comes back.
     *
     * @param id its id
     * @return what was taken away, for {@link #restore}
     * @throws IllegalArgumentException for the default project or a project that does not exist
     * @throws IOException if the change cannot be kept
     */
    public synchronized Removal remove(String id) throws IOException {
        if (Project.DEFAULT.equals(id)) {
            throw new IllegalArgumentException("the default project stays");
        }
        if (!projects.containsKey(id)) {
            throw new IllegalArgumentException("there is no project " + id);
        }
        java.util.LinkedHashSet<String> gone = new java.util.LinkedHashSet<>(List.of(id));
        for (boolean more = true; more; ) {
            more = false;
            for (Project p : projects.values()) {
                if (!gone.contains(p.id()) && !p.parents().isEmpty() && gone.containsAll(p.parents())) {
                    gone.add(p.id());
                    more = true;
                }
            }
        }
        List<Project> removed = gone.stream().map(projects::get).toList();
        List<List<String>> unlinked = new ArrayList<>();
        for (Project p : List.copyOf(projects.values())) {
            if (gone.contains(p.id())) {
                continue;
            }
            List<String> keep = new ArrayList<>();
            for (String parent : p.parents()) {
                if (gone.contains(parent)) {
                    unlinked.add(List.of(p.id(), parent));
                } else {
                    keep.add(parent);
                }
            }
            if (keep.size() != p.parents().size()) {
                projects.put(p.id(), p.withParents(keep));
            }
        }
        gone.forEach(projects::remove);
        if (gone.contains(active)) {
            active = Project.DEFAULT;
        }
        write();
        return new Removal(removed, unlinked);
    }

    /**
     * Puts back what {@link #remove} took away.
     *
     * @param r what was removed
     * @throws IllegalArgumentException if a project with one of the ids exists again
     * @throws IOException if the change cannot be kept
     */
    public synchronized void restore(Removal r) throws IOException {
        for (Project p : r.removed()) {
            if (projects.containsKey(p.id())) {
                throw new IllegalArgumentException("there is a project " + p.id() + " again");
            }
        }
        for (Project p : r.removed()) {
            // a parent that is gone for good meanwhile is left out
            projects.put(p.id(), p);
        }
        for (Project p : r.removed()) {
            projects.put(p.id(), p.withParents(p.parents().stream().filter(projects::containsKey).toList()));
        }
        for (List<String> link : r.unlinked()) {
            Project child = projects.get(link.get(0));
            if (child != null && projects.containsKey(link.get(1)) && !child.parents().contains(link.get(1))
                    && !Project.DEFAULT.equals(link.get(1)) && !descendants(child.id()).contains(link.get(1))) {
                List<String> ps = new ArrayList<>(child.parents());
                ps.add(link.get(1));
                projects.put(child.id(), child.withParents(ps));
            }
        }
        write();
    }

    /**
     * Deletes a project; see {@link #remove}.
     *
     * @param id its id
     * @return whether there was one
     * @throws IllegalArgumentException for the default project
     * @throws IOException if the change cannot be kept
     */
    public synchronized boolean delete(String id) throws IOException {
        if (Project.DEFAULT.equals(id)) {
            throw new IllegalArgumentException("the default project stays");
        }
        if (!projects.containsKey(id)) {
            return false;
        }
        remove(id);
        return true;
    }

    private String name(String id) {
        Project p = projects.get(id);
        return p == null ? id : p.name();
    }

    /**
     * Describes the projects for the page.
     *
     * @return {@code {active, projects}}
     */
    public synchronized Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("active", active);
        List<Object> ps = new ArrayList<>();
        for (Project p : projects.values()) {
            ps.add(p.toJson());
        }
        m.put("projects", ps);
        return m;
    }

    private void write() throws IOException {
        if (file == null) {
            return;
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.writeString(tmp, Json.write(toJson()));
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }
}
