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
     * Finds a project.
     *
     * @param id its id
     * @return it, if there is one
     */
    public synchronized Optional<Project> get(String id) {
        return Optional.ofNullable(id == null ? null : projects.get(id));
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
            p = new Project(id, p.name(), p.description(), p.folder(), p.tools(), p.models(), p.routes(), p.runners());
        }
        if (p.folder() != null && !Files.isDirectory(p.folder())) {
            throw new IllegalArgumentException("there is no folder " + p.folder());
        }
        projects.put(p.id(), p);
        write();
        return p;
    }

    /**
     * Deletes a project; its agents stay, and belong to the default project from then on.
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
        boolean had = projects.remove(id) != null;
        if (id.equals(active)) {
            active = Project.DEFAULT;
        }
        write();
        return had;
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
