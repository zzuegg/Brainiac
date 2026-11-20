package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.GpuResidency;
import io.github.zzuegg.brainiac.studio.project.Project;
import io.github.zzuegg.brainiac.studio.project.Projects;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The projects API and the GPU scheduling API.
 *
 * <ul>
 *   <li>{@code GET /api/projects}: {@code {active, projects}}.
 *   <li>{@code POST /api/projects {project, create?}}: adds a project ({@code create: true}) or
 *       changes one: {@code {id, name, description, folder, tools, models, routes: {medium: {model,
 *       policy, fallback}}}}; {@code tools} and {@code models} null allow everything.
 *   <li>{@code POST /api/projects/activate {id}} and {@code POST /api/projects/delete {id}}.
 *   <li>{@code GET /api/gpu}: what is loaded on this PC, what waits for room, and the policies.
 *   <li>{@code POST /api/gpu {policies?: {kind: policy}, pinned?: [kind], idleMinutes?}} changes the
 *       studio's own choices; {@code POST /api/gpu/unload {kind}} unloads a kind's model, or every
 *       model with {@code all}.
 * </ul>
 *
 * <p>Every API call belongs to a project: the one named in the {@value #HEADER} header, or else
 * the active one. Its routes and limits hold for that call. A {@value #POLICY_HEADER} header
 * ({@code swap}, {@code queue}, {@code remote} or {@code spill}) overrides the scheduling policy
 * for that call, which is how the page answers a {@code noRoom} error.
 */
final class ProjectRoutes {

    /** The request header naming the project a call belongs to. */
    static final String HEADER = "X-Studio-Project";

    /** The request header that chooses what to do, for this call, when a model does not fit in GPU memory. */
    static final String POLICY_HEADER = "X-Studio-Policy";

    private static final int MAX_BODY = 1 << 20;

    private final Projects projects;
    private final ChatService chat;

    ProjectRoutes(Projects projects, ChatService chat) {
        this.projects = projects;
        this.chat = chat;
    }

    void register(StudioServer server) {
        server.get("/api/projects", ex -> Exchanges.json(ex, 200, projects.toJson()));
        server.post("/api/projects", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Object p = body.get("project");
            if (p == null) {
                throw new Exchanges.BadRequest("send the project as {\"project\": {...}}");
            }
            Map<String, Object> sent = Json.object(p);
            Project project = Project.fromJson(sent);
            // a page that does not know about subprojects keeps the links it did not send
            Project before = projects.own(project.id()).orElse(null);
            if (before != null && !Boolean.TRUE.equals(body.get("create"))) {
                if (!sent.containsKey("parents")) {
                    project = project.withParents(before.parents());
                }
                if (!sent.containsKey("requests")) {
                    project = new Project(project.id(), project.name(), project.description(), project.folder(),
                            project.tools(), project.models(), project.routes(), project.runners(), project.parents(),
                            before.requests());
                }
            }
            // a runner that cannot be rented is refused now, not when a task needs it
            project.runners().values().forEach(io.github.zzuegg.brainiac.studio.train.VastRunners.Runner::fromJson);
            Project saved = projects.save(project, Boolean.TRUE.equals(body.get("create")));
            Map<String, Object> out = new LinkedHashMap<>(projects.toJson());
            out.put("saved", saved.id());
            Exchanges.json(ex, 200, out);
        });
        server.post("/api/projects/activate", ex -> {
            projects.activate(Json.string(Exchanges.body(ex, MAX_BODY), "id", ""));
            Exchanges.json(ex, 200, projects.toJson());
        });
        server.post("/api/projects/delete", ex -> {
            projects.delete(Json.string(Exchanges.body(ex, MAX_BODY), "id", ""));
            Exchanges.json(ex, 200, projects.toJson());
        });
        server.get("/api/gpu", ex -> Exchanges.json(ex, 200, gpu()));
        server.post("/api/gpu", ex -> {
            residency().change(Exchanges.body(ex, MAX_BODY));
            Exchanges.json(ex, 200, gpu());
        });
        server.post("/api/gpu/unload", ex -> {
            String kind = Json.string(Exchanges.body(ex, MAX_BODY), "kind", "all");
            Map<String, Object> out = new LinkedHashMap<>(gpu());
            out.put("unloaded", residency().unload(kind));
            out.putAll(gpu());
            Exchanges.json(ex, 200, out);
        });
    }

    private GpuResidency residency() {
        return chat.localDevice().residency();
    }

    private Map<String, Object> gpu() {
        io.github.zzuegg.brainiac.tensor.Device d;
        try {
            d = chat.localDevice().device();
        } catch (RuntimeException | LinkageError e) {
            d = null;
        }
        return residency().toJson(d);
    }
}
