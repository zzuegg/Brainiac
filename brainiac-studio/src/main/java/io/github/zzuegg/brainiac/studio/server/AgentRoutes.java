package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.agent.AgentRuns;
import io.github.zzuegg.brainiac.studio.agent.AgentTools;
import io.github.zzuegg.brainiac.studio.agent.Risk;
import io.github.zzuegg.brainiac.studio.agent.Role;
import io.github.zzuegg.brainiac.studio.agent.Rules;
import io.github.zzuegg.brainiac.studio.agent.Services;
import io.github.zzuegg.brainiac.studio.agent.StudioTools;
import io.github.zzuegg.brainiac.studio.chat.Format;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The agents API, and the studio's tools for programs that bring their own agent.
 *
 * <ul>
 *   <li>{@code GET /api/tools}: every tool with its JSON schema, its risk tier, and what it does if it acts.
 *   <li>{@code POST /api/tools/call {name, arguments}}: runs a tool; {@code {result}}, which starts
 *       with {@code error:} if it failed. Tools that run code or leave the PC are refused here.
 *   <li>{@code GET /api/agents}: the runs, newest first, the tools and the limits.
 *   <li>{@code POST /api/agents/get {id}}: one run with every step.
 *   <li>{@code POST /api/agents/start {goal, role?, model?, mode?, maxSteps?, tools?, format?}}:
 *       starts an agent; what is left out comes from the role.
 *   <li>{@code POST /api/agents/answer {id, allow, note?, mode?}}: answers what a run waits on.
 *   <li>{@code POST /api/agents/reply {id, text}}: lets a run that ended go on with another message.
 *   <li>{@code POST /api/agents/stop {id}}, {@code /stop-all}, {@code /delete {id}}.
 *   <li>{@code POST /api/agents/changes {id}} and {@code /undo {id}}: the files a run changed, and
 *       putting them back.
 *   <li>{@code GET /api/roles}, {@code POST /api/roles/save {role}}, {@code /delete {id}},
 *       {@code /resolve {id}}.
 *   <li>{@code GET /api/rules}, {@code POST /api/rules/save {rules}}, {@code /reset}.
 *   <li>{@code GET /api/services}, {@code POST /api/services/save {section, values, secret?}}.
 *   <li>{@code POST /api/agents/notes {project}} and {@code /notes/save {project, text}}: what
 *       agents remember per project.
 * </ul>
 */
final class AgentRoutes {

    private static final int MAX_BODY = 1 << 20;

    private final AgentRuns agents;
    private final Services services;
    private final Path memory;

    AgentRoutes(AgentRuns agents, Services services, Path memory) {
        this.agents = agents;
        this.services = services;
        this.memory = memory;
    }

    void register(StudioServer server) {
        server.get("/api/tools", ex -> Exchanges.json(ex, 200, Map.of("tools", tools())));
        server.post("/api/tools/call", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Object args = body.get("arguments");
            String result = agents.call(Json.string(body, "name", ""), args == null ? Map.of() : Json.object(args));
            Exchanges.json(ex, 200, Map.of("result", result));
        });
        server.get("/api/agents", ex -> {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("runs", agents.list());
            out.put("tools", tools());
            out.put("roles", agents.roles().list().stream().map(Role::toJson).toList());
            out.put("maxActive", AgentRuns.MAX_ACTIVE);
            out.put("maxSteps", AgentRuns.MAX_STEPS);
            Exchanges.json(ex, 200, out);
        });
        server.post("/api/agents/get", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            var run = agents.get(Json.string(body, "id", ""));
            if (run.isPresent()) {
                Exchanges.json(ex, 200, run.get());
            } else {
                Exchanges.error(ex, 404, "there is no such agent");
            }
        });
        server.post("/api/agents/start", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            List<String> tools = body.get("tools") == null ? null
                    : Json.array(body.get("tools")).stream().map(String::valueOf).toList();
            String model = Json.string(body, "model", "");
            String mode = Json.string(body, "mode", "");
            Exchanges.json(ex, 200, agents.start(new AgentRuns.Spec(Json.string(body, "goal", ""),
                    blank(Json.string(body, "role", "")), blank(model), blank(mode),
                    (int) Json.integer(body, "maxSteps", 0), tools, Format.of(body))));
        });
        server.post("/api/agents/answer", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = agents.answer(Json.string(body, "id", ""), Boolean.TRUE.equals(body.get("allow")),
                    Json.string(body, "note", ""), blank(Json.string(body, "mode", "")));
            Exchanges.json(ex, ok ? 200 : 409, Map.of("ok", ok));
        });
        server.post("/api/agents/reply", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            var run = agents.reply(Json.string(body, "id", ""), Json.string(body, "text", ""));
            if (run.isPresent()) {
                Exchanges.json(ex, 200, run.get());
            } else {
                Exchanges.error(ex, 404, "there is no such agent");
            }
        });
        server.post("/api/agents/stop", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = agents.stop(Json.string(body, "id", ""));
            Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
        });
        server.post("/api/agents/stop-all", ex -> Exchanges.json(ex, 200, Map.of("stopped", agents.stopAll())));
        server.post("/api/agents/delete", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = agents.delete(Json.string(body, "id", ""));
            Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
        });
        server.post("/api/agents/changes", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("changes", agents.changes(Json.string(body, "id", ""))));
        });
        server.post("/api/agents/undo", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("restored", agents.undo(Json.string(body, "id", ""))));
        });
        server.post("/api/agents/notes", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, Map.of("text", AgentTools.notesOf(memory, project(body))));
        });
        server.post("/api/agents/notes/save", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            AgentTools.saveNotes(memory, project(body), Json.string(body, "text", ""));
            Exchanges.json(ex, 200, Map.of("ok", true));
        });

        // roles
        server.get("/api/roles", ex -> Exchanges.json(ex, 200, roles()));
        server.post("/api/roles/save", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Object r = body.get("role");
            Role saved = agents.roles().save(Role.fromJson(Json.object(r == null ? body : r), false));
            agents.roles().resolve(saved.id(), agents.tools().keySet());   // a role that cannot be used is reported now
            Map<String, Object> out = roles();
            out.put("saved", saved.id());
            Exchanges.json(ex, 200, out);
        });
        server.post("/api/roles/delete", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = agents.roles().delete(Json.string(body, "id", ""));
            Exchanges.json(ex, ok ? 200 : 404, roles());
        });
        server.post("/api/roles/resolve", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, agents.roles().resolve(blank(Json.string(body, "id", "")), agents.tools().keySet())
                    .toJson());
        });

        // rules
        server.get("/api/rules", ex -> Exchanges.json(ex, 200, rules()));
        server.post("/api/rules/save", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            List<Rules.Rule> list = new ArrayList<>();
            for (Object o : Json.array(body.getOrDefault("rules", List.of()))) {
                list.add(Rules.Rule.fromJson(Json.object(o)));
            }
            agents.rules().save(list);
            Exchanges.json(ex, 200, rules());
        });
        server.post("/api/rules/reset", ex -> {
            agents.rules().save(Rules.shipped());
            Exchanges.json(ex, 200, rules());
        });

        // outside services
        server.get("/api/services", ex -> Exchanges.json(ex, 200, services.toJson()));
        server.post("/api/services/save", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Object values = body.get("values");
            services.set(Json.string(body, "section", ""), values == null ? Map.of() : Json.object(values),
                    Json.string(body, "secret", null));
            Exchanges.json(ex, 200, services.toJson());
        });
    }

    private Map<String, Object> roles() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("roles", agents.roles().list().stream().map(Role::toJson).toList());
        out.put("tiers", Arrays.stream(Risk.values()).map(r -> Map.of("key", r.key(), "what", r.what(),
                "ask", r.byDefault("ask").key(), "auto", r.byDefault("auto").key())).toList());
        out.put("tools", tools());
        return out;
    }

    private Map<String, Object> rules() {
        Map<String, Object> out = new LinkedHashMap<>(agents.rules().toJson());
        out.put("shipped", Rules.shipped().stream().map(Rules.Rule::toJson).toList());
        out.put("tools", agents.tools().keySet());
        out.put("tiers", Arrays.stream(Risk.values()).map(Risk::key).toList());
        return out;
    }

    private static String project(Map<String, Object> body) {
        String p = Json.string(body, "project", "");
        if (!p.matches("[a-z0-9][a-z0-9-]{0,39}")) {
            throw new IllegalArgumentException("say which project");
        }
        return p;
    }

    private static String blank(String s) {
        return s == null || s.isBlank() ? null : s;
    }

    private List<Map<String, Object>> tools() {
        return agents.tools().values().stream().map(StudioTools.Entry::toJson).toList();
    }
}
