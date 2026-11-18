package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.Format;
import io.github.zzuegg.brainiac.studio.agent.AgentRuns;
import io.github.zzuegg.brainiac.studio.agent.StudioTools;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The agents API, and the studio's tools for programs that bring their own agent.
 *
 * <ul>
 *   <li>{@code GET /api/tools}: every tool with its JSON schema, and what it does if it acts.
 *   <li>{@code POST /api/tools/call {name, arguments}}: runs a tool; {@code {result}}, which starts
 *       with {@code error:} if it failed. Callers hold the studio's token, so tools that act run
 *       without asking.
 *   <li>{@code GET /api/agents}: the runs, newest first, the tools and the limits.
 *   <li>{@code POST /api/agents/get {id}}: one run with every step.
 *   <li>{@code POST /api/agents/start {goal, model?, mode?, maxSteps?, tools?, format?}}: starts an
 *       agent; mode {@code ask} (the default) waits for the user before a tool that acts. With a
 *       {@code format} of {@code {type: "json"}} or {@code {type: "schema", schema}} its final
 *       answer is JSON, written in a turn of its own.
 *   <li>{@code POST /api/agents/answer {id, allow, note?}}: answers the tool call a run waits on.
 *   <li>{@code POST /api/agents/reply {id, text}}: lets a run that ended go on with another message.
 *   <li>{@code POST /api/agents/stop {id}} and {@code POST /api/agents/delete {id}}.
 * </ul>
 */
final class AgentRoutes {

    private static final int MAX_BODY = 1 << 20;

    private final AgentRuns agents;

    AgentRoutes(AgentRuns agents) {
        this.agents = agents;
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
            Exchanges.json(ex, 200, agents.start(Json.string(body, "goal", ""), Json.string(body, "model", "auto"),
                    Json.string(body, "mode", "ask"), (int) Json.integer(body, "maxSteps", 12), tools, Format.of(body)));
        });
        server.post("/api/agents/answer", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = agents.answer(Json.string(body, "id", ""), Boolean.TRUE.equals(body.get("allow")),
                    Json.string(body, "note", ""));
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
        server.post("/api/agents/delete", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = agents.delete(Json.string(body, "id", ""));
            Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
        });
    }

    private List<Map<String, Object>> tools() {
        return agents.tools().values().stream().map(StudioTools.Entry::toJson).toList();
    }
}
