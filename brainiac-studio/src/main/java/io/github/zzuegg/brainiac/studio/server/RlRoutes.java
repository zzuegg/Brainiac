package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.train.RlRuns;
import java.util.List;
import java.util.Map;

/**
 * The reinforcement learning API.
 *
 * <ul>
 *   <li>{@code GET /api/rl}: the environments, the learners and the runs, newest first.
 *   <li>{@code POST /api/rl/start {env, algorithm, steps?, seed?, hidden?}}: starts a run.
 *   <li>{@code POST /api/rl/get {id}}: one run with its curve and the episode to watch.
 *   <li>{@code POST /api/rl/stop {id}}: stops a run after the episode it is in.
 *   <li>{@code POST /api/rl/delete {id}}: forgets a run that has ended.
 * </ul>
 */
final class RlRoutes {

    private static final int MAX_BODY = 1 << 16;

    private final RlRuns runs;

    RlRoutes(RlRuns runs) {
        this.runs = runs;
    }

    void register(StudioServer server) {
        server.get("/api/rl", ex -> Exchanges.json(ex, 200, runs.toJson()));
        server.post("/api/rl/start", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            List<Integer> hidden = body.get("hidden") == null ? null
                    : Json.array(body.get("hidden")).stream().map(v -> v instanceof Number n ? (Integer) n.intValue() : null).toList();
            Exchanges.json(ex, 200, runs.start(Json.string(body, "env", ""), Json.string(body, "algorithm", "ppo"),
                    Json.integer(body, "steps", 50_000), Json.integer(body, "seed", 1), hidden));
        });
        server.post("/api/rl/get", ex -> {
            var run = runs.get(Json.string(Exchanges.body(ex, MAX_BODY), "id", ""));
            if (run.isPresent()) {
                Exchanges.json(ex, 200, run.get());
            } else {
                Exchanges.error(ex, 404, "there is no such run");
            }
        });
        server.post("/api/rl/stop", ex -> Exchanges.json(ex, 200,
                Map.of("ok", runs.stop(Json.string(Exchanges.body(ex, MAX_BODY), "id", "")))));
        server.post("/api/rl/delete", ex -> Exchanges.json(ex, 200,
                Map.of("ok", runs.delete(Json.string(Exchanges.body(ex, MAX_BODY), "id", "")))));
    }
}
