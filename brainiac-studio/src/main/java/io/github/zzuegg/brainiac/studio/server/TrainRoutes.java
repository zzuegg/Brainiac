package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.train.TrainingRun;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The training API.
 *
 * <ul>
 *   <li>{@code GET /api/train/options}: the optimizers, losses, schedules and metrics the installed
 *       modules offer, found from their API like the builder's layers, and the devices.
 *   <li>{@code GET /api/train/data}: the built-in data sets and the files in the {@code datasets} folder.
 *   <li>{@code GET /api/train/models}: the models in {@code models}, with their input and output shapes.
 *   <li>{@code POST /api/train/check {spec}}: whether the model and the data fit together.
 *   <li>{@code POST /api/train/start {spec}}: starts a run, or queues it behind the one training.
 *   <li>{@code POST /api/train/stop {id}} and {@code POST /api/train/delete {id}}.
 *   <li>{@code GET /api/train/runs}: every run with its progress and loss curve.
 * </ul>
 */
final class TrainRoutes {

    private static final int MAX_BODY = 1 << 20;

    private final TrainingService training;

    TrainRoutes(TrainingService training) {
        this.training = training;
    }

    void register(StudioServer server) {
        server.get("/api/train/options", ex -> Exchanges.json(ex, 200, training.optionsJson()));
        server.get("/api/train/data", ex -> Exchanges.json(ex, 200, training.data().list()));
        server.get("/api/train/models", ex -> Exchanges.json(ex, 200, Map.of("models", training.models())));
        server.post("/api/train/check", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, training.check(Json.object(body.get("spec"))));
        });
        server.post("/api/train/start", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            TrainingRun run = training.start(Json.object(body.get("spec")));
            Exchanges.json(ex, 200, run.toJson(false));
        });
        server.post("/api/train/stop", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = training.stop(Json.string(body, "id", ""));
            Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
        });
        server.post("/api/train/delete", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            boolean ok = training.delete(Json.string(body, "id", ""));
            Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
        });
        server.get("/api/train/runs", ex -> {
            List<TrainingRun> runs = training.runs();
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("runs", runs.stream().map(r -> r.toJson(true)).toList());
            out.put("active", runs.stream().filter(r -> switch (r.state()) {
                case QUEUED, LOADING, RUNNING -> true;
                default -> false;
            }).count());
            Exchanges.json(ex, 200, out);
        });
    }
}
