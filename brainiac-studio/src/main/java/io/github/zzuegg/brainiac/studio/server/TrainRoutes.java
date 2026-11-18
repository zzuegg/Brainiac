package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.train.Hosts;
import io.github.zzuegg.brainiac.studio.train.TrainingRun;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import io.github.zzuegg.brainiac.studio.train.WorkerClient;
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
 *   <li>{@code GET /api/hosts}: the other machines runs can train on, and how each is; {@code POST
 *       /api/hosts {name, url, token}} adds one after checking it answers, and {@code {id, remove: true}}
 *       forgets one. Tokens are never sent back.
 *   <li>{@code POST /api/train/predict {model, values? | data?, index?}} runs a saved model on one sample.
 *   <li>{@code POST /api/hosts/secret {host, name, lend}} lends a host one of this studio's secrets
 *       ({@code huggingface}: its Hugging Face token), sealed and held in the host's memory only,
 *       or takes it back.
 *   <li>{@code POST /api/hosts/download {host, repo, option}} has a host download a Hugging Face model
 *       itself, so it can chat with it; {@code POST /api/hosts/downloads {host}} lists that host's
 *       downloads and {@code POST /api/hosts/cancel {host, id}} stops one.
 * </ul>
 */
final class TrainRoutes {

    private static final int MAX_BODY = 1 << 20;

    private final TrainingService training;

    TrainRoutes(TrainingService training) {
        this.training = training;
    }

    private interface HostCall {
        Object answer(WorkerClient client, Map<String, Object> body) throws java.io.IOException;
    }

    // a call passed on to one worker host: 404 for a host that is not known, 400 with the worker's own words
    private void onHost(com.sun.net.httpserver.HttpExchange ex, HostCall call) throws java.io.IOException {
        Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
        Hosts.Host host = training.hosts().find(Json.string(body, "host", "")).orElse(null);
        if (host == null) {
            Exchanges.error(ex, 404, "no such host");
            return;
        }
        Object out;
        try {
            out = call.answer(training.hosts().client(host), body);
        } catch (java.io.IOException e) {
            Exchanges.error(ex, 400, host.name() + ": " + e.getMessage());
            return;
        }
        Exchanges.json(ex, 200, out);
    }

    void register(StudioServer server) {
        server.get("/api/train/options", ex -> Exchanges.json(ex, 200, training.optionsJson()));
        server.get("/api/train/data", ex -> Exchanges.json(ex, 200, training.data().list()));
        server.get("/api/train/models", ex -> Exchanges.json(ex, 200, Map.of("models", training.models())));
        server.post("/api/train/check", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            Exchanges.json(ex, 200, training.check(Json.object(body.get("spec"))));
        });
        server.post("/api/train/predict", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            try {
                Exchanges.json(ex, 200, training.predict(body));
            } catch (java.io.IOException e) {
                Exchanges.error(ex, 400, e.getMessage());
            }
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
        server.get("/api/hosts", ex -> Exchanges.json(ex, 200, Map.of("hosts", training.hosts().toJson())));
        server.post("/api/hosts", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            if (Boolean.TRUE.equals(body.get("remove"))) {
                training.hosts().takeBack(Json.string(body, "id", ""));
                boolean ok = training.hosts().remove(Json.string(body, "id", ""));
                Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
                return;
            }
            try {
                Hosts.Host h = training.hosts().add(Json.string(body, "name", ""), Json.string(body, "url", ""),
                        Json.string(body, "token", ""));
                Exchanges.json(ex, 200, Map.of("id", h.id(), "name", h.name(), "url", h.url()));
            } catch (java.io.IOException e) {
                Exchanges.error(ex, 400, e.getMessage());
            }
        });
        server.post("/api/hosts/secret", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            try {
                training.hosts().lend(Json.string(body, "host", ""), Json.string(body, "name", "huggingface"),
                        !Boolean.FALSE.equals(body.get("lend")));
                Exchanges.json(ex, 200, Map.of("hosts", training.hosts().toJson()));
            } catch (java.io.IOException e) {
                // no such host, no key to lend here, or a worker that does not answer or is too old
                Exchanges.error(ex, 400, e.getMessage());
            }
        });
        server.post("/api/hosts/download", ex -> onHost(ex, (client, body) -> {
            // a host that restarted lost the token it was lent: it gets it again before a gated download
            training.hosts().find(Json.string(body, "host", "")).ifPresent(training.hosts()::relend);
            return client.hubDownload(Json.string(body, "repo", "").strip(), Json.string(body, "option", ""));
        }));
        server.post("/api/hosts/downloads", ex -> onHost(ex, (client, body) -> Map.of("downloads", client.hubDownloads())));
        server.post("/api/hosts/cancel", ex -> onHost(ex, (client, body) ->
                Map.of("ok", client.hubCancel(Json.string(body, "id", "")))));
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
