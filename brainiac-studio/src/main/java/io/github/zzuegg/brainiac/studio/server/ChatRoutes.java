package io.github.zzuegg.brainiac.studio.server;

import com.sun.net.httpserver.HttpExchange;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatListener;
import io.github.zzuegg.brainiac.studio.chat.ChatMessage;
import io.github.zzuegg.brainiac.studio.chat.ChatResult;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.LocalImages;
import io.github.zzuegg.brainiac.studio.chat.LocalSpeech;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The chat API.
 *
 * <ul>
 *   <li>{@code GET /api/models} lists the providers and their models ({@code ?refresh=1} asks
 *       remote providers again).
 *   <li>{@code POST /api/keys} with {@code {"provider": "openai", "label": "work", "key": "..."}}
 *       saves a key under a name; {@code "remove": true} removes that name instead, and
 *       {@code "first": true} makes it the key tried first. Keys are never sent back, only their
 *       names and last four characters.
 *   <li>{@code POST /api/chat} with {@code {"model": "auto", "system": "...", "messages":
 *       [{"role": "user", "text": "...", "images": [{"type": "image/png", "data": "base64"}]}], "maxTokens": 4096, "temperature": 0.7, "topP": 0.9, "topK": 40, "repeatPenalty": 1.1, "key": "work"}} (the key's name is
 *       optional) answers as a server-sent event
 *       stream: {@code route} (the model chosen), then {@code text} and {@code thinking} pieces,
 *       then {@code done} or {@code error}.
 *   <li>{@code GET /api/serve} says whether one of this PC's chat models is served to other
 *       programs with the OpenAI API; {@code POST /api/serve} with {@code {"model": "local/qwen",
 *       "port": 8080, "network": false}} starts that, and with {@code {"stop": true}} stops it.
 *   <li>{@code GET /api/quantize} gives the sizes a smaller copy of a chat model can have and the
 *       copy being made; {@code POST /api/quantize} with {@code {"model": "local/qwen", "size":
 *       "Q4_K_M"}} starts one: a quantized GGUF file next to the original.
 *   <li>{@code GET /api/finetune} tells how teaching a chat model with LoRA is going; {@code POST
 *       /api/finetune} with {@code {"model": "local/qwen", "name": "support", "data": "<JSON
 *       lines>", "rank": 8, "alpha": 16, "epochs": 3, "learningRate": 0.0002, "length": 512,
 *       "rows": 1}} starts it, and with {@code {"stop": true}} stops it after the current step.
 *   <li>{@code GET /api/evaluate} lists the measurements of this PC's chat models, newest first;
 *       {@code POST /api/evaluate} with {@code {"model": "local/qwen", "task": "arc.jsonl", "data":
 *       "<JSON lines>", "limit": 100, "shots": 0}} scores a model on a benchmark file, with {@code
 *       {"model": "local/qwen", "speed": true}} times its reading and writing, with {@code {"stop":
 *       true}} stops the one under way and with {@code {"remove": "<id>"}} forgets one.
 *   <li>{@code POST /api/images} with {@code {"model": "auto", "prompt": "...", "aspect": "16:9",
 *       "count": 2}} draws pictures; {@code POST /api/speech} with {@code {"model": "auto", "text":
 *       "...", "voice": "coral", "speed": 1, "instructions": "calm", "format": "mp3"}} speaks.
 *       Both answer with the model used and each file, base64 encoded, and where it was saved.
 *   <li>{@code POST /api/images/draw} with {@code {"model": "local/sdxl", "prompt": "...",
 *       "negativePrompt": "", "aspect": "1:1", "count": 2, "steps": 30, "guidance": 6, "seed": 42,
 *       "sampler": "DPM_PP_2M_KARRAS"}} draws on this PC with a Stable Diffusion model, as a
 *       server-sent event stream: {@code route}, {@code step} ({@code {step, total}}) with a rough
 *       {@code preview} (a small PNG) every other step, one {@code image} per picture
 *       ({@code {mime, data, file, seed}}), then {@code done} or {@code error}. Closing the stream
 *       stops the drawing. Leave {@code seed} out for a random one. {@code POST
 *       /api/images/options} with {@code {"model": "local/sdxl"}} gives the samplers and the
 *       model's usual steps and guidance.
 *   <li>{@code POST /api/speech/voice} with {@code {"model": "local/f5-tts", "name": "Me", "audio":
 *       "<base64 WAV>", "transcript": "..."}} keeps a voice made from a recording, for a model that
 *       clones voices; leave {@code transcript} empty to have a Whisper model write it down. It
 *       answers {@code {voice, transcript}}. {@code "remove": true} with the model and name forgets
 *       one.
 *   <li>{@code POST /api/transcribe} with {@code {"model": "auto", "audio": "<base64 WAV>",
 *       "language": "de", "translate": false, "timestamps": true, "prompt": ""}} writes down what is
 *       said with a Whisper model on this PC, as a server-sent event stream: {@code route}, one
 *       {@code segment} ({@code {start, end, text}}) at a time, then {@code done} ({@code {text,
 *       language, audioSeconds, seconds}}) or {@code error}. Closing the stream stops it. {@code POST
 *       /api/transcribe/options} with {@code {"model": "local/whisper-base"}} gives its languages
 *       and whether it translates.
 *   <li>{@code POST /api/providers} with {@code {"provider": "huggingface", "auto": true, "models":
 *       ["openai/gpt-oss-120b"], "url": "", "options": {"billTo": "my-org"}}} changes how a provider is
 *       used: whether Auto may pick it, which models to show ({@code null} for all), its address
 *       ({@code ""} for its own) and its options. Fields left out stay as they are.
 *   <li>{@code GET /api/chats} lists the saved conversations, newest first; {@code POST /api/chats}
 *       with a conversation ({@code id}, {@code title}, {@code messages}, {@code status} open or
 *       finished) saves it, and with {@code {"id": "...", "delete": true}} deletes it.
 *   <li>{@code GET /api/state} returns what the page remembers (tabs, layout, chosen models);
 *       {@code POST /api/state} with {@code {"name": value}} changes those values, null removes one.
 *   <li>{@code POST /api/local} with {@code {"path": "D:/models/qwen.gguf"}} adds a place to find
 *       models on this PC (a {@code .gguf} file, a Hugging Face model folder or a folder holding them),
 *       and with {@code "remove": true} forgets one; {@code POST /api/local/browse} with {@code {"path":
 *       "D:/models"}} lists that folder's subfolders and models to pick from (no path: where to start).
 *   <li>{@code POST /api/hub/search} with {@code {"query": "qwen2.5"}} ({@code "kind": "speech"} for
 *       Whisper and VITS speech models) finds text generation
 *       models on Hugging Face that the core runs; {@code POST /api/hub/files} with {@code {"repo": "..."}}
 *       lists what can be downloaded from one and whether the studio runs it; {@code POST
 *       /api/hub/download} with {@code {"repo": "...", "option": "folder" or a GGUF file}} starts a
 *       download into {@code downloads} in the data folder; {@code GET /api/hub/downloads} shows them;
 *       {@code POST /api/hub/cancel} with {@code {"id": "1"}} stops one, and {@code POST
 *       /api/hub/delete} with {@code {"name": "..."}} deletes a downloaded model.
 *   <li>{@code GET /api/vast}: the Vast.ai key (its last four characters), the account's balance
 *       and the machines rented; {@code POST /api/vast/key} with {@code {"key": "..."}} saves the
 *       key ({@code {"remove": true}} removes it); {@code POST /api/vast/offers} with {@code {"gpu":
 *       "4090", "minGpuRam": 24, "gpus": 1, "maxPrice": 0.5}} lists machines to rent, cheapest first.
 *       Renting is in {@link VastRoutes}.
 *   <li>{@code POST /api/hub/token} with {@code {"token": "hf_..."}} saves the Hugging Face token to
 *       {@code ~/.brainiac/hf_token}, readable only by this account, where the core's downloader
 *       finds it too; {@code {"remove": true}} deletes that file. Only its last four characters come back.
 *   <li>{@code GET /api/usage} sums every call by provider, key, model and day;
 *       {@code GET /api/balances} asks the providers that report one what is left on each key.
 * </ul>
 */
final class ChatRoutes {

    private static final System.Logger LOG = System.getLogger(ChatRoutes.class.getName());
    private static final int MAX_BODY = 32 << 20;   // a conversation with its pictures
    private static final int DEFAULT_MAX_TOKENS = 8192;

    private final ChatService chat;

    ChatRoutes(ChatService chat) {
        this.chat = chat;
    }

    void register(StudioServer server) {
        server.get("/api/models", ex -> Exchanges.json(ex, 200,
                chat.describe("refresh=1".equals(ex.getRequestURI().getQuery()))));
        server.get("/api/local/device", ex -> Exchanges.json(ex, 200, localDevice()));
        server.post("/api/local/device", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 4096);
            if (body.get("memory") instanceof Number mb) {
                chat.localDevice().limitMemory(mb.intValue());
            }
            if (body.containsKey("device") || !body.containsKey("memory")) {
                chat.localDevice().choose(Json.string(body, "device", ""));
            }
            Exchanges.json(ex, 200, localDevice());
        });
        server.get("/api/serve", ex -> Exchanges.json(ex, 200, chat.serving().state()));
        server.post("/api/serve", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 4096);
            if (Boolean.TRUE.equals(body.get("stop"))) {
                chat.serving().stop();
                Exchanges.json(ex, 200, chat.serving().state());
                return;
            }
            String model = Json.string(body, "model", "");
            if (model.startsWith("local/")) {
                model = model.substring(6);
            }
            try {
                Exchanges.json(ex, 200, chat.serving().start(model, (int) Json.integer(body, "port", 8080),
                        Boolean.TRUE.equals(body.get("network"))));
            } catch (IOException e) {
                // no such model, it does not load, or the port is taken: the caller's to fix
                throw new Exchanges.BadRequest(e.getMessage());
            }
        });
        server.get("/api/quantize", ex -> Exchanges.json(ex, 200, chat.quantizer().state()));
        server.post("/api/quantize", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 4096);
            String model = Json.string(body, "model", "");
            if (model.startsWith("local/")) {
                model = model.substring(6);
            }
            try {
                Exchanges.json(ex, 200, chat.quantizer().start(model, Json.string(body, "size", "Q4_K_M")));
            } catch (IOException | IllegalArgumentException e) {
                // no such model or size, the copy is there already, or one is being made: the caller's to fix
                throw new Exchanges.BadRequest(e.getMessage());
            }
        });
        server.get("/api/finetune", ex -> Exchanges.json(ex, 200, chat.fineTune().state()));
        server.post("/api/finetune", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            if (Boolean.TRUE.equals(body.get("stop"))) {
                chat.fineTune().stop();
                Exchanges.json(ex, 200, chat.fineTune().state());
                return;
            }
            String model = Json.string(body, "model", "");
            if (model.startsWith("local/")) {
                model = model.substring(6);
            }
            try {
                int rank = (int) Json.integer(body, "rank", 8);
                Exchanges.json(ex, 200, chat.fineTune().start(model, Json.string(body, "name", ""),
                        Json.string(body, "data", ""),
                        new io.github.zzuegg.brainiac.studio.chat.LocalFineTune.Settings(rank,
                                (float) Json.number(body, "alpha", 2.0 * rank), (int) Json.integer(body, "epochs", 3),
                                (float) Json.number(body, "learningRate", 2e-4), (int) Json.integer(body, "length", 512),
                                (int) Json.integer(body, "rows", 1))));
            } catch (IOException | IllegalArgumentException e) {
                // no such model, a bad setting or name, or another run: the caller's to fix
                throw new Exchanges.BadRequest(e.getMessage());
            }
        });
        server.get("/api/evaluate", ex -> Exchanges.json(ex, 200, chat.evaluation().state()));
        server.post("/api/evaluate", ex -> {
            Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
            if (Boolean.TRUE.equals(body.get("stop"))) {
                chat.evaluation().stop();
                Exchanges.json(ex, 200, chat.evaluation().state());
                return;
            }
            if (body.get("remove") instanceof String id) {
                chat.evaluation().remove(id);
                Exchanges.json(ex, 200, chat.evaluation().state());
                return;
            }
            String model = Json.string(body, "model", "");
            if (model.startsWith("local/")) {
                model = model.substring(6);
            }
            try {
                Exchanges.json(ex, 200, Boolean.TRUE.equals(body.get("speed")) ? chat.evaluation().startSpeed(model)
                        : chat.evaluation().start(model, Json.string(body, "task", ""), Json.string(body, "data", ""),
                                (int) Json.integer(body, "limit", 100), (int) Json.integer(body, "shots", 0)));
            } catch (IOException | IllegalArgumentException e) {
                // no such model, a file the studio cannot read, or another measurement: the caller's to fix
                throw new Exchanges.BadRequest(e.getMessage());
            }
        });
        server.post("/api/keys", this::keys);
        server.post("/api/providers", this::providers);
        server.get("/api/chats", ex -> Exchanges.json(ex, 200, Map.of("chats", chat.conversations().list())));
        server.post("/api/chats", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 32 << 20);
            if (Boolean.TRUE.equals(body.get("delete"))) {
                chat.conversations().delete(body.get("id"));
            } else {
                chat.conversations().save(body);
            }
            Exchanges.json(ex, 200, Map.of("ok", true));
        });
        server.get("/api/state", ex -> Exchanges.json(ex, 200, chat.page().values()));
        server.post("/api/state", ex -> {
            chat.page().merge(Exchanges.body(ex, 2 << 20));
            Exchanges.json(ex, 200, Map.of("ok", true));
        });
        server.post("/api/chat", this::chat);
        server.post("/api/images", this::images);
        server.post("/api/images/draw", this::draw);
        server.post("/api/images/options", ex -> Exchanges.json(ex, 200,
                chat.drawOptions(Json.string(Exchanges.body(ex, 8192), "model", ""))));
        server.post("/api/transcribe", this::transcribe);
        server.post("/api/speech/voice", this::voice);
        server.post("/api/transcribe/options", ex -> Exchanges.json(ex, 200,
                chat.transcribeOptions(Json.string(Exchanges.body(ex, 8192), "model", ""))));
        server.get("/api/usage", ex -> Exchanges.json(ex, 200, chat.usage().summary(java.time.ZoneId.systemDefault())));
        server.get("/api/balances", ex -> {
            try {
                Exchanges.json(ex, 200, chat.balances());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Exchanges.error(ex, 503, "stopped");
            }
        });
        server.post("/api/speech", this::speech);
        server.post("/api/local", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 64 << 10);
            String path = Json.string(body, "path", "").strip();
            if (path.isEmpty()) {
                Exchanges.error(ex, 400, "say which file or folder");
                return;
            }
            try {
                if (Boolean.TRUE.equals(body.get("remove"))) {
                    boolean ok = chat.removeLocal(Path.of(path));
                    Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
                } else {
                    Exchanges.json(ex, 200, Map.of("models", chat.addLocal(Path.of(path))));
                }
            } catch (java.nio.file.InvalidPathException e) {
                Exchanges.error(ex, 400, "not a path: " + path);
            } catch (IOException e) {
                Exchanges.error(ex, 400, e.getMessage());
            }
        });
        server.post("/api/hub/search", ex -> hub(ex, body -> chat.hub().search(Json.string(body, "query", ""), Json.string(body, "kind", "text"))));
        server.post("/api/hub/files", ex -> hub(ex, body -> chat.hub().files(Json.string(body, "repo", "").strip())));
        server.post("/api/hub/download", ex -> hub(ex, body -> chat.hub().download(Json.string(body, "repo", "").strip(),
                Json.string(body, "option", ""))));
        server.get("/api/hub/downloads", ex -> {
            Map<String, Object> out = new java.util.LinkedHashMap<>();
            out.put("downloads", chat.hub().downloads());
            out.put("folder", chat.hub().folder().toString());
            out.put("runs", chat.hub().runs());
            out.put("token", chat.hubToken());
            Exchanges.json(ex, 200, out);
        });
        server.get("/api/vast", ex -> Exchanges.json(ex, 200, chat.vast() == null ? Map.of() : chat.vast().status()));
        server.post("/api/vast/key", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 8192);
            if (Boolean.TRUE.equals(body.get("remove"))) {
                chat.removeKey(io.github.zzuegg.brainiac.studio.chat.VastAi.PROVIDER, "default");
            } else {
                String key = Json.string(body, "key", "").strip();
                if (key.isEmpty() || key.chars().anyMatch(c -> c <= ' ')) {
                    Exchanges.error(ex, 400, "paste the key on its own, without spaces");
                    return;
                }
                chat.addKey(io.github.zzuegg.brainiac.studio.chat.VastAi.PROVIDER, "default", key);
            }
            Exchanges.json(ex, 200, chat.vast().status());
        });
        server.post("/api/vast/offers", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 8192);
            try {
                Exchanges.json(ex, 200, Map.of("offers", chat.vast().offers(Json.string(body, "gpu", ""),
                        Json.number(body, "minGpuRam", 0), (int) Json.integer(body, "gpus", 0),
                        Json.number(body, "maxPrice", 0))));
            } catch (java.io.IOException e) {
                Exchanges.error(ex, 400, e.getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Exchanges.error(ex, 500, "interrupted");
            }
        });
        server.post("/api/hub/token", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 8192);
            boolean remove = Boolean.TRUE.equals(body.get("remove"));
            boolean ok = chat.saveHubToken(remove ? null : Json.string(body, "token", ""));
            Map<String, Object> out = new java.util.LinkedHashMap<>();
            out.put("ok", ok);
            out.put("token", chat.hubToken());
            Exchanges.json(ex, 200, out);
        });
        server.post("/api/hub/cancel", ex -> hub(ex, body -> Map.of("ok", chat.hub().cancel(Json.string(body, "id", "")))));
        server.post("/api/hub/delete", ex -> hub(ex, body -> Map.of("ok", chat.hub().delete(Json.string(body, "name", "")))));
        server.post("/api/local/browse", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 64 << 10);
            String path = Json.string(body, "path", "").strip();
            try {
                Exchanges.json(ex, 200, chat.browse(path.isEmpty() ? null : Path.of(path)));
            } catch (java.nio.file.InvalidPathException e) {
                Exchanges.error(ex, 400, "not a path: " + path);
            } catch (IOException e) {
                Exchanges.error(ex, 400, e.getMessage());
            }
        });
    }

    private interface HubCall {
        Object answer(Map<String, Object> body) throws IOException;
    }

    // a Hub call answers its result, or 400 with what went wrong
    private void hub(HttpExchange ex, HubCall call) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 64 << 10);
        Object out;
        try {
            out = call.answer(body);
        } catch (IOException e) {
            Exchanges.error(ex, 400, e.getMessage());
            return;
        }
        Exchanges.json(ex, 200, out);
    }

    // {"provider": "huggingface", "auto": true, "models": ["a", "b"] or null for all, "url": "" for its own,
    // "options": {"billTo": "my-org"}}; a field left out stays as it is
    private void providers(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 1024 * 1024);
        String provider = Json.string(body, "provider", "");
        Boolean auto = body.get("auto") instanceof Boolean b ? b : null;
        java.util.Optional<List<String>> models = null;
        if (body.containsKey("models")) {
            if (body.get("models") == null) {
                models = java.util.Optional.empty();
            } else {
                List<String> ids = new ArrayList<>();
                for (Object o : Json.array(body.get("models"))) {
                    if (!(o instanceof String id)) {
                        throw new IllegalArgumentException("models are listed by id");
                    }
                    ids.add(id);
                }
                models = java.util.Optional.of(ids);
            }
        }
        String url = body.get("url") instanceof String u ? u : null;
        Map<String, String> options = null;
        if (body.get("options") != null) {
            options = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : Json.object(body.get("options")).entrySet()) {
                options.put(e.getKey(), e.getValue() == null ? "" : String.valueOf(e.getValue()));
            }
        }
        chat.configure(provider, auto, models, url, options);
        Exchanges.json(ex, 200, chat.describe(url != null));
    }

    private void images(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 256 * 1024);
        long start = System.nanoTime();
        generated(ex, start, () -> chat.images(Json.string(body, "model", ChatService.AUTO),
                Json.string(body, "prompt", ""), Json.string(body, "aspect", "1:1"),
                (int) Json.integer(body, "count", 1), Json.string(body, "key", null)));
    }

    private void draw(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 256 * 1024);
        String model = Json.string(body, "model", ChatService.AUTO);
        String prompt = Json.string(body, "prompt", "");
        if (prompt.isBlank()) {
            throw new Exchanges.BadRequest("describe the picture first");
        }
        long seed = body.get("seed") instanceof Number n ? n.longValue() : new java.util.Random().nextLong(1L << 32);
        LocalImages.Settings settings = new LocalImages.Settings(Json.string(body, "negativePrompt", ""),
                Json.string(body, "aspect", "1:1"), (int) Json.integer(body, "count", 1), (int) Json.integer(body, "steps", 25),
                Json.number(body, "guidance", 7), seed, Json.string(body, "sampler", "DPM_PP_2M_KARRAS"));

        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.getResponseHeaders().set("X-Content-Type-Options", "nosniff");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            Events events = new Events(out);
            long start = System.nanoTime();
            try {
                ChatService.Drawn d = chat.draw(model, prompt, settings, (step, total, preview) -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("step", step);
                    m.put("total", total);
                    events.sendQuietly("step", m);
                    if (preview != null) {
                        Map<String, Object> pv = new LinkedHashMap<>();
                        pv.put("width", preview.width());
                        pv.put("height", preview.height());
                        pv.put("data", java.util.Base64.getEncoder().encodeToString(preview.png()));
                        events.sendQuietly("preview", pv);
                    }
                    return !events.broken;   // the page went away: stop drawing
                }, m -> {
                    Map<String, Object> route = new LinkedHashMap<>();
                    route.put("key", m.key());
                    route.put("name", m.name());
                    route.put("seed", seed);
                    events.sendQuietly("route", route);
                });
                for (int i = 0; i < d.generated().media().size(); i++) {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("mime", d.generated().media().get(i).mime());
                    m.put("data", java.util.Base64.getEncoder().encodeToString(d.generated().media().get(i).data()));
                    m.put("file", d.generated().files().get(i).toString());
                    m.put("seed", d.seeds().get(i));
                    events.send("image", m);
                }
                events.send("done", Map.of("seconds", (System.nanoTime() - start) / 1e9));
            } catch (java.util.concurrent.CancellationException e) {
                events.sendQuietly("error", Map.of("message", "stopped"));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                events.sendQuietly("error", Map.of("message", "stopped"));
            } catch (IOException | RuntimeException e) {
                if (events.broken) {
                    return;
                }
                LOG.log(System.Logger.Level.INFO, "drawing failed: " + e.getMessage());
                events.send("error", Map.of("message", String.valueOf(e.getMessage())));
            }
        } catch (IOException closed) {
            // the page stopped reading
        }
    }

    private void voice(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 16 * 1024 * 1024);
        String model = Json.string(body, "model", "");
        String name = Json.string(body, "name", "").strip();
        try {
            if (Boolean.TRUE.equals(body.get("remove"))) {
                chat.removeVoice(model, name);
                Exchanges.json(ex, 200, Map.of("removed", name));
                return;
            }
            byte[] wav;
            try {
                wav = java.util.Base64.getDecoder().decode(Json.string(body, "audio", ""));
            } catch (IllegalArgumentException e) {
                throw new Exchanges.BadRequest("send the recording as base64 WAV in \"audio\"");
            }
            String transcript = chat.saveVoice(model, name, wav, Json.string(body, "transcript", ""));
            Exchanges.json(ex, 200, Map.of("voice", name, "transcript", transcript));
        } catch (IllegalArgumentException e) {
            throw new Exchanges.BadRequest(e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted");
        }
    }

    private void transcribe(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 64 * 1024 * 1024);
        byte[] wav;
        try {
            wav = java.util.Base64.getDecoder().decode(Json.string(body, "audio", ""));
        } catch (IllegalArgumentException e) {
            throw new Exchanges.BadRequest("send the recording as base64 WAV in \"audio\"");
        }
        if (wav.length < 44) {
            throw new Exchanges.BadRequest("send the recording as base64 WAV in \"audio\"");
        }
        String language = Json.string(body, "language", "");
        LocalSpeech.Listening listening = new LocalSpeech.Listening(language.isBlank() || "auto".equals(language) ? null : language,
                Boolean.TRUE.equals(body.get("translate")), !Boolean.FALSE.equals(body.get("timestamps")),
                Json.string(body, "prompt", ""));

        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.getResponseHeaders().set("X-Content-Type-Options", "nosniff");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            Events events = new Events(out);
            long start = System.nanoTime();
            try {
                ChatService.Transcribed t = chat.transcribe(Json.string(body, "model", ChatService.AUTO), wav, listening, seg -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("start", seg.start());
                    m.put("end", seg.end());
                    m.put("text", seg.text());
                    events.sendQuietly("segment", m);
                }, m -> {
                    Map<String, Object> route = new LinkedHashMap<>();
                    route.put("key", m.key());
                    route.put("name", m.name());
                    events.sendQuietly("route", route);
                }, () -> events.broken);
                Map<String, Object> done = new LinkedHashMap<>();
                done.put("text", t.transcription().text());
                done.put("language", t.transcription().language());
                done.put("audioSeconds", t.transcription().seconds());
                done.put("seconds", (System.nanoTime() - start) / 1e9);
                events.send("done", done);
            } catch (java.util.concurrent.CancellationException e) {
                events.sendQuietly("error", Map.of("message", "stopped"));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                events.sendQuietly("error", Map.of("message", "stopped"));
            } catch (IOException | RuntimeException e) {
                if (events.broken) {
                    return;
                }
                LOG.log(System.Logger.Level.INFO, "transcribing failed: " + e.getMessage());
                events.send("error", Map.of("message", String.valueOf(e.getMessage())));
            }
        } catch (IOException closed) {
            // the page stopped reading
        }
    }

    private void speech(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 1024 * 1024);
        long start = System.nanoTime();
        generated(ex, start, () -> chat.speech(Json.string(body, "model", ChatService.AUTO),
                Json.string(body, "text", ""), Json.string(body, "voice", ""), Json.number(body, "speed", 1),
                Json.string(body, "instructions", ""), Json.string(body, "format", "mp3"),
                Json.string(body, "key", null)));
    }

    /** Makes something with a provider. */
    @FunctionalInterface
    private interface Generation {
        ChatService.Generated run() throws IOException, InterruptedException;
    }

    // A provider failure is a 502 with the provider's own words; a bad request was already a 400.
    private static void generated(HttpExchange ex, long start, Generation g) throws IOException {
        ChatService.Generated out;
        try {
            out = g.run();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Exchanges.error(ex, 503, "stopped");
            return;
        } catch (IOException e) {
            Exchanges.error(ex, 502, String.valueOf(e.getMessage()));
            return;
        }
        List<Object> items = new ArrayList<>();
        for (int i = 0; i < out.media().size(); i++) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("mime", out.media().get(i).mime());
            m.put("data", java.util.Base64.getEncoder().encodeToString(out.media().get(i).data()));
            m.put("file", out.files().get(i).toString());
            items.add(m);
        }
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("key", out.model().key());
        model.put("name", out.model().name());
        model.put("provider", out.model().provider());
        Map<String, Object> answer = new LinkedHashMap<>();
        answer.put("model", model);
        answer.put("items", items);
        answer.put("seconds", (System.nanoTime() - start) / 1e9);
        Exchanges.json(ex, 200, answer);
    }

    private void keys(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, 64 * 1024);
        String provider = Json.string(body, "provider", "");
        String label = Json.string(body, "label", "default");
        if (Boolean.TRUE.equals(body.get("remove"))) {
            if (!chat.removeKey(provider, label)) {
                throw new Exchanges.BadRequest("no saved " + provider + " key named " + label
                        + " (a key from an environment variable is removed there)");
            }
        } else if (Boolean.TRUE.equals(body.get("first"))) {
            chat.makeKeyFirst(provider, label);
        } else {
            chat.addKey(provider, label, Json.string(body, "key", ""));
        }
        Exchanges.json(ex, 200, chat.describe(false));
    }

    /**
     * Reads the pictures on a message: {@code images: [{type, data}]}, the data in base64.
     *
     * @param message the message as it was sent
     * @return its pictures, none if it has none
     * @throws Exchanges.BadRequest if one is not base64
     */
    static List<io.github.zzuegg.brainiac.studio.chat.Media> images(Map<String, Object> message) {
        List<io.github.zzuegg.brainiac.studio.chat.Media> images = new ArrayList<>();
        for (Object i : Json.array(message.getOrDefault("images", List.of()))) {
            Map<String, Object> image = Json.object(i);
            try {
                images.add(new io.github.zzuegg.brainiac.studio.chat.Media(
                        java.util.Base64.getDecoder().decode(Json.string(image, "data", "")),
                        Json.string(image, "type", "")));
            } catch (IllegalArgumentException e) {
                throw new Exchanges.BadRequest("a picture is not valid base64");
            }
        }
        return images;
    }

    private Map<String, Object> localDevice() {
        io.github.zzuegg.brainiac.studio.chat.LocalDevice d = chat.localDevice();
        Map<String, Object> out = new java.util.LinkedHashMap<>();
        out.put("choice", d.choice());
        out.put("device", d.device().name());
        // the GPU memory limit in MB: the one set here, and the one this run started with
        out.put("memory", d.memory());
        out.put("memoryInEffect", io.github.zzuegg.brainiac.studio.chat.LocalDevice.memoryInEffect());
        return out;
    }

    /**
     * Reads a request's {@code temperature}.
     *
     * @param body the request
     * @return the temperature, or null when it is left to the model
     */
    static Float temperature(Map<String, Object> body) {
        if (!(body.get("temperature") instanceof Number n)) {
            return null;
        }
        float t = n.floatValue();
        if (!(t >= 0 && t <= 2)) {
            throw new Exchanges.BadRequest("temperature must be between 0 and 2");
        }
        return t;
    }

    /**
     * Reads a request's {@code topP}, {@code topK} and {@code repeatPenalty}.
     *
     * @param body the request
     * @return them, each null when it is left to the model
     * @throws Exchanges.BadRequest if one is out of its range
     */
    static io.github.zzuegg.brainiac.studio.chat.Sampler sampler(Map<String, Object> body) {
        try {
            return new io.github.zzuegg.brainiac.studio.chat.Sampler(
                    body.get("topP") instanceof Number p ? p.floatValue() : null,
                    body.get("topK") instanceof Number k ? k.intValue() : null,
                    body.get("repeatPenalty") instanceof Number r ? r.floatValue() : null);
        } catch (IllegalArgumentException e) {
            throw new Exchanges.BadRequest(e.getMessage());
        }
    }

    /**
     * Reads a request's {@code format}, the form its answer must have.
     *
     * @param body the request
     * @return the format; free text when the request has none
     * @throws Exchanges.BadRequest if it is not a format
     */
    static io.github.zzuegg.brainiac.studio.chat.Format format(Map<String, Object> body) {
        try {
            return io.github.zzuegg.brainiac.studio.chat.Format.of(body);
        } catch (IllegalArgumentException e) {
            throw new Exchanges.BadRequest(e.getMessage());
        }
    }

    private void chat(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
        String model = Json.string(body, "model", ChatService.AUTO);
        String system = Json.string(body, "system", "");
        String key = Json.string(body, "key", null);
        int maxTokens = (int) Math.min(Integer.MAX_VALUE, Json.integer(body, "maxTokens", DEFAULT_MAX_TOKENS));
        List<ChatMessage> messages = new ArrayList<>();
        for (Object o : Json.array(body.getOrDefault("messages", List.of()))) {
            Map<String, Object> m = Json.object(o);
            messages.add(new ChatMessage(Json.string(m, "role", ""), Json.string(m, "text", ""), images(m)));
        }
        Float temperature = temperature(body);
        io.github.zzuegg.brainiac.studio.chat.Sampler sampler = sampler(body);
        io.github.zzuegg.brainiac.studio.chat.Format format = format(body);
        if (messages.isEmpty() || !"user".equals(messages.getLast().role()) || maxTokens < 1) {
            throw new Exchanges.BadRequest("send at least one message, ending with the user's, and maxTokens >= 1");
        }

        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.getResponseHeaders().set("X-Content-Type-Options", "nosniff");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            Events events = new Events(out);
            long start = System.nanoTime();
            try {
                ChatResult r = chat.chat(model, system, messages, maxTokens, key, temperature, sampler, format, m -> {
                    Map<String, Object> route = new LinkedHashMap<>();
                    route.put("key", m.key());
                    route.put("provider", m.provider());
                    route.put("name", m.name());
                    route.put("local", "local".equals(m.provider()));
                    events.sendQuietly("route", route);
                }, events);
                Map<String, Object> done = new LinkedHashMap<>();
                done.put("stop", r.stop());
                done.put("inputTokens", r.inputTokens());
                done.put("outputTokens", r.outputTokens());
                done.put("seconds", (System.nanoTime() - start) / 1e9);
                events.send("done", done);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                events.send("error", Map.of("message", "stopped"));
            } catch (IOException | RuntimeException e) {
                if (events.broken) {
                    return;   // the page went away; nothing to tell
                }
                LOG.log(System.Logger.Level.INFO, "chat failed: " + e.getMessage());
                events.send("error", Map.of("message", String.valueOf(e.getMessage())));
            }
        } catch (IOException closed) {
            // the page stopped reading
        }
    }

    /** Writes events to the page; a failed write stops the answer. */
    private static final class Events implements ChatListener {
        private final OutputStream out;
        private boolean broken;

        Events(OutputStream out) {
            this.out = out;
        }

        @Override
        public void text(String text) throws IOException {
            send("text", Map.of("text", text));
        }

        @Override
        public void thinking(String text) throws IOException {
            send("thinking", Map.of("text", text));
        }

        void sendQuietly(String event, Map<String, Object> data) {
            try {
                send(event, data);
            } catch (IOException e) {
                broken = true;
            }
        }

        synchronized void send(String event, Map<String, Object> data) throws IOException {
            if (broken) {
                throw new IOException("the page stopped reading");
            }
            try {
                out.write(("event: " + event + "\ndata: " + Json.write(data) + "\n\n").getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException e) {
                broken = true;
                throw e;
            }
        }
    }
}
