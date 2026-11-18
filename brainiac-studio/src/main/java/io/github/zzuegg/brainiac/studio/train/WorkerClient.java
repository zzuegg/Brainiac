package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.FileTime;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Talks to a studio started with {@code --worker} on another machine: what it has, sending it
 * the model and data files of a run (each only once, by its SHA-256), starting and stopping runs
 * there, and fetching the trained model back.
 */
public final class WorkerClient {

    /** The header a worker reads its token from. */
    public static final String AUTHORIZATION = "Authorization";

    private final URI base;
    private final String token;
    private final HttpClient http;

    /**
     * Uses a worker.
     *
     * @param url its address, such as {@code http://192.168.1.20:7870/}
     * @param token the token it printed when it started
     */
    public WorkerClient(String url, String token) {
        this.base = normalize(url);
        this.token = token;
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    }

    /**
     * Turns what someone typed into a worker address: {@code 192.168.1.20:7870} becomes
     * {@code http://192.168.1.20:7870/}.
     *
     * @param url what was typed
     * @return the address
     */
    public static URI normalize(String url) {
        String u = url == null ? "" : url.strip();
        if (u.isEmpty()) {
            throw new IllegalArgumentException("give the worker's address, such as 192.168.1.20:7870");
        }
        if (!u.contains("://")) {
            u = "http://" + u;
        }
        URI uri = URI.create(u);
        if (!"http".equals(uri.getScheme()) && !"https".equals(uri.getScheme()) || uri.getHost() == null) {
            throw new IllegalArgumentException("not a worker address: " + url);
        }
        String path = uri.getPath() == null || uri.getPath().isEmpty() ? "/" : uri.getPath();
        return URI.create(uri.getScheme() + "://" + uri.getRawAuthority() + (path.endsWith("/") ? path : path + "/"));
    }

    /**
     * Returns the worker's address.
     *
     * @return the address
     */
    public URI base() {
        return base;
    }

    private HttpRequest.Builder request(String path) {
        return HttpRequest.newBuilder(base.resolve(path)).header(AUTHORIZATION, "Bearer " + token)
                .timeout(Duration.ofSeconds(30));
    }

    private Map<String, Object> json(HttpRequest req) throws IOException {
        HttpResponse<String> r = send(req, HttpResponse.BodyHandlers.ofString());
        Object body;
        try {
            body = Json.parse(r.body());
        } catch (RuntimeException e) {
            throw new IOException("the worker answered " + r.statusCode() + " with something that is not JSON");
        }
        Map<String, Object> m = Json.object(body);
        if (r.statusCode() >= 400) {
            throw new IOException(r.statusCode() == 401 || r.statusCode() == 403 ? "the worker did not accept the token; copy it again from where the worker printed it"
                    : Json.string(m, "error", "the worker answered " + r.statusCode()));
        }
        return m;
    }

    private <T> HttpResponse<T> send(HttpRequest req, HttpResponse.BodyHandler<T> handler) throws IOException {
        try {
            return http.send(req, handler);
        } catch (java.net.ConnectException | java.net.http.HttpConnectTimeoutException e) {
            throw new IOException("nothing answers at " + base + "; is the worker running, and is the port open?", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted", e);
        }
    }

    /**
     * Asks the worker what it is and what it has.
     *
     * @return {@code {name, devices, active, queued, cores, memory}}
     * @throws IOException if it cannot be reached or does not accept the token
     */
    public Map<String, Object> info() throws IOException {
        return json(request("worker/info").timeout(Duration.ofSeconds(5)).GET().build());
    }

    /**
     * Lends the worker a secret, sealed so that only this worker can open it, or takes it back.
     * The worker holds it in memory only.
     *
     * @param name what it is, such as {@code huggingface}
     * @param value the secret, or empty to take it back
     * @return the names of the secrets the worker now holds
     * @throws IOException if it cannot be reached, is too old to be lent secrets, or refuses
     */
    public List<String> lend(String name, String value) throws IOException {
        Map<String, Object> offer = json(request("worker/secrets").timeout(Duration.ofSeconds(10)).GET().build());
        if (!(offer.get("public") instanceof String theirs)) {
            throw new IOException("this worker is too old to be lent secrets: update it");
        }
        java.security.KeyPair pair = SealedSecret.pair();
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", Json.string(offer, "id", ""));
        body.put("public", SealedSecret.publicText(pair));
        body.put("name", name);
        try {
            body.put("sealed", SealedSecret.seal(SealedSecret.key(pair.getPrivate(), theirs, token), name, value));
        } catch (java.security.GeneralSecurityException e) {
            throw new IOException("the worker's one-time key is not a key");
        }
        Map<String, Object> m = json(request("worker/secrets").timeout(Duration.ofSeconds(10))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body))).build());
        return Json.array(m.getOrDefault("names", List.of())).stream().map(String::valueOf).toList();
    }

    /**
     * Tells the worker to download a model from Hugging Face into its own downloads folder.
     *
     * @param repo the repository, such as {@code Qwen/Qwen2.5-0.5B-Instruct-GGUF}
     * @param option a GGUF file's path in it, or {@code folder} for the whole model folder
     * @return the download, as the worker describes it
     * @throws IOException if it cannot be reached, or it refuses the download
     */
    public Map<String, Object> hubDownload(String repo, String option) throws IOException {
        Map<String, Object> body = new java.util.LinkedHashMap<>();
        body.put("repo", repo);
        body.put("option", option);
        return json(request("worker/hub/download").timeout(Duration.ofSeconds(60)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body))).build());
    }

    /**
     * Lists the worker's model downloads since it started, newest first.
     *
     * @return the downloads
     * @throws IOException if it cannot be reached
     */
    public List<Object> hubDownloads() throws IOException {
        Map<String, Object> m = json(request("worker/hub/downloads").timeout(Duration.ofSeconds(5)).GET().build());
        return Json.array(m.getOrDefault("downloads", List.of()));
    }

    /**
     * Stops one of the worker's downloads; what it fetched is kept, so starting it again resumes.
     *
     * @param id the download's id on the worker
     * @return whether it was running
     * @throws IOException if it cannot be reached
     */
    public boolean hubCancel(String id) throws IOException {
        Map<String, Object> m = json(request("worker/hub/cancel").timeout(Duration.ofSeconds(10))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(Map.of("id", id)))).build());
        return Boolean.TRUE.equals(m.get("ok"));
    }

    /**
     * Lists the chat models the worker serves.
     *
     * @return the models, with the worker's own provider id
     * @throws IOException if it cannot be reached or does not accept the token
     */
    public List<io.github.zzuegg.brainiac.studio.chat.ModelInfo> models() throws IOException {
        Map<String, Object> m = json(request("worker/models").timeout(Duration.ofSeconds(5)).GET().build());
        List<io.github.zzuegg.brainiac.studio.chat.ModelInfo> out = new ArrayList<>();
        Map<String, Map<String, Object>> heard = new LinkedHashMap<>();
        for (Object o : Json.array(m.getOrDefault("models", List.of()))) {
            Map<String, Object> x = Json.object(o);
            List<String> media = Json.array(x.getOrDefault("media", List.of("text"))).stream().map(String::valueOf).toList();
            out.add(new io.github.zzuegg.brainiac.studio.chat.ModelInfo(Json.string(x, "provider", "local"),
                    Json.string(x, "id", ""), Json.string(x, "name", Json.string(x, "id", "")), media, 0,
                    Json.string(x, "detail", "")));
            if (x.get("speech") instanceof Map<?, ?> speech) {
                heard.put(Json.string(x, "id", ""), Json.object(speech));
            }
        }
        voices = heard;
        return out;
    }

    // the voices of the worker's speech models, as its last model list told them
    private volatile Map<String, Map<String, Object>> voices;

    /**
     * Returns the voices and other choices of one of the worker's speech models, as its last
     * model list told them.
     *
     * @param model the worker's model id
     * @return them, or null if it told none
     */
    public Map<String, Object> voiceOptions(String model) {
        if (voices == null) {
            try {
                models();   // a client that never listed asks once
            } catch (IOException e) {
                return null;
            }
        }
        return voices.get(model);
    }

    // reads a worker's event stream to its end; "done" and "error" end it, the rest go to the handler
    private Map<String, Object> stream(String path, Map<String, Object> body, String what,
            java.util.function.BiPredicate<String, Map<String, Object>> onEvent) throws IOException {
        HttpResponse<InputStream> r = send(request(path).timeout(Duration.ofMinutes(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body))).build(), HttpResponse.BodyHandlers.ofInputStream());
        if (r.statusCode() >= 400) {
            String text;
            try (InputStream in = r.body()) {
                text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
            String msg;
            try {
                msg = Json.string(Json.object(Json.parse(text)), "error", "");
            } catch (RuntimeException e) {
                msg = "";
            }
            throw new IOException(r.statusCode() == 401 ? "the worker did not accept the token"
                    : r.statusCode() == 404 ? "the worker is an older build that cannot " + what + ": update it"
                    : msg.isEmpty() ? "the worker answered " + r.statusCode() : msg);
        }
        Object[] end = {null, null};
        boolean[] stopped = {false};
        try (InputStream in = r.body()) {
            io.github.zzuegg.brainiac.studio.chat.Sse.read(in, (event, data) -> {
                Map<String, Object> d = Json.object(Json.parse(data));
                switch (event) {
                    case "done" -> {
                        end[0] = d;
                        return false;
                    }
                    case "error" -> {
                        end[1] = Json.string(d, "message", "the worker failed");
                        return false;
                    }
                    default -> {
                        if (!onEvent.test(event, d)) {
                            stopped[0] = true;   // closing the stream tells the worker to stop
                            return false;
                        }
                    }
                }
                return true;
            });
        }
        if (stopped[0]) {
            throw new java.util.concurrent.CancellationException("stopped");
        }
        if (end[1] != null) {
            throw new IOException((String) end[1]);
        }
        if (end[0] == null) {
            throw new IOException("the worker stopped before it was done");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> d = (Map<String, Object>) end[0];
        return d;
    }

    /**
     * Has the worker draw pictures with one of its image models.
     *
     * @param model the worker's model id
     * @param prompt what to draw
     * @param s size, steps, guidance, seed and sampler
     * @param listener hears each step (without previews) and can stop the drawing
     * @return the pictures with their seeds
     * @throws IOException if the worker fails
     */
    public List<io.github.zzuegg.brainiac.studio.chat.LocalImages.Picture> draw(String model, String prompt,
            io.github.zzuegg.brainiac.studio.chat.LocalImages.Settings s,
            io.github.zzuegg.brainiac.studio.chat.LocalImages.Listener listener) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("prompt", prompt);
        body.put("negativePrompt", s.negativePrompt());
        body.put("aspect", s.aspect());
        body.put("count", s.count());
        body.put("steps", s.steps());
        body.put("guidance", s.guidance());
        body.put("seed", s.seed());
        body.put("sampler", s.sampler());
        List<io.github.zzuegg.brainiac.studio.chat.LocalImages.Picture> out = new ArrayList<>();
        stream("worker/draw", body, "draw", (event, d) -> {
            if (event.equals("step")) {
                return listener.step((int) Json.integer(d, "step", 0), (int) Json.integer(d, "total", 0), null);
            }
            if (event.equals("image")) {
                out.add(new io.github.zzuegg.brainiac.studio.chat.LocalImages.Picture(
                        new io.github.zzuegg.brainiac.studio.chat.Media(
                                java.util.Base64.getDecoder().decode(Json.string(d, "data", "")),
                                Json.string(d, "mime", "image/png")), Json.integer(d, "seed", 0)));
            }
            return true;
        });
        return out;
    }

    /**
     * Asks the worker for a video model's usual settings and sizes.
     *
     * @param model the worker's model id
     * @return what the worker says, or null if it cannot be asked or does not know the model
     */
    public Map<String, Object> videoOptions(String model) {
        try {
            Map<String, Object> m = json(request("worker/video/options").timeout(Duration.ofSeconds(10))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(Json.write(Map.of("model", model)))).build());
            return m.isEmpty() ? null : m;
        } catch (IOException e) {
            return null;   // an older worker, or one that is off: the page shows the usual settings
        }
    }

    /**
     * Has the worker make a video clip with one of its video models.
     *
     * @param model the worker's model id
     * @param prompt what the clip shows
     * @param s size, length, steps, guidance, seed and the first frame
     * @param listener hears each step and can stop the clip
     * @return the clip
     * @throws IOException if the worker fails
     */
    public io.github.zzuegg.brainiac.studio.chat.LocalVideos.Clip video(String model, String prompt,
            io.github.zzuegg.brainiac.studio.chat.LocalVideos.Settings s,
            io.github.zzuegg.brainiac.studio.chat.LocalVideos.Listener listener) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("prompt", prompt);
        body.put("negativePrompt", s.negativePrompt());
        body.put("aspect", s.aspect());
        body.put("small", s.small());
        body.put("seconds", s.seconds());
        body.put("steps", s.steps());
        body.put("guidance", s.guidance());
        body.put("seed", s.seed());
        body.put("width", s.width());
        body.put("height", s.height());
        if (s.start() != null) {
            body.put("start", Map.of("type", s.start().mime(),
                    "data", java.util.Base64.getEncoder().encodeToString(s.start().data())));
        }
        io.github.zzuegg.brainiac.studio.chat.LocalVideos.Clip[] clip = {null};
        stream("worker/video", body, "video", (event, d) -> {
            if (event.equals("step")) {
                return listener.step((int) Json.integer(d, "step", 0), (int) Json.integer(d, "total", 0));
            }
            if (event.equals("video")) {
                clip[0] = new io.github.zzuegg.brainiac.studio.chat.LocalVideos.Clip(
                        new io.github.zzuegg.brainiac.studio.chat.Media(
                                java.util.Base64.getDecoder().decode(Json.string(d, "data", "")),
                                Json.string(d, "mime", "image/apng")), Json.integer(d, "seed", 0),
                        (int) Json.integer(d, "width", 0), (int) Json.integer(d, "height", 0),
                        (int) Json.integer(d, "frames", 0), (int) Json.integer(d, "fps", 0));
            }
            return true;
        });
        if (clip[0] == null) {
            throw new IOException("the worker sent no clip");
        }
        return clip[0];
    }

    /**
     * Has the worker speak a text with one of its speech models.
     *
     * @param model the worker's model id
     * @param text what to say
     * @param voice the voice, or an empty string for the model's own
     * @param speed 1 for the usual speed
     * @return the audio
     * @throws IOException if the worker fails
     */
    public io.github.zzuegg.brainiac.studio.chat.Media speak(String model, String text, String voice, double speed)
            throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("text", text);
        body.put("voice", voice == null ? "" : voice);
        body.put("speed", speed);
        Map<String, Object> d = stream("worker/speak", body, "speak", (event, x) -> true);
        return new io.github.zzuegg.brainiac.studio.chat.Media(
                java.util.Base64.getDecoder().decode(Json.string(d, "data", "")), Json.string(d, "mime", "audio/wav"));
    }

    /**
     * Has the worker write down a recording with one of its transcribing models.
     *
     * @param model the worker's model id
     * @param wav the recording, a WAV file
     * @param l the language and what else to do
     * @param onSegment hears each part as it is written down
     * @param stop asked after each part whether to give up
     * @return what was said
     * @throws IOException if the worker fails
     */
    public io.github.zzuegg.brainiac.audio.Transcription transcribe(String model, byte[] wav,
            io.github.zzuegg.brainiac.studio.chat.LocalSpeech.Listening l,
            java.util.function.Consumer<io.github.zzuegg.brainiac.audio.Transcription.Segment> onSegment,
            java.util.function.BooleanSupplier stop) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("audio", java.util.Base64.getEncoder().encodeToString(wav));
        body.put("language", l.language() == null ? "" : l.language());
        body.put("translate", l.translate());
        body.put("timestamps", l.timestamps());
        body.put("prompt", l.prompt() == null ? "" : l.prompt());
        List<io.github.zzuegg.brainiac.audio.Transcription.Segment> segments = new ArrayList<>();
        Map<String, Object> d = stream("worker/transcribe", body, "transcribe", (event, x) -> {
            if (event.equals("segment")) {
                io.github.zzuegg.brainiac.audio.Transcription.Segment seg =
                        new io.github.zzuegg.brainiac.audio.Transcription.Segment(Json.number(x, "start", 0),
                                Json.number(x, "end", 0), Json.string(x, "text", ""), Json.number(x, "avgLogProb", 0),
                                Json.number(x, "noSpeechProb", 0));
                segments.add(seg);
                onSegment.accept(seg);
            }
            return !stop.getAsBoolean();
        });
        return new io.github.zzuegg.brainiac.audio.Transcription(Json.string(d, "text", ""),
                Json.string(d, "language", ""), segments, Json.number(d, "seconds", 0));
    }

    /**
     * Has the worker answer a conversation with one of its models, passing the answer on as it
     * arrives.
     *
     * @param request the conversation; its model is the worker's model id
     * @param listener receives the answer
     * @return how it ended
     * @throws IOException if the worker fails or the listener stops the answer
     */
    public io.github.zzuegg.brainiac.studio.chat.ChatResult chat(io.github.zzuegg.brainiac.studio.chat.ChatRequest request,
            io.github.zzuegg.brainiac.studio.chat.ChatListener listener) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", request.model());
        body.put("system", request.system());
        body.put("maxTokens", request.maxTokens());
        if (request.temperature() != null) {
            body.put("temperature", request.temperature());
        }
        if (request.sampler().topP() != null) {
            body.put("topP", request.sampler().topP());
        }
        if (request.sampler().topK() != null) {
            body.put("topK", request.sampler().topK());
        }
        if (request.sampler().repeatPenalty() != null) {
            body.put("repeatPenalty", request.sampler().repeatPenalty());
        }
        List<Object> messages = new java.util.ArrayList<>();
        for (io.github.zzuegg.brainiac.studio.chat.ChatMessage m : request.messages()) {
            Map<String, Object> message = new LinkedHashMap<>();
            message.put("role", m.role());
            message.put("text", m.text());
            if (!m.images().isEmpty()) {
                // pictures go along for the host's models that look at them
                message.put("images", m.images().stream().map(i -> Map.of("type", i.mime(), "data",
                        io.github.zzuegg.brainiac.studio.chat.ChatMessage.base64(i))).toList());
            }
            messages.add(message);
        }
        body.put("messages", messages);
        if (!request.format().none()) {
            body.put("format", request.format().toJson());
        }
        // the answer streams, so only the wait for its first line is limited
        HttpResponse<InputStream> r = send(request("worker/chat").timeout(Duration.ofMinutes(10))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body))).build(), HttpResponse.BodyHandlers.ofInputStream());
        if (r.statusCode() >= 400) {
            String text;
            try (InputStream in = r.body()) {
                text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
            String msg;
            try {
                msg = Json.string(Json.object(Json.parse(text)), "error", "");
            } catch (RuntimeException e) {
                msg = "";
            }
            throw new IOException(r.statusCode() == 401 ? "the worker did not accept the token"
                    : msg.isEmpty() ? "the worker answered " + r.statusCode() : msg);
        }
        Object[] end = {null, null};
        try (InputStream in = r.body()) {
            io.github.zzuegg.brainiac.studio.chat.Sse.read(in, (event, data) -> {
                Map<String, Object> d = Json.object(Json.parse(data));
                switch (event) {
                    case "text" -> listener.text(Json.string(d, "text", ""));
                    case "thinking" -> listener.thinking(Json.string(d, "text", ""));
                    case "done" -> {
                        end[0] = d;
                        return false;
                    }
                    case "error" -> {
                        end[1] = Json.string(d, "message", "the worker failed");
                        return false;
                    }
                    default -> { }
                }
                return true;
            });
        }
        if (end[1] != null) {
            throw new IOException((String) end[1]);
        }
        if (end[0] == null) {
            throw new IOException("the worker stopped before the answer was done");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> d = (Map<String, Object>) end[0];
        return new io.github.zzuegg.brainiac.studio.chat.ChatResult(Json.string(d, "stop", "end"),
                Json.integer(d, "inputTokens", -1), Json.integer(d, "outputTokens", -1));
    }

    /**
     * Sends a file unless the worker has it already.
     *
     * @param file the file
     * @return {@code sha:<sha256>/<name>}, how a run refers to it
     * @throws IOException if it cannot be read or sent
     */
    public String upload(Path file) throws IOException {
        String sha = sha256(file);
        String name = file.getFileName().toString();
        String q = "worker/files/" + sha + "?name=" + URLEncoder.encode(name, StandardCharsets.UTF_8);
        HttpResponse<String> has = send(request(q).GET().build(), HttpResponse.BodyHandlers.ofString());
        if (has.statusCode() != 200) {
            HttpRequest put = request(q).timeout(Duration.ofHours(2))
                    .PUT(HttpRequest.BodyPublishers.ofFile(file)).build();
            json(put);
        }
        return "sha:" + sha + "/" + name;
    }

    /**
     * Sends a folder, packed as a zip, unless the worker has it already.
     *
     * @param folder the folder, such as images with one subfolder per class
     * @return {@code zip:<sha256>/<name>.zip}, how a run refers to it
     * @throws IOException if it cannot be read or sent
     */
    public String uploadFolder(Path folder) throws IOException {
        Path dir = Files.createTempDirectory("brainiac-data");
        Path zip = dir.resolve(folder.getFileName() + ".zip");
        try {
            pack(folder, zip);
            return "zip:" + upload(zip).substring(4);
        } finally {
            Files.deleteIfExists(zip);
            Files.deleteIfExists(dir);
        }
    }

    /** Zips a folder the same way every time, so the same folder gives the same hash. */
    static void pack(Path folder, Path zip) throws IOException {
        List<Path> files;
        try (Stream<Path> walk = Files.walk(folder)) {
            files = walk.filter(Files::isRegularFile).sorted().toList();
        }
        try (OutputStream raw = Files.newOutputStream(zip); ZipOutputStream out = new ZipOutputStream(raw)) {
            for (Path f : files) {
                ZipEntry e = new ZipEntry(folder.relativize(f).toString().replace('\\', '/'));
                e.setLastModifiedTime(FileTime.fromMillis(0));
                out.putNextEntry(e);
                Files.copy(f, out);
                out.closeEntry();
            }
        }
    }

    /**
     * Starts a run on the worker.
     *
     * @param spec the run settings, with files referred to as {@link #upload} returned them
     * @param model the model, as {@link #upload} returned it
     * @return the run there
     * @throws IOException if the worker refuses it
     */
    public Map<String, Object> start(Map<String, Object> spec, String model) throws IOException {
        String body = Json.write(Map.of("spec", spec, "model", model));
        return json(request("worker/runs").header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build());
    }

    /**
     * Reads a run on the worker.
     *
     * @param id its id there
     * @return the run, as {@link TrainingRun#toJson} describes it
     * @throws IOException if it cannot be read
     */
    public Map<String, Object> run(String id) throws IOException {
        return json(request("worker/runs?id=" + URLEncoder.encode(id, StandardCharsets.UTF_8)).GET().build());
    }

    /**
     * Asks the worker to stop a run.
     *
     * @param id its id there
     * @throws IOException if the worker cannot be reached
     */
    public void stop(String id) throws IOException {
        json(request("worker/stop").header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(Map.of("id", id)))).build());
    }

    /**
     * Fetches the model a run trained.
     *
     * @param id its id there
     * @param target where to put it
     * @throws IOException if it cannot be fetched
     */
    public void download(String id, Path target) throws IOException {
        HttpRequest req = request("worker/result?id=" + URLEncoder.encode(id, StandardCharsets.UTF_8))
                .timeout(Duration.ofHours(1)).GET().build();
        Path tmp = target.resolveSibling(target.getFileName() + ".part");
        HttpResponse<Path> r = send(req, HttpResponse.BodyHandlers.ofFile(tmp));
        if (r.statusCode() != 200) {
            Files.deleteIfExists(tmp);
            throw new IOException("the worker has no model for run " + id + " (" + r.statusCode() + ")");
        }
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
    }

    /**
     * Hashes a file.
     *
     * @param file the file
     * @return its SHA-256 in hex
     * @throws IOException if it cannot be read
     */
    public static String sha256(Path file) throws IOException {
        MessageDigest md = sha256Digest();
        try (InputStream in = Files.newInputStream(file)) {
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0; ) {
                md.update(buf, 0, n);
            }
        }
        return HexFormat.of().formatHex(md.digest());
    }

    /**
     * Returns a fresh SHA-256 digest.
     *
     * @return the digest
     */
    public static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
