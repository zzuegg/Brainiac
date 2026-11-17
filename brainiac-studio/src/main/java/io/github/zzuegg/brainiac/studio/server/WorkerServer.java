package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.studio.chat.ModelInfo;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.ChatResult;
import io.github.zzuegg.brainiac.studio.chat.ChatRequest;
import io.github.zzuegg.brainiac.studio.chat.ChatProvider;
import io.github.zzuegg.brainiac.studio.chat.ChatMessage;
import io.github.zzuegg.brainiac.studio.chat.ChatListener;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import io.github.zzuegg.brainiac.studio.train.TrainingRun;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * A studio started with {@code --worker}: it trains runs that another studio sends it. It
 * listens on the network (all addresses unless told otherwise) and answers only callers that
 * send the token it keeps in {@code worker.token} and prints when it starts. It has no web page.
 *
 * <ul>
 *   <li>{@code GET /worker/info}: its name, devices and how busy it is.
 *   <li>{@code GET|PUT /worker/files/<sha256>?name=<name>}: whether it has a file, or sends one.
 *   <li>{@code POST /worker/runs {spec, model}}: starts a run; files are referred to as
 *       {@code sha:<sha256>/<name>} (or {@code zip:...} for a folder sent as a zip).
 *   <li>{@code GET /worker/runs?id=<id>}, {@code POST /worker/stop {id}} and
 *       {@code GET /worker/result?id=<id>}, which sends the trained model.
 *   <li>{@code GET /worker/models}: the chat models it serves, from its model folders and its
 *       {@code downloads} folder; {@code POST /worker/chat {model, system, messages, maxTokens}}
 *       answers with one as server-sent events: {@code text} pieces, then {@code done} or {@code error}.
 * </ul>
 */
public final class WorkerServer implements AutoCloseable {

    /** The port a worker uses when none is given. */
    public static final int DEFAULT_PORT = 7870;

    private static final System.Logger LOG = System.getLogger(WorkerServer.class.getName());
    private static final Pattern SHA = Pattern.compile("[0-9a-f]{64}");
    private static final long MAX_UPLOAD = 16L << 30;
    private static final long MAX_UNZIPPED = 32L << 30;
    private static final int MAX_BODY = 1 << 20;

    private final HttpServer http;
    private final ExecutorService executor;
    private final String token;
    private final TrainingService training;
    private final Path files;
    private final ChatProvider models;

    private WorkerServer(HttpServer http, ExecutorService executor, String token, Path folder, ChatProvider models) {
        this.http = http;
        this.executor = executor;
        this.token = token;
        this.training = new TrainingService(folder);
        this.files = folder.resolve("files");
        this.models = models;
    }

    /**
     * Starts a worker.
     *
     * @param address where to listen, such as {@code 0.0.0.0:7870}
     * @param folder where it keeps its token, the files it is sent, the models it trains and its runs
     * @return the running worker
     * @throws IOException if the port cannot be opened or the token cannot be kept
     */
    public static WorkerServer start(InetSocketAddress address, Path folder) throws IOException {
        return start(address, folder, List.of(), null);
    }

    /**
     * Starts a worker that also serves chat models.
     *
     * @param address where to listen, such as {@code 127.0.0.1:7870}
     * @param folder where it keeps its token, files, models and runs
     * @param modelFolders more folders to serve chat models from
     * @param givenToken the token to use, such as {@code $BRAINIAC_WORKER_TOKEN}, or null to use
     *     (or create) the one in {@code worker.token}
     * @return the running worker
     * @throws IOException if the port cannot be opened or the token cannot be kept
     */
    public static WorkerServer start(InetSocketAddress address, Path folder, List<Path> modelFolders, String givenToken)
            throws IOException {
        List<Path> where = new ArrayList<>(modelFolders);
        where.add(folder.resolve("downloads"));
        where.add(folder.resolve("models"));
        return start(address, folder, ChatService.localModels(where), givenToken);
    }

    /**
     * Starts a worker that serves the chat models of a provider.
     *
     * @param address where to listen
     * @param folder where it keeps its token, files, models and runs
     * @param models the chat models it serves
     * @param givenToken the token to use, or null to use (or create) the one in {@code worker.token}
     * @return the running worker
     * @throws IOException if the port cannot be opened or the token cannot be kept
     */
    public static WorkerServer start(InetSocketAddress address, Path folder, ChatProvider models, String givenToken)
            throws IOException {
        Files.createDirectories(folder);
        String token;
        if (givenToken != null && !givenToken.isBlank()) {
            token = givenToken.strip();
            if (token.length() < 32 || token.chars().anyMatch(c -> c <= ' ')) {
                throw new IllegalArgumentException("the worker token must be at least 32 characters without spaces");
            }
        } else {
            token = token(folder.resolve("worker.token"));
        }
        HttpServer http = HttpServer.create(address, 0);
        ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
        http.setExecutor(executor);
        WorkerServer w = new WorkerServer(http, executor, token, folder, models);
        http.createContext("/worker/", w::handle);
        http.createContext("/", ex -> {
            Exchanges.error(ex, 404, "this is a Brainiac Studio worker; add it on the Train screen of your studio");
        });
        http.start();
        return w;
    }

    private static String token(Path file) throws IOException {
        if (Files.isRegularFile(file)) {
            String t = Files.readString(file, StandardCharsets.UTF_8).strip();
            if (t.length() >= 32) {
                return t;
            }
        }
        byte[] secret = new byte[24];
        new SecureRandom().nextBytes(secret);
        String t = HexFormat.of().formatHex(secret);
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.deleteIfExists(tmp);
        Keys.ownerOnly(Files.createFile(tmp));
        Files.writeString(tmp, t, StandardCharsets.UTF_8);
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        return t;
    }

    /**
     * Returns the token callers must send as {@code Authorization: Bearer <token>}.
     *
     * @return the token
     */
    public String token() {
        return token;
    }

    /**
     * Returns the port it listens on.
     *
     * @return the port
     */
    public int port() {
        return http.getAddress().getPort();
    }

    /**
     * Returns the addresses other machines can probably reach it at.
     *
     * @return such as {@code http://192.168.1.20:7870/}
     */
    public List<URI> addresses() {
        List<URI> out = new ArrayList<>();
        InetAddress bound = http.getAddress().getAddress();
        if (!bound.isAnyLocalAddress()) {
            // listening on one address only, such as 127.0.0.1 behind an SSH tunnel
            String host = bound.getHostAddress();
            out.add(URI.create("http://" + (host.contains(":") ? "[" + host + "]" : host) + ":" + port() + "/"));
            return out;
        }
        try {
            for (var ni : java.util.Collections.list(java.net.NetworkInterface.getNetworkInterfaces())) {
                if (!ni.isUp() || ni.isLoopback() || ni.isVirtual()) {
                    continue;
                }
                for (InetAddress a : java.util.Collections.list(ni.getInetAddresses())) {
                    if (a instanceof java.net.Inet4Address && !a.isLinkLocalAddress()) {
                        out.add(URI.create("http://" + a.getHostAddress() + ":" + port() + "/"));
                    }
                }
            }
        } catch (IOException e) {
            // fall back to the loopback address below
        }
        if (out.isEmpty()) {
            out.add(URI.create("http://127.0.0.1:" + port() + "/"));
        }
        return out;
    }

    @Override
    public void close() {
        http.stop(0);
        training.close();
        executor.shutdownNow();
    }

    private void handle(HttpExchange ex) throws IOException {
        try {
            String auth = ex.getRequestHeaders().getFirst("Authorization");
            String given = auth != null && auth.startsWith("Bearer ") ? auth.substring(7).strip() : "";
            if (!MessageDigest.isEqual(given.getBytes(StandardCharsets.UTF_8), token.getBytes(StandardCharsets.UTF_8))) {
                Exchanges.error(ex, 401, "missing or wrong token");
                return;
            }
            String path = ex.getRequestURI().getPath();
            String method = ex.getRequestMethod();
            Map<String, String> q = query(ex.getRequestURI().getRawQuery());
            if (path.equals("/worker/info") && method.equals("GET")) {
                Exchanges.json(ex, 200, info());
            } else if (path.equals("/worker/models") && method.equals("GET")) {
                List<Object> list = new ArrayList<>();
                for (ModelInfo m : models.models()) {
                    list.add(m.toJson());
                }
                Exchanges.json(ex, 200, Map.of("models", list));
            } else if (path.equals("/worker/chat") && method.equals("POST")) {
                chat(ex, Exchanges.body(ex, 32 << 20));
            } else if (path.startsWith("/worker/files/")) {
                file(ex, path.substring("/worker/files/".length()), q.getOrDefault("name", "data"));
            } else if (path.equals("/worker/runs") && method.equals("POST")) {
                Exchanges.json(ex, 200, startRun(Exchanges.body(ex, MAX_BODY)).toJson(false));
            } else if (path.equals("/worker/runs") && method.equals("GET")) {
                TrainingRun r = training.run(q.getOrDefault("id", "")).orElse(null);
                if (r == null) {
                    Exchanges.error(ex, 404, "no run " + q.get("id"));
                } else {
                    Exchanges.json(ex, 200, r.toJson(true));
                }
            } else if (path.equals("/worker/stop") && method.equals("POST")) {
                boolean ok = training.stop(Json.string(Exchanges.body(ex, MAX_BODY), "id", ""));
                Exchanges.json(ex, ok ? 200 : 404, Map.of("ok", ok));
            } else if (path.equals("/worker/result") && method.equals("GET")) {
                Path f = training.output(q.getOrDefault("id", "")).orElse(null);
                if (f == null) {
                    Exchanges.error(ex, 404, "that run saved no model");
                } else {
                    ex.getResponseHeaders().set("Content-Type", "application/octet-stream");
                    ex.sendResponseHeaders(200, Files.size(f));
                    try (OutputStream out = ex.getResponseBody()) {
                        Files.copy(f, out);
                    }
                }
            } else {
                Exchanges.error(ex, 404, "no such endpoint: " + method + " " + path);
            }
        } catch (Exchanges.BadRequest | IllegalArgumentException e) {
            answerError(ex, 400, e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            answerError(ex, 503, "the worker is stopping");
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "worker request " + ex.getRequestURI() + " failed", e);
            answerError(ex, 500, String.valueOf(e.getMessage()));
        } finally {
            ex.close();
        }
    }

    private void chat(HttpExchange ex, Map<String, Object> body) throws IOException {
        List<ChatMessage> messages = new ArrayList<>();
        for (Object o : Json.array(body.getOrDefault("messages", List.of()))) {
            Map<String, Object> m = Json.object(o);
            // "text" as the studio sends it, or "content" as most chat APIs do
            String text = Json.string(m, "text", Json.string(m, "content", ""));
            if (text.isBlank()) {
                throw new IllegalArgumentException("every message needs a non-empty \"text\"");
            }
            messages.add(new ChatMessage(Json.string(m, "role", "user"), text));
        }
        if (messages.isEmpty()) {
            throw new IllegalArgumentException("\"messages\" is empty");
        }
        // the model's id, or its key from /worker/models ("local/<id>")
        String model = Json.string(body, "model", "");
        String prefix = models.id() + "/";
        if (model.startsWith(prefix)) {
            model = model.substring(prefix.length());
        }
        ChatRequest request = new ChatRequest(model, Json.string(body, "system", ""), messages,
                (int) Math.max(1, Math.min(32768, Json.integer(body, "maxTokens", 1024))), null);
        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.sendResponseHeaders(200, 0);
        OutputStream out = ex.getResponseBody();
        ChatListener listener = new ChatListener() {
            @Override
            public void text(String text) throws IOException {
                event(out, "text", Map.of("text", text));
            }

            @Override
            public void thinking(String text) throws IOException {
                event(out, "thinking", Map.of("text", text));
            }
        };
        try {
            ChatResult r = models.chat(request, listener);
            Map<String, Object> done = new LinkedHashMap<>();
            done.put("stop", r.stop());
            done.put("inputTokens", r.inputTokens());
            done.put("outputTokens", r.outputTokens());
            event(out, "done", done);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            event(out, "error", Map.of("message", "stopped"));
        } catch (IOException | RuntimeException e) {
            try {
                event(out, "error", Map.of("message", String.valueOf(e.getMessage())));
            } catch (IOException gone) {
                // the studio stopped reading
            }
        }
        out.close();
    }

    private static void event(OutputStream out, String name, Map<String, Object> data) throws IOException {
        out.write(("event: " + name + "\ndata: " + Json.write(data) + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private static void answerError(HttpExchange ex, int status, String message) {
        try {
            Exchanges.error(ex, status, message);
        } catch (IOException | RuntimeException ignored) {
            ex.close();   // the answer had already started
        }
    }

    private Map<String, Object> info() {
        Map<String, Object> m = new LinkedHashMap<>();
        String name;
        try {
            name = InetAddress.getLocalHost().getHostName();
        } catch (IOException e) {
            name = "worker";
        }
        m.put("name", name);
        m.put("version", 1);
        List<Object> devices = new ArrayList<>();
        for (Device d : Device.available()) {
            devices.add(Map.of("name", d.name(), "kind", d.kind().name().toLowerCase(java.util.Locale.ROOT)));
        }
        m.put("devices", devices);
        m.put("cores", Runtime.getRuntime().availableProcessors());
        m.put("memory", Runtime.getRuntime().maxMemory());
        List<TrainingRun> runs = training.runs();
        m.put("active", runs.stream().filter(r -> r.state() == TrainingRun.State.RUNNING
                || r.state() == TrainingRun.State.LOADING).count());
        m.put("queued", runs.stream().filter(r -> r.state() == TrainingRun.State.QUEUED).count());
        runs.stream().filter(r -> r.state() == TrainingRun.State.RUNNING).findFirst().ifPresent(r -> {
            Map<String, Object> j = r.toJson(false);
            m.put("training", Map.of("name", r.name(), "epoch", j.get("epoch"), "epochs", j.get("epochs"),
                    "samplesPerSecond", j.getOrDefault("samplesPerSecond", 0)));
        });
        return m;
    }

    private static Map<String, String> query(String raw) {
        Map<String, String> out = new LinkedHashMap<>();
        if (raw == null) {
            return out;
        }
        for (String part : raw.split("&")) {
            int eq = part.indexOf('=');
            if (eq > 0) {
                out.put(URLDecoder.decode(part.substring(0, eq), StandardCharsets.UTF_8),
                        URLDecoder.decode(part.substring(eq + 1), StandardCharsets.UTF_8));
            }
        }
        return out;
    }

    /** A file name that is safe to keep: letters, digits, dot, dash and underscore. */
    static String safeName(String name) {
        String n = name.replaceAll("[^A-Za-z0-9._-]", "_");
        if (n.isEmpty() || n.startsWith(".")) {
            n = "f" + n;
        }
        return n.length() > 100 ? n.substring(n.length() - 100) : n;
    }

    private void file(HttpExchange ex, String sha, String name) throws IOException {
        if (!SHA.matcher(sha).matches()) {
            throw new IllegalArgumentException("files are named by their SHA-256");
        }
        Path target = files.resolve(sha).resolve(safeName(name));
        if (ex.getRequestMethod().equals("GET")) {
            Exchanges.json(ex, Files.isRegularFile(target) ? 200 : 404, Map.of("exists", Files.isRegularFile(target)));
            return;
        }
        if (!ex.getRequestMethod().equals("PUT")) {
            Exchanges.error(ex, 405, "use GET or PUT");
            return;
        }
        Files.createDirectories(target.getParent());
        Path tmp = Files.createTempFile(target.getParent(), "upload", ".part");
        MessageDigest md = io.github.zzuegg.brainiac.studio.train.WorkerClient.sha256Digest();
        long total = 0;
        try (InputStream in = ex.getRequestBody(); OutputStream out = Files.newOutputStream(tmp)) {
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0; ) {
                total += n;
                if (total > MAX_UPLOAD) {
                    throw new IllegalArgumentException("files over 16 GB are not taken");
                }
                md.update(buf, 0, n);
                out.write(buf, 0, n);
            }
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(tmp);
            throw e;
        }
        if (!HexFormat.of().formatHex(md.digest()).equals(sha)) {
            Files.deleteIfExists(tmp);
            throw new IllegalArgumentException("the file arrived damaged: its SHA-256 does not match");
        }
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
        Exchanges.json(ex, 200, Map.of("ok", true, "bytes", total));
    }

    /** Finds a file sent earlier from its reference, {@code sha:<sha256>/<name>}. */
    private Path sent(String ref) {
        if (ref == null || !(ref.startsWith("sha:") || ref.startsWith("zip:")) || ref.indexOf('/') != 68) {
            throw new IllegalArgumentException("a worker only trains on files sent to it, referred to as sha:<sha256>/<name>");
        }
        String sha = ref.substring(4, 68);
        if (!SHA.matcher(sha).matches()) {
            throw new IllegalArgumentException("not a file reference: " + ref);
        }
        Path f = files.resolve(sha).resolve(safeName(ref.substring(69)));
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("the worker was not sent " + ref.substring(69));
        }
        return f;
    }

    private synchronized TrainingRun startRun(Map<String, Object> body) throws IOException {
        Map<String, Object> spec = new LinkedHashMap<>(Json.object(body.get("spec")));
        String modelRef = Json.string(body, "model", null);
        Path model = sent(modelRef);
        String name = "m-" + modelRef.substring(4, 16);
        Path models = training.modelsFolder();
        Files.createDirectories(models);
        Path local = models.resolve(name + ".brainiac");
        if (!Files.isRegularFile(local)) {
            Files.copy(model, local);
        }
        spec.put("model", name);
        spec.remove("host");
        Map<String, Object> data = new LinkedHashMap<>(Json.object(spec.getOrDefault("data", Map.of())));
        switch (Json.string(data, "kind", "builtin")) {
            case "csv" -> data.put("file", sent(Json.string(data, "file", null)).toString());
            case "idx" -> {
                data.put("images", sent(Json.string(data, "images", null)).toString());
                data.put("labels", sent(Json.string(data, "labels", null)).toString());
            }
            case "images" -> data.put("folder", unzip(sent(Json.string(data, "folder", null))).toString());
            case "builtin" -> {
                // made here
            }
            default -> throw new IllegalArgumentException("unknown data kind " + data.get("kind"));
        }
        spec.put("data", data);
        return training.start(spec);
    }

    /** Unpacks a sent zip next to it, once, refusing entries that would land outside. */
    private static Path unzip(Path zip) throws IOException {
        Path dir = zip.resolveSibling("unzipped");
        Path done = zip.resolveSibling("unzipped.done");
        if (Files.exists(done)) {
            return dir;
        }
        Files.createDirectories(dir);
        long total = 0;
        try (InputStream raw = Files.newInputStream(zip); ZipInputStream in = new ZipInputStream(raw)) {
            for (ZipEntry e; (e = in.getNextEntry()) != null; ) {
                Path out = dir.resolve(e.getName()).normalize();
                if (!out.startsWith(dir)) {
                    throw new IllegalArgumentException("the zip has an entry outside its folder: " + e.getName());
                }
                if (e.isDirectory()) {
                    Files.createDirectories(out);
                    continue;
                }
                Files.createDirectories(out.getParent());
                total += Files.copy(in, out, StandardCopyOption.REPLACE_EXISTING);
                if (total > MAX_UNZIPPED) {
                    throw new IllegalArgumentException("the folder is over 32 GB unpacked");
                }
            }
        }
        Files.writeString(done, "", StandardCharsets.UTF_8);
        return dir;
    }
}
