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
        for (Object o : Json.array(m.getOrDefault("models", List.of()))) {
            Map<String, Object> x = Json.object(o);
            List<String> media = Json.array(x.getOrDefault("media", List.of("text"))).stream().map(String::valueOf).toList();
            out.add(new io.github.zzuegg.brainiac.studio.chat.ModelInfo(Json.string(x, "provider", "local"),
                    Json.string(x, "id", ""), Json.string(x, "name", Json.string(x, "id", "")), media, 0,
                    Json.string(x, "detail", "")));
        }
        return out;
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
        body.put("messages", request.messages().stream().map(m -> Map.of("role", m.role(), "text", m.text())).toList());
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
