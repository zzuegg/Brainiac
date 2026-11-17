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
import java.util.HexFormat;
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
