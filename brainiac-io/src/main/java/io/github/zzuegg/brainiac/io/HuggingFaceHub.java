package io.github.zzuegg.brainiac.io;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Downloads model repositories from the Hugging Face Hub, so that gated models (Llama, Gemma)
 * load with the user's read token.
 *
 * <p>The token is looked up in this order, and never printed:
 * <ol>
 * <li>{@code ~/.brainiac/hf_token}, the file Brainiac Studio manages,</li>
 * <li>the {@code HF_TOKEN} (or {@code HUGGINGFACE_HUB_TOKEN}) environment variable,</li>
 * <li>{@code $HF_HOME/token}, else {@code ~/.cache/huggingface/token}, where the
 * {@code huggingface-cli login} keeps it.</li>
 * </ol>
 * Without a token only public repositories download. {@code HF_ENDPOINT} picks another hub.
 *
 * <pre>
 * Path dir = HuggingFaceHub.standard().download("google/gemma-3-1b-it", Path.of("models/gemma-3-1b-it"));
 * Llama model = Llama.load(dir, ...);
 * </pre>
 */
public final class HuggingFaceHub {

    /** The files a model needs to load: weights, configuration and tokenizer. */
    public static final List<String> MODEL_FILES = List.of("*.json", "*.safetensors", "tokenizer.model",
            "*.jinja");

    private final URI endpoint;
    private final String token;
    private final HttpClient http;

    /**
     * A client for a hub.
     *
     * @param endpoint the hub, such as {@code https://huggingface.co}
     * @param token the read token, or {@code null} for public repositories only
     */
    public HuggingFaceHub(URI endpoint, String token) {
        this.endpoint = URI.create(stripSlash(Objects.requireNonNull(endpoint, "endpoint").toString()));
        this.token = token == null || token.isBlank() ? null : token.strip();
        this.http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NEVER)
                .connectTimeout(Duration.ofSeconds(30)).build();
    }

    /** @return a client for {@code HF_ENDPOINT} or huggingface.co, with the {@linkplain #findToken token} found */
    public static HuggingFaceHub standard() {
        String ep = System.getenv("HF_ENDPOINT");
        return new HuggingFaceHub(URI.create(ep == null || ep.isBlank() ? "https://huggingface.co" : ep),
                findToken(Path.of(System.getProperty("user.home")), System.getenv()).orElse(null));
    }

    /**
     * Finds the read token in the order the class describes.
     *
     * @param home the user's home folder
     * @param env the environment variables
     * @return the token, if any
     */
    public static Optional<String> findToken(Path home, Map<String, String> env) {
        Optional<String> t = readToken(home.resolve(".brainiac").resolve("hf_token"));
        if (t.isPresent()) {
            return t;
        }
        for (String name : List.of("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")) {
            String v = env.get(name);
            if (v != null && !v.isBlank()) {
                return Optional.of(v.strip());
            }
        }
        String hfHome = env.get("HF_HOME");
        Path cache = hfHome != null && !hfHome.isBlank() ? Path.of(hfHome)
                : home.resolve(".cache").resolve("huggingface");
        return readToken(cache.resolve("token"));
    }

    private static Optional<String> readToken(Path file) {
        try {
            if (Files.isRegularFile(file)) {
                String s = Files.readString(file).strip();
                return s.isEmpty() ? Optional.empty() : Optional.of(s);
            }
        } catch (IOException e) {
            // unreadable counts as missing
        }
        return Optional.empty();
    }

    /** @return whether a token is set; the token itself is never exposed */
    public boolean hasToken() {
        return token != null;
    }

    /**
     * Lists a repository's files.
     *
     * @param repo such as {@code google/gemma-3-1b-it}
     * @param revision a branch, tag or commit, such as {@code main}
     * @return the paths in the repository
     * @throws IOException when the hub refuses or cannot be reached
     */
    public List<String> files(String repo, String revision) throws IOException {
        URI uri = URI.create(endpoint + "/api/models/" + repo + "/revision/" + encode(revision));
        HttpResponse<String> r = send(request(uri), HttpResponse.BodyHandlers.ofString(), repo);
        List<String> out = new ArrayList<>();
        for (Object s : Json.array(Json.object(Json.parse(r.body())).getOrDefault("siblings", List.of()))) {
            out.add(Json.string(Json.object(s), "rfilename", null));
        }
        return out;
    }

    /**
     * Downloads the files a model needs from the {@code main} branch.
     *
     * @param repo the repository
     * @param dir the folder to write into; files already there are kept
     * @return {@code dir}
     * @throws IOException when the hub refuses or a file cannot be written
     */
    public Path download(String repo, Path dir) throws IOException {
        return download(repo, "main", MODEL_FILES, dir);
    }

    /**
     * Downloads the files matching any of the glob patterns. A file already in {@code dir} is
     * not downloaded again; a partial download is written to a {@code .part} file and moved
     * into place only when complete.
     *
     * @param repo the repository
     * @param revision a branch, tag or commit
     * @param patterns globs such as {@code *.safetensors}, matched against the path in the repository
     * @param dir the folder to write into
     * @return {@code dir}
     * @throws IOException when the hub refuses or a file cannot be written
     */
    public Path download(String repo, String revision, List<String> patterns, Path dir) throws IOException {
        List<PathMatcher> matchers = patterns.stream()
                .map(p -> FileSystems.getDefault().getPathMatcher("glob:" + p)).toList();
        List<String> wanted = new ArrayList<>();
        for (String f : files(repo, revision)) {
            Path p = Path.of(f);
            if (matchers.stream().anyMatch(m -> m.matches(p))) {
                wanted.add(f);
            }
        }
        if (wanted.isEmpty()) {
            throw new IOException("no file of " + repo + " matches " + patterns);
        }
        Files.createDirectories(dir);
        for (String f : wanted) {
            downloadFile(repo, revision, f, dir);
        }
        return dir;
    }

    /**
     * Downloads one file.
     *
     * @param repo the repository
     * @param revision a branch, tag or commit
     * @param file the path in the repository
     * @param dir the folder to write into
     * @return the downloaded file
     * @throws IOException when the hub refuses or the file cannot be written
     */
    public Path downloadFile(String repo, String revision, String file, Path dir) throws IOException {
        Path target = dir.resolve(file).normalize();
        if (!target.startsWith(dir.normalize()) || file.contains("..")) {
            throw new IOException("refusing to write " + file + " outside " + dir);
        }
        if (Files.exists(target)) {
            return target;
        }
        Files.createDirectories(target.getParent());
        URI uri = URI.create(endpoint + "/" + repo + "/resolve/" + encode(revision) + "/" + file);
        Path part = target.resolveSibling(target.getFileName() + ".part");
        HttpResponse<InputStream> r = send(request(uri), HttpResponse.BodyHandlers.ofInputStream(), repo);
        try (InputStream in = r.body()) {
            Files.copy(in, part, StandardCopyOption.REPLACE_EXISTING);
        }
        Files.move(part, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        return target;
    }

    private HttpRequest request(URI uri) {
        HttpRequest.Builder b = HttpRequest.newBuilder(uri).header("User-Agent", "brainiac");
        if (token != null && Objects.equals(uri.getHost(), endpoint.getHost())) {   // never to a CDN
            b.header("Authorization", "Bearer " + token);
        }
        return b.GET().build();
    }

    private <T> HttpResponse<T> send(HttpRequest req, HttpResponse.BodyHandler<T> body, String repo)
            throws IOException {
        HttpResponse<T> r;
        HttpRequest first = req;
        for (int hops = 0; ; hops++) {
            try {
                r = http.send(req, body);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted", e);
            }
            int code = r.statusCode();
            Optional<String> location = r.headers().firstValue("Location");
            if (code / 100 != 3 || location.isEmpty() || hops == 10) {
                break;
            }
            if (r.body() instanceof InputStream in) {
                in.close();
            }
            // follows redirects itself, so the token goes only to the hub and not to its file storage
            req = request(req.uri().resolve(location.get()));
        }
        req = first;
        int code = r.statusCode();
        if (code / 100 == 2) {
            return r;
        }
        if (r.body() instanceof InputStream in) {
            in.close();
        }
        String why = switch (code) {
            case 401 -> token == null ? "it needs a read token (see HuggingFaceHub)" : "the token was refused";
            case 403 -> "access is not granted: accept the model's licence on its Hugging Face page";
            case 404 -> "it was not found";
            default -> "the hub answered " + code;
        };
        throw new IOException("cannot download " + repo + " (" + req.uri().getPath() + "): " + why);
    }

    private static String encode(String s) {
        return URLEncoder.encode(s, StandardCharsets.UTF_8);
    }

    private static String stripSlash(String s) {
        return s.endsWith("/") ? s.substring(0, s.length() - 1) : s;
    }

    @Override
    public String toString() {
        return "HuggingFaceHub[" + endpoint + (token != null ? ", with token" : "") + "]";
    }
}
