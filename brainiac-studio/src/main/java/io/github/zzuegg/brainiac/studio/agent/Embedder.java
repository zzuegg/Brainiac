package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Encoder;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Turns text into vectors, so passages can be found by meaning: with a sentence-embedding model
 * on this PC (such as BGE, E5, GTE or MiniLM, downloaded from the Hub), or with an
 * OpenAI-compatible embeddings API (OpenAI, Ollama, and the like) set on the settings page.
 */
public final class Embedder {

    /** The most characters of a text that are embedded. */
    static final int TEXT_CHARS = 1500;

    private static final int BATCH = 16;
    private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15)).build();

    private final Services services;
    private final Supplier<List<Path>> folders;
    private volatile Local local;

    /** A loaded local model. */
    private record Local(Path dir, Encoder encoder, PretrainedTokenizer tokenizer, Encoder.Pooling pooling, String prefix) {
    }

    /**
     * Creates the embedder.
     *
     * @param services where an embeddings API is set
     * @param folders where downloaded models are found
     */
    public Embedder(Services services, Supplier<List<Path>> folders) {
        this.services = services;
        this.folders = folders;
    }

    /**
     * Names the model that would embed now, so an index knows whether its vectors still fit.
     *
     * @return the model's name, if there is one
     */
    public Optional<String> name() {
        Optional<String> model = services.get("embeddings", "model");
        if (services.get("embeddings", "url").isPresent() && model.isPresent()) {
            return Optional.of("api:" + model.get());
        }
        if (services.get("embeddings", "source").filter("off"::equals).isPresent()) {
            return Optional.empty();
        }
        return localDir().map(d -> "local:" + d.getFileName());
    }

    /**
     * Embeds texts.
     *
     * @param texts the texts
     * @param query whether they are search queries rather than passages (some models mark them)
     * @return one unit-length vector per text
     */
    public float[][] embed(List<String> texts, boolean query) {
        List<String> cut = texts.stream().map(t -> t.length() > TEXT_CHARS ? t.substring(0, TEXT_CHARS) : t).toList();
        Optional<String> url = services.get("embeddings", "url");
        if (url.isPresent()) {
            return remote(url.get(), services.get("embeddings", "model").orElseThrow(() -> new IllegalArgumentException(
                    "set the embedding model on the settings page")), cut);
        }
        Local l = load();
        float[][] out = new float[cut.size()][];
        for (int i = 0; i < cut.size(); i += BATCH) {
            List<String> batch = new ArrayList<>();
            for (String t : cut.subList(i, Math.min(cut.size(), i + BATCH))) {
                batch.add(l.prefix().isEmpty() ? t : (query ? "query: " : "passage: ") + t);
            }
            float[][] v;
            synchronized (l) {
                v = l.encoder().embed(l.tokenizer(), batch, l.pooling(), true);
            }
            System.arraycopy(v, 0, out, i, v.length);
        }
        return out;
    }

    private Local load() {
        Local l = local;
        Path dir = localDir().orElseThrow(() -> new IllegalArgumentException("there is no embedding model: download "
                + "one, such as BAAI/bge-small-en-v1.5, or set an embeddings API on the settings page"));
        if (l != null && l.dir().equals(dir)) {
            return l;
        }
        synchronized (this) {
            if (local != null && local.dir().equals(dir)) {
                return local;
            }
            String n = dir.getFileName().toString().toLowerCase(Locale.ROOT);
            Encoder e = Encoder.load(dir, Device.cpu(), DType.FLOAT32);
            PretrainedTokenizer t = PretrainedTokenizer.load(dir);
            local = new Local(dir, e, t, n.contains("bge") || n.contains("gte") && !n.contains("multilingual")
                    ? Encoder.Pooling.CLS : Encoder.Pooling.MEAN, n.contains("e5") ? "e5" : "");
            return local;
        }
    }

    /**
     * Finds a sentence-embedding model among the downloaded ones.
     *
     * @return its folder
     */
    Optional<Path> localDir() {
        Optional<String> set = services.get("embeddings", "folder");
        if (set.isPresent()) {
            Path p = Path.of(set.get());
            return Files.isRegularFile(p.resolve("config.json")) ? Optional.of(p) : Optional.empty();
        }
        List<Path> found = new ArrayList<>();
        for (Path root : folders.get()) {
            if (!Files.isDirectory(root)) {
                continue;
            }
            try (Stream<Path> s = Files.walk(root, 4)) {
                s.filter(p -> p.getFileName().toString().equals("config.json")).map(Path::getParent)
                        .filter(Embedder::isEncoder).forEach(found::add);
            } catch (IOException | UncheckedIOException ignored) {
                // an unreadable folder holds nothing usable
            }
        }
        return found.stream().min(Comparator.comparingInt(Embedder::rank).thenComparing(Path::toString));
    }

    // a smaller rank is a likelier sentence-embedding model
    private static int rank(Path dir) {
        String n = dir.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String k : List.of("bge", "e5", "gte", "minilm", "mpnet", "embed", "sentence")) {
            if (n.contains(k)) {
                return List.of("bge", "e5", "gte", "minilm", "mpnet", "embed", "sentence").indexOf(k);
            }
        }
        return 99;
    }

    static boolean isEncoder(Path dir) {
        try {
            Map<String, Object> c = Json.object(Json.parse(Files.readString(dir.resolve("config.json"), StandardCharsets.UTF_8)));
            String type = Json.string(c, "model_type", "");
            boolean encoder = List.of("bert", "roberta", "xlm-roberta", "distilbert", "mpnet").contains(type);
            boolean head = String.valueOf(c.get("architectures")).contains("ForSequenceClassification");
            boolean weights;
            try (Stream<Path> s = Files.list(dir)) {
                weights = s.anyMatch(p -> p.getFileName().toString().endsWith(".safetensors"));
            }
            return encoder && !head && weights && Files.isRegularFile(dir.resolve("tokenizer.json"))
                    && rank(dir) < 99;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    private float[][] remote(String base, String model, List<String> texts) {
        String b = base.endsWith("/") ? base : base + "/";
        float[][] out = new float[texts.size()][];
        for (int i = 0; i < texts.size(); i += 64) {
            List<String> batch = texts.subList(i, Math.min(texts.size(), i + 64));
            HttpRequest.Builder r = HttpRequest.newBuilder(URI.create(b + "embeddings")).timeout(Duration.ofMinutes(2))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(Json.write(Map.of("model", model, "input", batch))));
            services.secret("embeddings").ifPresent(k -> r.header("Authorization", "Bearer " + k));
            try {
                HttpResponse<String> res = HTTP.send(r.build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                if (res.statusCode() / 100 != 2) {
                    throw new IllegalArgumentException("the embeddings API said " + res.statusCode() + ": "
                            + res.body().substring(0, Math.min(200, res.body().length())));
                }
                List<Object> data = Json.array(Json.object(Json.parse(res.body())).get("data"));
                for (Object o : data) {
                    Map<String, Object> d = Json.object(o);
                    List<Object> v = Json.array(d.get("embedding"));
                    float[] f = new float[v.size()];
                    for (int k = 0; k < f.length; k++) {
                        f[k] = ((Number) v.get(k)).floatValue();
                    }
                    out[i + (int) Json.integer(d, "index", 0)] = unit(f);
                }
            } catch (IOException e) {
                throw new UncheckedIOException("cannot reach the embeddings API: " + e.getMessage(), e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new AgentRuns.Stopped();
            }
        }
        return out;
    }

    static float[] unit(float[] v) {
        double s = 0;
        for (float x : v) {
            s += x * x;
        }
        float n = (float) Math.sqrt(s);
        if (n > 0) {
            for (int i = 0; i < v.length; i++) {
                v[i] /= n;
            }
        }
        return v;
    }

    static double cosine(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * Writes a vector compactly, as Base64 of its float32 bytes.
     *
     * @param v the vector
     * @return the text
     */
    static String pack(float[] v) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(v.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putFloat(x);
        }
        return java.util.Base64.getEncoder().encodeToString(b.array());
    }

    /**
     * Reads a vector written by {@link #pack}.
     *
     * @param s the text
     * @return the vector
     */
    static float[] unpack(String s) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.wrap(java.util.Base64.getDecoder().decode(s))
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] v = new float[b.remaining() / 4];
        for (int i = 0; i < v.length; i++) {
            v[i] = b.getFloat();
        }
        return v;
    }
}
