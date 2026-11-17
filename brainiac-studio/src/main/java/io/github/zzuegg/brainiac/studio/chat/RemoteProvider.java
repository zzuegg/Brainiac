package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The shared part of providers reached over HTTPS with the user's keys. With several keys, the
 * model list joins what each key can see, and a chat that a key may not make (refused, out of
 * credit, rate limited) moves on to the next key before any text has arrived.
 */
abstract class RemoteProvider implements ChatProvider {

    /** The name of the key the last call on this thread used, for the usage log. */
    static final ThreadLocal<String> LAST_KEY = new ThreadLocal<>();

    /** Answers that mean another key may do better: refused, unpaid, or rate limited. */
    private static final Set<Integer> NEXT_KEY = Set.of(401, 402, 403, 429);
    /** Words that mark an empty balance, which Anthropic and others answer with a plain 400. */
    private static final java.util.regex.Pattern NO_CREDIT = java.util.regex.Pattern.compile(
            "credit balance|insufficient (balance|credit|quota|funds)|exceeded your current quota|out of credit",
            java.util.regex.Pattern.CASE_INSENSITIVE);

    private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15))
            .followRedirects(HttpClient.Redirect.NEVER).build();

    private final Keys keys;
    private final URI base;
    private final Map<String, String> keyErrors = new ConcurrentHashMap<>();

    /** An error answer from the provider. */
    static final class Failure extends IOException {
        private static final long serialVersionUID = 1L;
        final int status;
        /** Whether the next key should be tried. */
        final boolean nextKey;

        Failure(int status, String message) {
            this(status, message, NEXT_KEY.contains(status));
        }

        Failure(int status, String message, boolean nextKey) {
            super(message);
            this.status = status;
            this.nextKey = nextKey;
        }
    }

    RemoteProvider(Keys keys, URI base) {
        this.keys = keys;
        this.base = base.toString().endsWith("/") ? base : URI.create(base + "/");
    }

    @Override
    public boolean ready() {
        return keys.has(id());
    }

    /** Lists the models one key can use. */
    abstract List<ModelInfo> models(String key) throws IOException, InterruptedException;

    /** Writes an answer with one key. */
    abstract ChatResult chat(ChatRequest request, ChatListener listener, String key)
            throws IOException, InterruptedException;

    @Override
    public List<ModelInfo> models() throws IOException, InterruptedException {
        Map<String, ModelInfo> all = new LinkedHashMap<>();
        IOException first = null;
        List<Keys.Entry> entries = keys.list(id());
        for (Keys.Entry e : entries) {
            try {
                for (ModelInfo m : models(keys.get(id(), e.label()).orElseThrow())) {
                    all.putIfAbsent(m.id(), m);
                }
                keyErrors.remove(e.label());
            } catch (IOException ex) {
                keyErrors.put(e.label(), String.valueOf(ex.getMessage()));
                first = first == null ? ex : first;
            }
        }
        if (all.isEmpty() && first != null) {
            throw first;
        }
        return new ArrayList<>(all.values());
    }

    /**
     * Describes the keys, with the error each one got the last time models were listed.
     *
     * @return one map per key: label, source, hint and maybe error
     */
    List<Map<String, Object>> keyStatus() {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Keys.Entry e : keys.list(id())) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("label", e.label());
            m.put("source", e.source());
            m.put("hint", e.hint());
            String err = keyErrors.get(e.label());
            if (err != null) {
                m.put("error", err);
            }
            out.add(m);
        }
        return out;
    }

    /** One call made with one key. */
    @FunctionalInterface
    interface KeyCall<T> {
        T call(String key) throws IOException, InterruptedException;
    }

    /**
     * Makes a call with the named key, or with each key in turn until one is not refused, rate
     * limited or out of credit.
     *
     * @param label the key's name, or null for any
     * @param retry false once the call has passed something on, so it is not made twice
     */
    <T> T withKeys(String label, java.util.function.BooleanSupplier retry, KeyCall<T> call)
            throws IOException, InterruptedException {
        List<String> labels = new ArrayList<>();
        if (label != null) {
            keys.get(id(), label).orElseThrow(() -> new IOException("no " + name() + " key named " + label));
            labels.add(label);
        } else {
            keys.list(id()).forEach(e -> labels.add(e.label()));
        }
        if (labels.isEmpty()) {
            throw new IOException("no API key for " + name() + ": add one under Models, Providers");
        }
        for (int i = 0; ; i++) {
            try {
                LAST_KEY.set(labels.get(i));
                return call.call(keys.get(id(), labels.get(i)).orElseThrow());
            } catch (Failure f) {
                if (!retry.getAsBoolean() || i + 1 == labels.size() || !f.nextKey) {
                    throw labels.size() == 1 ? f : new IOException(f.getMessage() + " (key " + labels.get(i) + ")", f);
                }
            }
        }
    }

    @Override
    public ChatResult chat(ChatRequest request, ChatListener listener) throws IOException, InterruptedException {
        boolean[] started = {false};
        ChatListener watched = new ChatListener() {
            @Override
            public void text(String text) throws IOException {
                started[0] = true;
                listener.text(text);
            }

            @Override
            public void thinking(String text) throws IOException {
                started[0] = true;
                listener.thinking(text);
            }
        };
        return withKeys(request.key(), () -> !started[0], key -> chat(request, watched, key));
    }

    /**
     * Draws pictures.
     *
     * @param request what to draw
     * @return the pictures
     * @throws IOException if the provider fails or has no image models
     * @throws InterruptedException if interrupted while waiting
     */
    public List<Media> images(ImageRequest request) throws IOException, InterruptedException {
        return withKeys(request.key(), () -> true, key -> images(request, key));
    }

    /**
     * Speaks text.
     *
     * @param request what to say
     * @return the audio
     * @throws IOException if the provider fails or has no speech models
     * @throws InterruptedException if interrupted while waiting
     */
    public Media speech(SpeechRequest request) throws IOException, InterruptedException {
        return withKeys(request.key(), () -> true, key -> speech(request, key));
    }

    /** Draws pictures with one key. */
    List<Media> images(ImageRequest request, String key) throws IOException, InterruptedException {
        throw new IOException(name() + " has no image models here");
    }

    /** Speaks with one key. */
    Media speech(SpeechRequest request, String key) throws IOException, InterruptedException {
        throw new IOException(name() + " has no speech models here");
    }

    /** A short line on what the provider offers, shown with it. */
    String note() {
        return "";
    }

    /** Whether {@link #balance} reads anything at this provider. */
    boolean reportsBalance() {
        return false;
    }

    /**
     * Reads what is left on a key, for providers that report it.
     *
     * @param key the key
     * @return the balance, or null if the provider does not report one
     * @throws IOException if the provider fails
     * @throws InterruptedException if interrupted while waiting
     */
    Balance balance(String key) throws IOException, InterruptedException {
        return null;
    }

    /**
     * Reads the balance of each key.
     *
     * @return one map per key: label, and balance or error, or nothing when the provider reports none
     */
    List<Map<String, Object>> balances() throws InterruptedException {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Keys.Entry e : keys.list(id())) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("label", e.label());
            try {
                Balance b = balance(keys.get(id(), e.label()).orElseThrow());
                if (b == null) {
                    return List.of();
                }
                m.put("balance", b.toJson());
            } catch (IOException | RuntimeException ex) {
                m.put("error", String.valueOf(ex.getMessage()));
            }
            out.add(m);
        }
        return out;
    }

    /**
     * Says what a speech model offers.
     *
     * @param model the provider's own model id
     * @return its voices and settings
     */
    SpeechOptions speechOptions(String model) {
        return new SpeechOptions(List.of(), false, false, List.of());
    }

    /** The headers that carry the key, as name, value pairs. */
    abstract List<String> auth(String key);

    URI uri(String path) {
        return base.resolve(path);
    }

    private HttpRequest.Builder request(URI uri, String key) {
        HttpRequest.Builder b = HttpRequest.newBuilder(uri).timeout(Duration.ofMinutes(10));
        List<String> h = auth(key);
        for (int i = 0; i < h.size(); i += 2) {
            b.header(h.get(i), h.get(i + 1));
        }
        return b;
    }

    Map<String, Object> getJson(URI uri, String key) throws IOException, InterruptedException {
        HttpResponse<String> r = HTTP.send(request(uri, key).timeout(Duration.ofSeconds(30)).GET().build(),
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (r.statusCode() / 100 != 2) {
            throw failure(r.statusCode(), r.body());
        }
        return Json.object(Json.parse(r.body()));
    }

    Map<String, Object> postJson(URI uri, String key, Map<String, Object> body)
            throws IOException, InterruptedException {
        HttpRequest req = request(uri, key).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body), StandardCharsets.UTF_8)).build();
        HttpResponse<String> r = HTTP.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (r.statusCode() / 100 != 2) {
            throw failure(r.statusCode(), r.body());
        }
        return Json.object(Json.parse(r.body()));
    }

    Media postForBytes(URI uri, String key, Map<String, Object> body) throws IOException, InterruptedException {
        HttpRequest req = request(uri, key).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body), StandardCharsets.UTF_8)).build();
        HttpResponse<byte[]> r = HTTP.send(req, HttpResponse.BodyHandlers.ofByteArray());
        if (r.statusCode() / 100 != 2) {
            throw failure(r.statusCode(), new String(r.body(), StandardCharsets.UTF_8));
        }
        String type = r.headers().firstValue("Content-Type").orElse("application/octet-stream");
        int semi = type.indexOf(';');
        return new Media(r.body(), (semi < 0 ? type : type.substring(0, semi)).trim());
    }

    void stream(URI uri, String key, Map<String, Object> body, Sse.Handler handler)
            throws IOException, InterruptedException {
        stream(uri, key, body, List.of(), handler);
    }

    void stream(URI uri, String key, Map<String, Object> body, List<String> headers, Sse.Handler handler)
            throws IOException, InterruptedException {
        HttpRequest.Builder b = request(uri, key);
        for (int i = 0; i < headers.size(); i += 2) {
            b.header(headers.get(i), headers.get(i + 1));
        }
        HttpRequest req = b.header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body), StandardCharsets.UTF_8)).build();
        HttpResponse<InputStream> r = HTTP.send(req, HttpResponse.BodyHandlers.ofInputStream());
        try (InputStream in = r.body()) {
            if (r.statusCode() / 100 != 2) {
                throw failure(r.statusCode(), new String(in.readNBytes(64 * 1024), StandardCharsets.UTF_8));
            }
            Sse.read(in, handler);
        }
    }

    /** Turns an error answer into a short message: the provider's own words when it gives them. */
    Failure failure(int status, String body) {
        String message = null;
        try {
            Object e = Json.object(Json.parse(body)).get("error");
            if (e instanceof Map<?, ?> m && m.get("message") instanceof String s) {
                message = s;
            } else if (e instanceof String s) {
                message = s;
            }
        } catch (IllegalArgumentException notJson) {
            // keep the status alone
        }
        boolean broke = message != null && NO_CREDIT.matcher(message).find();
        String hint = switch (status) {
            case 401, 403 -> " (check the API key)";
            case 402 -> " (out of credit)";
            case 429 -> " (rate limited or out of credit)";
            default -> broke ? " (out of credit)" : "";
        };
        return new Failure(status, name() + " answered " + status + hint + (message == null ? "" : ": " + message),
                broke || NEXT_KEY.contains(status));
    }

    static Map<String, Object> obj(Object o) {
        return o instanceof Map<?, ?> ? Json.object(o) : Map.of();
    }

    static List<Object> arr(Object o) {
        return o instanceof List<?> ? Json.array(o) : List.of();
    }

    static long num(Object o) {
        return o instanceof Number n ? n.longValue() : -1;
    }
}
