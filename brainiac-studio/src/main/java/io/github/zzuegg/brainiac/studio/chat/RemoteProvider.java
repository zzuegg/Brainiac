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

    /** Answers that mean "this key can't, another might". */
    private static final Set<Integer> NEXT_KEY = Set.of(401, 403, 429);

    private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15))
            .followRedirects(HttpClient.Redirect.NEVER).build();

    private final Keys keys;
    private final URI base;
    private final Map<String, String> keyErrors = new ConcurrentHashMap<>();

    /** An error answer from the provider. */
    static final class Failure extends IOException {
        private static final long serialVersionUID = 1L;
        final int status;

        Failure(int status, String message) {
            super(message);
            this.status = status;
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

    @Override
    public ChatResult chat(ChatRequest request, ChatListener listener) throws IOException, InterruptedException {
        List<String> labels = new ArrayList<>();
        if (request.key() != null) {
            keys.get(id(), request.key()).orElseThrow(() -> new IOException("no " + name() + " key named "
                    + request.key()));
            labels.add(request.key());
        } else {
            keys.list(id()).forEach(e -> labels.add(e.label()));
        }
        if (labels.isEmpty()) {
            throw new IOException("no API key for " + name() + ": add one under Models, Providers");
        }
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
        for (int i = 0; ; i++) {
            try {
                return chat(request, watched, keys.get(id(), labels.get(i)).orElseThrow());
            } catch (Failure f) {
                if (started[0] || i + 1 == labels.size() || !NEXT_KEY.contains(f.status)) {
                    throw labels.size() == 1 ? f : new IOException(f.getMessage() + " (key " + labels.get(i) + ")", f);
                }
            }
        }
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
        String hint = switch (status) {
            case 401, 403 -> " (check the API key)";
            case 429 -> " (rate limited or out of credit)";
            default -> "";
        };
        return new Failure(status, name() + " answered " + status + hint + (message == null ? "" : ": " + message));
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
