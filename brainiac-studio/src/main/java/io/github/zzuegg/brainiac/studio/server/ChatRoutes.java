package io.github.zzuegg.brainiac.studio.server;

import com.sun.net.httpserver.HttpExchange;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatListener;
import io.github.zzuegg.brainiac.studio.chat.ChatMessage;
import io.github.zzuegg.brainiac.studio.chat.ChatResult;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
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
 *       [{"role": "user", "text": "..."}], "maxTokens": 4096, "key": "work"}} (the key's name is
 *       optional) answers as a server-sent event
 *       stream: {@code route} (the model chosen), then {@code text} and {@code thinking} pieces,
 *       then {@code done} or {@code error}.
 * </ul>
 */
final class ChatRoutes {

    private static final System.Logger LOG = System.getLogger(ChatRoutes.class.getName());
    private static final int MAX_BODY = 8 << 20;
    private static final int DEFAULT_MAX_TOKENS = 8192;

    private final ChatService chat;

    ChatRoutes(ChatService chat) {
        this.chat = chat;
    }

    void register(StudioServer server) {
        server.get("/api/models", ex -> Exchanges.json(ex, 200,
                chat.describe("refresh=1".equals(ex.getRequestURI().getQuery()))));
        server.post("/api/keys", this::keys);
        server.post("/api/chat", this::chat);
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

    private void chat(HttpExchange ex) throws IOException {
        Map<String, Object> body = Exchanges.body(ex, MAX_BODY);
        String model = Json.string(body, "model", ChatService.AUTO);
        String system = Json.string(body, "system", "");
        String key = Json.string(body, "key", null);
        int maxTokens = (int) Math.min(Integer.MAX_VALUE, Json.integer(body, "maxTokens", DEFAULT_MAX_TOKENS));
        List<ChatMessage> messages = new ArrayList<>();
        for (Object o : Json.array(body.getOrDefault("messages", List.of()))) {
            Map<String, Object> m = Json.object(o);
            messages.add(new ChatMessage(Json.string(m, "role", ""), Json.string(m, "text", "")));
        }
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
                ChatResult r = chat.chat(model, system, messages, maxTokens, key, m -> {
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
