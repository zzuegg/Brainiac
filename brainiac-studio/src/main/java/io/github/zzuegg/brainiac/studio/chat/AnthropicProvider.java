package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.net.URI;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Claude models through the Anthropic Messages API. */
final class AnthropicProvider extends RemoteProvider {

    static final URI BASE = URI.create("https://api.anthropic.com/");
    static final String VERSION = "2026-10-21";
    static final String DEFAULT_MODEL = "claude-opus-5-5";
    // On Fable models a policy decline is retried on a fallback model inside the same call.
    private static final String FALLBACK_BETA = "server-side-fallback-2026-07-01";

    AnthropicProvider(Keys keys, URI base) {
        super(keys, base);
    }

    @Override
    public String id() {
        return "anthropic";
    }

    @Override
    public String name() {
        return "Anthropic";
    }

    @Override
    List<String> auth(String key) {
        return List.of("x-api-key", key, "anthropic-version", VERSION);
    }

    @Override
    List<ModelInfo> models(String key) throws IOException, InterruptedException {
        List<ModelInfo> out = new ArrayList<>();
        String after = null;
        do {
            Map<String, Object> page = getJson(uri("v1/models?limit=1000"
                    + (after == null ? "" : "&after_id=" + after)), key);
            for (Object o : arr(page.get("data"))) {
                Map<String, Object> m = obj(o);
                String id = Json.string(m, "id", null);
                if (id == null) {
                    continue;
                }
                boolean vision = obj(obj(m.get("capabilities")).get("image_input")).get("supported") != Boolean.FALSE;
                long context = num(m.get("max_input_tokens"));
                out.add(new ModelInfo(id(), id, Json.string(m, "display_name", id),
                        vision ? List.of("text", "vision") : List.of("text"), created(m.get("created_at")),
                        ModelInfo.context(context)));
            }
            after = Boolean.TRUE.equals(page.get("has_more")) ? Json.string(page, "last_id", null) : null;
        } while (after != null);
        return out;
    }

    @Override
    public ModelInfo preferred(List<ModelInfo> models) {
        return models.stream().filter(m -> m.id().equals(DEFAULT_MODEL)).findFirst()
                .orElseGet(() -> super.preferred(models));
    }

    @Override
    ChatResult chat(ChatRequest request, ChatListener listener, String key) throws IOException, InterruptedException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", request.model());
        body.put("max_tokens", request.maxTokens());
        body.put("stream", true);
        if (!request.system().isBlank()) {
            body.put("system", request.system());
        }
        List<Object> messages = new ArrayList<>();
        for (ChatMessage m : request.messages()) {
            messages.add(Map.of("role", m.role(), "content", m.text()));
        }
        body.put("messages", messages);
        List<String> headers = List.of();
        if (request.model().startsWith("claude-fable-")) {
            body.put("fallbacks", "default");
            headers = List.of("anthropic-beta", FALLBACK_BETA);
        }
        long[] usage = {-1, -1};
        String[] stop = {"end"};
        stream(uri("v1/messages"), key, body, headers, (event, data) -> {
            Map<String, Object> e = Json.object(Json.parse(data));
            switch (Json.string(e, "type", event)) {
                case "message_start" -> usage[0] = num(obj(obj(e.get("message")).get("usage")).get("input_tokens"));
                case "content_block_delta" -> {
                    Map<String, Object> d = obj(e.get("delta"));
                    switch (Json.string(d, "type", "")) {
                        case "text_delta" -> listener.text(Json.string(d, "text", ""));
                        case "thinking_delta" -> listener.thinking(Json.string(d, "thinking", ""));
                        default -> { }
                    }
                }
                case "message_delta" -> {
                    stop[0] = stopReason(Json.string(obj(e.get("delta")), "stop_reason", "end_turn"));
                    long out = num(obj(e.get("usage")).get("output_tokens"));
                    if (out >= 0) {
                        usage[1] = out;
                    }
                }
                case "error" -> throw new IOException("Anthropic stopped the answer: "
                        + Json.string(obj(e.get("error")), "message", data));
                case "message_stop" -> {
                    return false;
                }
                default -> { }
            }
            return true;
        });
        return new ChatResult(stop[0], usage[0], usage[1]);
    }

    private static String stopReason(String r) {
        return switch (r) {
            case "end_turn", "stop_sequence" -> "end";
            case "max_tokens" -> "length";
            default -> r;
        };
    }

    private static long created(Object at) {
        try {
            return at instanceof String s ? Instant.parse(s).getEpochSecond() : 0;
        } catch (DateTimeParseException e) {
            return 0;
        }
    }
}
