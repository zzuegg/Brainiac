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
    /** Models that take adaptive thinking, from the model list; their reasoning is asked for as a summary to show. */
    private final java.util.Set<String> adaptive = java.util.concurrent.ConcurrentHashMap.newKeySet();
    private volatile boolean listed;
    /** Before the list is read: the models from Claude 4.6 on take adaptive thinking. */
    private static final java.util.regex.Pattern ADAPTIVE = java.util.regex.Pattern.compile(
            "claude-(opus|sonnet)-4-[6-9].*|claude-[a-z]+-[5-9].*");

    AnthropicProvider(Keys keys, URI base) {
        super(keys, base);
    }

    @Override
    public String id() {
        return "anthropic";
    }

    @Override
    String note() {
        return "Claude models: text, vision and visible thinking";
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
                if (Boolean.TRUE.equals(obj(obj(obj(obj(m.get("capabilities")).get("thinking")).get("types"))
                        .get("adaptive")).get("supported"))) {
                    adaptive.add(id);
                }
                long context = num(m.get("max_input_tokens"));
                out.add(new ModelInfo(id(), id, Json.string(m, "display_name", id),
                        vision ? List.of("text", "vision") : List.of("text"), created(m.get("created_at")),
                        ModelInfo.context(context)));
            }
            after = Boolean.TRUE.equals(page.get("has_more")) ? Json.string(page, "last_id", null) : null;
        } while (after != null);
        listed = true;
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
        // no temperature: the current API version rejects sampling parameters
        if (!request.system().isBlank()) {
            body.put("system", request.system());
        }
        List<Object> messages = new ArrayList<>();
        boolean tools = !request.tools().isEmpty();
        for (ToolCalls.Turn t : tools ? ToolCalls.turns(request.messages()) : CompatibleProvider.plain(request.messages())) {
            ChatMessage m = t.source();
            List<Object> parts = new ArrayList<>();
            if (!t.calls().isEmpty()) {
                if (!t.text().isEmpty()) {
                    parts.add(Map.of("type", "text", "text", t.text()));
                }
                for (ToolCalls.Call c : t.calls()) {
                    parts.add(Map.of("type", "tool_use", "id", c.id(), "name", c.name(), "input", c.arguments()));
                }
                messages.add(Map.of("role", "assistant", "content", parts));
                continue;
            }
            for (ToolCalls.Result r : t.results()) {
                parts.add(Map.of("type", "tool_result", "tool_use_id", r.id(), "content", r.text()));
            }
            String text = t.results().isEmpty() ? m.text() : t.text();
            if (parts.isEmpty() && m.images().isEmpty()) {
                messages.add(Map.of("role", m.role(), "content", text));
                continue;
            }
            for (Media image : m.images()) {
                parts.add(Map.of("type", "image", "source", Map.of("type", "base64", "media_type", image.mime(),
                        "data", ChatMessage.base64(image))));
            }
            if (!text.isBlank()) {
                parts.add(Map.of("type", "text", "text", text));
            }
            messages.add(Map.of("role", m.role(), "content", parts));
        }
        body.put("messages", messages);
        if (tools) {
            List<Map<String, Object>> defs = new ArrayList<>(request.tools().stream().map(ToolCalls::anthropic).toList());
            // the tools and instructions stay the same through a run: cache them, and the conversation so far
            Map<String, Object> last = new LinkedHashMap<>(defs.getLast());
            last.put("cache_control", Map.of("type", "ephemeral"));
            defs.set(defs.size() - 1, last);
            body.put("tools", defs);
            body.put("cache_control", Map.of("type", "ephemeral"));
        }
        // thinking with tools needs its signed blocks sent back, which a conversation kept as text has not
        if (!tools && (listed ? adaptive.contains(request.model()) : ADAPTIVE.matcher(request.model()).matches())) {
            // the default display leaves thinking blocks empty; the chat shows a summary instead
            body.put("thinking", Map.of("type", "adaptive", "display", "summarized"));
        }
        if (request.format().schema() != null) {
            // structured output; a plain JSON answer has no setting here and is asked for in the instructions
            body.put("output_config", Map.of("format",
                    Map.of("type", "json_schema", "schema", request.format().closedSchema())));
        }
        List<String> headers = List.of();
        if (request.model().startsWith("claude-fable-")) {
            body.put("fallbacks", "default");
            headers = List.of("anthropic-beta", FALLBACK_BETA);
        }
        long[] usage = {-1, -1};
        String[] stop = {"end"};
        ToolCalls.Pending[] call = {null};
        stream(uri("v1/messages"), key, body, headers, (event, data) -> {
            Map<String, Object> e = Json.object(Json.parse(data));
            switch (Json.string(e, "type", event)) {
                case "content_block_start" -> {
                    Map<String, Object> b = obj(e.get("content_block"));
                    if ("tool_use".equals(Json.string(b, "type", ""))) {
                        call[0] = new ToolCalls.Pending();
                        call[0].id = Json.string(b, "id", null);
                        call[0].name = Json.string(b, "name", null);
                    }
                }
                case "content_block_stop" -> {
                    if (call[0] != null) {
                        String h = call[0].hermes();
                        call[0] = null;
                        if (!h.isEmpty()) {
                            listener.text("\n" + h);
                        }
                    }
                }
                case "message_start" -> {
                    // cached input is counted apart; all of it was read
                    Map<String, Object> u = obj(obj(e.get("message")).get("usage"));
                    long in = num(u.get("input_tokens"));
                    usage[0] = in < 0 ? in : in + Math.max(0, num(u.get("cache_read_input_tokens")))
                            + Math.max(0, num(u.get("cache_creation_input_tokens")));
                }
                case "content_block_delta" -> {
                    Map<String, Object> d = obj(e.get("delta"));
                    switch (Json.string(d, "type", "")) {
                        case "text_delta" -> listener.text(Json.string(d, "text", ""));
                        case "input_json_delta" -> {
                            if (call[0] != null) {
                                call[0].arguments.append(Json.string(d, "partial_json", ""));
                            }
                        }
                        case "thinking_delta" -> {
                            String t = Json.string(d, "thinking", "");
                            if (!t.isEmpty()) {
                                listener.thinking(t);
                            }
                        }
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
            case "end_turn", "stop_sequence", "tool_use" -> "end";
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

    @Override
    public boolean nativeTools(String model) {
        return true;
    }
}
