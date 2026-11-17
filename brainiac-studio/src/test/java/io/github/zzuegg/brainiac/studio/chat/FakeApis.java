package io.github.zzuegg.brainiac.studio.chat;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Stand-ins for the Anthropic, OpenAI and Gemini APIs that answer like the real ones. */
final class FakeApis implements AutoCloseable {

    final HttpServer http;
    /** Where outputs and the usage log go when a test has no model folder. */
    final java.nio.file.Path scratch;
    /** The last request body and headers seen per path prefix: a, o, g. */
    final Map<String, String> bodies = new ConcurrentHashMap<>();
    final Map<String, com.sun.net.httpserver.Headers> headers = new ConcurrentHashMap<>();
    volatile int failWith;
    /** A key that is out of credit: Anthropic answers 400, the others 429. */
    volatile String failKey;

    FakeApis() throws IOException {
        http = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        http.createContext("/a/", ex -> handle(ex, "a"));
        http.createContext("/o/", ex -> handle(ex, "o"));
        http.createContext("/g/", ex -> handle(ex, "g"));
        http.createContext("/d/", ex -> handle(ex, "d"));
        http.createContext("/r/", ex -> handle(ex, "r"));
        http.createContext("/h/", ex -> handle(ex, "h"));
        scratch = java.nio.file.Files.createTempDirectory("studio-fake");
        http.start();
    }

    URI base(String prefix) {
        return URI.create("http://127.0.0.1:" + http.getAddress().getPort() + "/" + prefix + "/");
    }

    /** How many calls each path got. */
    final Map<String, Integer> calls = new ConcurrentHashMap<>();

    ChatService service(Keys keys, List<java.nio.file.Path> folders) {
        return ChatService.at(keys, folders, Map.of("anthropic", base("a"), "openai", base("o").resolve("v1/"),
                "google", base("g"), "deepseek", base("d"), "openrouter", base("r").resolve("api/v1/"),
                "huggingface", base("h").resolve("v1/")),
                folders.isEmpty() ? scratch.resolve("outputs") : folders.getFirst().resolve("outputs"));
    }

    private static String b64(String s) {
        return java.util.Base64.getEncoder().encodeToString(s.getBytes(StandardCharsets.UTF_8));
    }

    private void handle(HttpExchange ex, String p) throws IOException {
        String path = ex.getRequestURI().getPath().substring(2);
        bodies.put(p + " " + ex.getRequestMethod(),
                new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
        headers.put(p, ex.getRequestHeaders());
        calls.merge(p + " " + path, 1, Integer::sum);
        String auth = String.valueOf(ex.getRequestHeaders().getFirst("x-api-key")) + " "
                + ex.getRequestHeaders().getFirst("Authorization") + " " + ex.getRequestHeaders().getFirst("x-goog-api-key");
        if (failKey != null && auth.contains(failKey)) {
            if (p.equals("a")) {   // what Anthropic really answers: a 400, not a 429
                send(ex, 400, "application/json", "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\","
                        + "\"message\":\"Your credit balance is too low to access the Anthropic API.\"}}");
            } else {
                send(ex, 429, "application/json", "{\"error\":{\"message\":\"credit balance too low\"}}");
            }
            return;
        }
        if (failWith != 0) {
            send(ex, failWith, "application/json", "{\"type\":\"error\",\"error\":{\"type\":\"x\",\"message\":\"invalid x-api-key\"}}");
            return;
        }
        switch (p + " " + path) {
            case "a /v1/models" -> send(ex, 200, "application/json", """
                    {"data":[{"id":"claude-opus-5-5","display_name":"Claude Opus 5.5","created_at":"2026-09-01T00:00:00Z",
                      "max_input_tokens":1000000,"capabilities":{"image_input":{"supported":true},
                      "thinking":{"supported":true,"types":{"adaptive":{"supported":true}}}}},
                     {"id":"claude-haiku-5-5","display_name":"Claude Haiku 5.5","created_at":"2026-09-01T00:00:00Z"}],
                     "has_more":false}""");
            case "a /v1/messages" -> send(ex, 200, "text/event-stream", """
                    event: message_start
                    data: {"type":"message_start","message":{"usage":{"input_tokens":12}}}

                    event: ping
                    data: {"type":"ping"}

                    event: content_block_delta
                    data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}

                    event: content_block_delta
                    data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hmm."}}

                    event: content_block_delta
                    data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}

                    event: content_block_delta
                    data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" there"}}

                    event: message_delta
                    data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

                    event: message_stop
                    data: {"type":"message_stop"}

                    """);
            case "o /v1/models" -> send(ex, 200, "application/json", """
                    {"object":"list","data":[{"id":"gpt-5","created":1754000000},{"id":"gpt-5-mini","created":1754000001},
                     {"id":"tts-1","created":1},{"id":"whisper-1","created":1},{"id":"text-embedding-3-large","created":1},
                     {"id":"gpt-image-1","created":2},{"id":"dall-e-3","created":1},{"id":"gpt-4o-mini-tts","created":4},{"id":"gpt-4o-realtime-preview","created":3}]}""");
            case "o /v1/chat/completions" -> send(ex, 200, "text/event-stream", """
                    data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

                    data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}

                    data: {"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"length"}]}

                    data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2}}

                    data: [DONE]

                    """);
            case "g /v1beta/models" -> send(ex, 200, "application/json", """
                    {"models":[{"name":"models/gemini-2.5-pro","displayName":"Gemini 2.5 Pro","inputTokenLimit":1048576,
                       "supportedGenerationMethods":["generateContent","countTokens"]},
                     {"name":"models/gemini-3-pro","displayName":"Gemini 3 Pro","supportedGenerationMethods":["generateContent"]},
                     {"name":"models/imagen-4.0-generate-001","displayName":"Imagen 4","supportedGenerationMethods":["predict"]},
                     {"name":"models/gemini-2.5-flash-preview-tts","displayName":"Gemini 2.5 Flash TTS",
                       "supportedGenerationMethods":["generateContent"]},
                     {"name":"models/veo-3.0-generate-001","displayName":"Veo 3","supportedGenerationMethods":["predictLongRunning"]},
                     {"name":"models/gemini-embedding-001","supportedGenerationMethods":["embedContent"]}]}""");
            case "g /v1beta/models/gemini-3-pro:streamGenerateContent" -> send(ex, 200, "text/event-stream", """
                    data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Planning","thought":true}]}}]}

                    data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Bonjour"}]}}]}

                    data: {"candidates":[{"content":{"role":"model","parts":[{"text":"."}]},"finishReason":"SAFETY"}],
                    data:  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"thoughtsTokenCount":4}}

                    """);
            case "o /v1/images/generations" -> {
                int n = ((Number) io.github.zzuegg.brainiac.io.Json.object(io.github.zzuegg.brainiac.io.Json.parse(
                        bodies.get("o POST"))).get("n")).intValue();
                StringBuilder data = new StringBuilder();
                for (int i = 0; i < n; i++) {
                    data.append(i == 0 ? "" : ",").append("{\"b64_json\":\"").append(b64("PNG" + i)).append("\"}");
                }
                send(ex, 200, "application/json", "{\"created\":1,\"data\":[" + data + "]}");
            }
            case "o /v1/audio/speech" -> send(ex, 200, "audio/mpeg", "ID3 fake mp3");
            case "g /v1beta/models/imagen-4.0-generate-001:predict" -> send(ex, 200, "application/json",
                    "{\"predictions\":[{\"bytesBase64Encoded\":\"" + b64("IMG-A") + "\",\"mimeType\":\"image/png\"},"
                    + "{\"bytesBase64Encoded\":\"" + b64("IMG-B") + "\",\"mimeType\":\"image/png\"}]}");
            case "g /v1beta/models/gemini-2.5-flash-preview-tts:generateContent" -> send(ex, 200, "application/json",
                    "{\"candidates\":[{\"content\":{\"parts\":[{\"inlineData\":{\"mimeType\":"
                    + "\"audio/L16;codec=pcm;rate=24000\",\"data\":\"" + b64("abcd") + "\"}}]}}]}");
            case "d /models" -> send(ex, 200, "application/json", """
                    {"object":"list","data":[{"id":"deepseek-chat","object":"model","owned_by":"deepseek"},
                     {"id":"deepseek-reasoner","object":"model","owned_by":"deepseek"}]}""");
            case "d /chat/completions" -> send(ex, 200, "text/event-stream", """
                    data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Ni"}}]}

                    data: {"choices":[{"index":0,"delta":{"content":" hao"},"finish_reason":"stop"}]}

                    data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}

                    data: [DONE]

                    """);
            case "d /user/balance" -> send(ex, 200, "application/json", """
                    {"is_available":true,"balance_infos":[{"currency":"CNY","total_balance":"110.00",
                     "granted_balance":"10.00","topped_up_balance":"100.00"}]}""");
            case "r /api/v1/models" -> send(ex, 200, "application/json", """
                    {"data":[{"id":"qwen/qwen3-vl-235b","name":"Qwen3 VL","created":1750000000,"context_length":262144},
                     {"id":"openai/text-embedding-3-small","name":"Embedding","created":1}]}""");
            case "r /api/v1/key" -> send(ex, 200, "application/json", """
                    {"data":{"label":"sk-or-v1-abc","limit":20,"limit_remaining":12.5,"usage":7.5}}""");
            case "h /v1/models" -> send(ex, 200, "application/json", """
                    {"object":"list","data":[
                     {"id":"openai/gpt-oss-120b","object":"model","created":1754000000,"owned_by":"openai","providers":[
                       {"provider":"groq","status":"live","context_length":131072,"pricing":{"input":0.15,"output":0.6}},
                       {"provider":"cerebras","status":"live","context_length":131072,"pricing":{"input":0.25,"output":0.69}},
                       {"provider":"novita","status":"offline","pricing":{"input":0.05,"output":0.1}}]},
                     {"id":"meta-llama/Llama-3.3-70B-Instruct","object":"model","created":1733000000,"owned_by":"meta-llama",
                      "providers":[{"provider":"together","status":"live","context_length":131072}]},
                     {"id":"old/retired-model","object":"model","created":1,"providers":[{"provider":"x","status":"deprecated"}]}]}""");
            case "h /v1/chat/completions" -> send(ex, 200, "text/event-stream", """
                    data: {"choices":[{"index":0,"delta":{"content":"Hello from the hub"},"finish_reason":"stop"}]}

                    data: [DONE]

                    """);
            default -> send(ex, 404, "application/json", "{\"error\":{\"message\":\"no route " + path + "\"}}");
        }
    }

    private static void send(HttpExchange ex, int status, String type, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type);
        ex.sendResponseHeaders(status, b.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(b);
        }
    }

    @Override
    public void close() {
        http.stop(0);
    }
}
