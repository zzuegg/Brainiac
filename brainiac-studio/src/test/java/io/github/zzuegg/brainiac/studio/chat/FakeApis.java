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
    /** The last request body and headers seen per path prefix: a, o, g. */
    final Map<String, String> bodies = new ConcurrentHashMap<>();
    final Map<String, com.sun.net.httpserver.Headers> headers = new ConcurrentHashMap<>();
    volatile int failWith;
    /** A key that gets 429, as if out of credit. */
    volatile String failKey;

    FakeApis() throws IOException {
        http = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        http.createContext("/a/", ex -> handle(ex, "a"));
        http.createContext("/o/", ex -> handle(ex, "o"));
        http.createContext("/g/", ex -> handle(ex, "g"));
        http.start();
    }

    URI base(String prefix) {
        return URI.create("http://127.0.0.1:" + http.getAddress().getPort() + "/" + prefix + "/");
    }

    ChatService service(Keys keys, List<java.nio.file.Path> folders) {
        return ChatService.at(keys, folders, base("a"), base("o"), base("g"));
    }

    private void handle(HttpExchange ex, String p) throws IOException {
        String path = ex.getRequestURI().getPath().substring(2);
        bodies.put(p + " " + ex.getRequestMethod(),
                new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
        headers.put(p, ex.getRequestHeaders());
        String auth = String.valueOf(ex.getRequestHeaders().getFirst("x-api-key")) + " "
                + ex.getRequestHeaders().getFirst("Authorization") + " " + ex.getRequestHeaders().getFirst("x-goog-api-key");
        if (failKey != null && auth.contains(failKey)) {
            send(ex, 429, "application/json", "{\"error\":{\"message\":\"credit balance too low\"}}");
            return;
        }
        if (failWith != 0) {
            send(ex, failWith, "application/json", "{\"type\":\"error\",\"error\":{\"type\":\"x\",\"message\":\"invalid x-api-key\"}}");
            return;
        }
        switch (p + " " + path) {
            case "a /v1/models" -> send(ex, 200, "application/json", """
                    {"data":[{"id":"claude-opus-5-5","display_name":"Claude Opus 5.5","created_at":"2026-09-01T00:00:00Z",
                      "max_input_tokens":1000000,"capabilities":{"image_input":{"supported":true}}},
                     {"id":"claude-haiku-5-5","display_name":"Claude Haiku 5.5","created_at":"2026-09-01T00:00:00Z"}],
                     "has_more":false}""");
            case "a /v1/messages" -> send(ex, 200, "text/event-stream", """
                    event: message_start
                    data: {"type":"message_start","message":{"usage":{"input_tokens":12}}}

                    event: ping
                    data: {"type":"ping"}

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
                     {"id":"gpt-image-1","created":2},{"id":"gpt-4o-realtime-preview","created":3}]}""");
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
                     {"name":"models/veo-3.0-generate-001","displayName":"Veo 3","supportedGenerationMethods":["predictLongRunning"]},
                     {"name":"models/gemini-embedding-001","supportedGenerationMethods":["embedContent"]}]}""");
            case "g /v1beta/models/gemini-3-pro:streamGenerateContent" -> send(ex, 200, "text/event-stream", """
                    data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Planning","thought":true}]}}]}

                    data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Bonjour"}]}}]}

                    data: {"candidates":[{"content":{"role":"model","parts":[{"text":"."}]},"finishReason":"SAFETY"}],
                    data:  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"thoughtsTokenCount":4}}

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
