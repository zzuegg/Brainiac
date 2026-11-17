package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.server.CapabilityRegistry;
import io.github.zzuegg.brainiac.studio.server.StudioServer;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The chat endpoints, end to end against stand-in provider APIs. */
class ChatApiTest {

    private static final HttpClient HTTP = HttpClient.newHttpClient();

    @TempDir
    Path dir;
    private FakeApis apis;
    private StudioServer server;

    @BeforeEach
    void start() throws Exception {
        apis = new FakeApis();
        Keys keys = new Keys(dir.resolve("keys.properties"), Map.of());
        server = StudioServer.start(0, CapabilityRegistry.discover(), apis.service(keys, List.of(dir)));
    }

    @AfterEach
    void stop() {
        server.close();
        apis.close();
    }

    private HttpResponse<String> call(String method, String path, String body) throws Exception {
        HttpRequest.Builder b = HttpRequest.newBuilder(server.uri().resolve(path))
                .header(StudioServer.TOKEN_HEADER, server.token());
        b = body == null ? b.GET() : b.method(method, HttpRequest.BodyPublishers.ofString(body));
        return HTTP.send(b.build(), HttpResponse.BodyHandlers.ofString());
    }

    @Test
    void aSavedKeyMakesTheProviderReadyAndIsNeverSentBack() throws Exception {
        Map<String, Object> before = Json.object(Json.parse(call("GET", "/api/models", null).body()));
        assertEquals(Map.of(), before.get("auto"));
        HttpResponse<String> r = call("POST", "/api/keys", "{\"provider\":\"anthropic\",\"label\":\"work\",\"key\":\"sk-ant-secret-9876\"}");
        assertEquals(200, r.statusCode());
        assertFalse(r.body().contains("sk-ant-secret"));
        assertTrue(r.body().contains("\"hint\":\"…9876\""), r.body());
        Map<String, Object> after = Json.object(Json.parse(r.body()));
        assertEquals(Map.of("text", "anthropic/claude-opus-5-5"), after.get("auto"));
        Map<String, Object> anthropic = Json.array(after.get("providers")).stream().map(Json::object)
                .filter(p -> p.get("id").equals("anthropic")).findFirst().orElseThrow();
        assertEquals("work", Json.object(Json.array(anthropic.get("keys")).getFirst()).get("label"));
        assertEquals(2L, ((Number) anthropic.get("models")).longValue());
    }

    @Test
    void chatAnswersAsAnEventStream() throws Exception {
        call("POST", "/api/keys", "{\"provider\":\"anthropic\",\"key\":\"k\"}");
        HttpResponse<String> r = call("POST", "/api/chat",
                "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"text\":\"Say hello\"}]}");
        assertEquals(200, r.statusCode());
        assertTrue(r.headers().firstValue("Content-Type").orElse("").startsWith("text/event-stream"));
        String s = r.body();
        assertTrue(s.startsWith("event: route\ndata: {"), s);
        assertTrue(s.contains("\"key\":\"anthropic/claude-opus-5-5\""), s);
        assertTrue(s.contains("event: thinking\ndata: {\"text\":\"Hmm.\"}"), s);
        assertTrue(s.contains("event: text\ndata: {\"text\":\"Hello\"}"), s);
        assertTrue(s.contains("event: done\ndata: {\"stop\":\"end\",\"inputTokens\":12,\"outputTokens\":3"), s);
    }

    @Test
    void aProviderFailureArrivesAsAnErrorEvent() throws Exception {
        call("POST", "/api/keys", "{\"provider\":\"openai\",\"key\":\"k\"}");
        apis.failWith = 429;
        HttpResponse<String> r = call("POST", "/api/chat",
                "{\"model\":\"openai/gpt-5\",\"messages\":[{\"role\":\"user\",\"text\":\"Hi\"}]}");
        assertTrue(r.body().contains("event: error\ndata: {\"message\":\"OpenAI answered 429 (rate limited or out of credit)"),
                r.body());
    }

    @Test
    void badRequestsAreRefusedBeforeAnythingIsSent() throws Exception {
        assertEquals(400, call("POST", "/api/chat", "not json").statusCode());
        assertEquals(400, call("POST", "/api/chat", "{\"messages\":[]}").statusCode());
        assertEquals(400, call("POST", "/api/chat",
                "{\"messages\":[{\"role\":\"robot\",\"text\":\"x\"}]}").statusCode());
        assertEquals(400, call("POST", "/api/keys", "{\"provider\":\"nobody\",\"key\":\"k\"}").statusCode());
        assertEquals(400, call("POST", "/api/keys", "{\"provider\":\"openai\",\"label\":\"x\",\"remove\":true}")
                .statusCode());
        assertEquals(405, call("GET", "/api/chat", null).statusCode());
        assertEquals(null, apis.bodies.get("a POST"));
    }

    @Test
    void picturesAndSpeechComeBackAsBase64WithTheirFiles() throws Exception {
        call("POST", "/api/keys", "{\"provider\":\"openai\",\"key\":\"k\"}");
        HttpResponse<String> r = call("POST", "/api/images",
                "{\"model\":\"auto\",\"prompt\":\"a lighthouse\",\"aspect\":\"1:1\",\"count\":2}");
        assertEquals(200, r.statusCode(), r.body());
        Map<String, Object> answer = Json.object(Json.parse(r.body()));
        assertEquals("openai/gpt-image-1", Json.object(answer.get("model")).get("key"));
        List<Object> items = Json.array(answer.get("items"));
        assertEquals(2, items.size());
        Map<String, Object> first = Json.object(items.getFirst());
        assertEquals("PNG0", new String(java.util.Base64.getDecoder().decode((String) first.get("data"))));
        assertTrue(java.nio.file.Files.exists(Path.of((String) first.get("file"))));

        r = call("POST", "/api/speech", "{\"text\":\"Hello\",\"voice\":\"coral\"}");
        assertEquals(200, r.statusCode(), r.body());
        assertEquals("audio/mpeg", Json.object(Json.array(Json.object(Json.parse(r.body())).get("items")).getFirst()).get("mime"));

        assertEquals(400, call("POST", "/api/images", "{\"prompt\":\"x\",\"aspect\":\"2:1\"}").statusCode());
        apis.failWith = 500;
        r = call("POST", "/api/images", "{\"model\":\"openai/gpt-image-1\",\"prompt\":\"x\"}");
        assertEquals(502, r.statusCode());
        assertTrue(r.body().contains("OpenAI answered 500"), r.body());
    }

    @Test
    void usageAndBalancesAreReadable() throws Exception {
        call("POST", "/api/keys", "{\"provider\":\"deepseek\",\"label\":\"lab\",\"key\":\"sk-ds-1\"}");
        call("POST", "/api/chat", "{\"model\":\"deepseek/deepseek-chat\",\"messages\":[{\"role\":\"user\",\"text\":\"Hi\"}]}");
        HttpResponse<String> r = call("GET", "/api/usage", null);
        assertEquals(200, r.statusCode());
        Map<String, Object> usage = Json.object(Json.parse(r.body()));
        assertEquals(1L, ((Number) Json.object(Json.object(usage.get("keys")).get("deepseek/lab")).get("calls")).longValue());
        r = call("GET", "/api/balances", null);
        assertEquals(200, r.statusCode());
        assertTrue(r.body().contains("\"label\":\"lab\""), r.body());
        assertTrue(r.body().contains("\"currency\":\"CNY\""), r.body());
        assertFalse(r.body().contains("sk-ds-1"));
    }
}
