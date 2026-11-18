package io.github.zzuegg.brainiac.studio;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.server.CapabilityRegistry;
import io.github.zzuegg.brainiac.studio.spi.Capability;
import io.github.zzuegg.brainiac.studio.spi.CapabilityProvider;
import io.github.zzuegg.brainiac.studio.spi.Medium;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class StudioServerTest {

    private static Studio studio;
    private static final HttpClient HTTP = HttpClient.newHttpClient();

    @BeforeAll
    static void start() throws IOException {
        Path home = Files.createTempDirectory("studio-test");
        studio = Studio.start(new Studio.Settings(0, List.of(home.resolve("models")), home.resolve("data"), Map.of()));
    }

    @AfterAll
    static void stop() {
        studio.close();
    }

    private static HttpResponse<String> get(String path, boolean withToken) throws Exception {
        HttpRequest.Builder b = HttpRequest.newBuilder(studio.uri().resolve(path));
        if (withToken) {
            b.header("X-Studio-Token", studio.token());
        }
        return HTTP.send(b.build(), HttpResponse.BodyHandlers.ofString());
    }

    @Test
    void servesThePageWithTheTokenInIt() throws Exception {
        HttpResponse<String> r = get("/", false);
        assertEquals(200, r.statusCode());
        assertTrue(r.body().startsWith("<!doctype html>"));
        assertTrue(r.body().contains("<meta name=\"studio-token\" content=\"" + studio.token() + "\">"));
        assertTrue(r.body().indexOf("<title>") < r.body().indexOf("</head>"), "title goes in the head");
        assertTrue(r.headers().firstValue("Content-Security-Policy").orElse("").contains("default-src 'self'"));
        HttpResponse<String> css = get("/app.css", false);
        assertEquals(200, css.statusCode());
        assertTrue(css.headers().firstValue("Content-Type").orElse("").startsWith("text/css"));
    }

    @Test
    void everythingIsKeptInTheDataFolder() {
        Path data = Path.of("some", "data").toAbsolutePath();
        Studio.Settings s = Studio.Settings.defaults().withData(Path.of("some", "data"));
        assertEquals(data, s.data());
        assertEquals(data.resolve("keys.properties"), s.keyFile());
        assertEquals(data.resolve("outputs"), s.outputs());
    }

    @Test
    void servesOnlyFilesInTheWebFolder() throws Exception {
        assertEquals(404, get("/missing.js", false).statusCode());
        assertEquals(404, get("/..%2Fserver%2FStudioServer.class", false).statusCode());
        assertEquals(404, get("/server/StudioServer.class", false).statusCode());
    }

    @Test
    void apiNeedsTheToken() throws Exception {
        assertEquals(403, get("/api/capabilities", false).statusCode());
        assertEquals(200, get("/api/capabilities", true).statusCode());
        assertEquals(404, get("/api/nothing", true).statusCode());
    }

    @Test
    void refusesOtherHostNames() throws Exception {
        // a page on another site whose name resolves to 127.0.0.1 sends its own Host header
        int port = studio.uri().getPort();
        try (Socket s = new Socket("127.0.0.1", port)) {
            OutputStream out = s.getOutputStream();
            out.write(("GET / HTTP/1.1\r\nHost: evil.example:" + port + "\r\nConnection: close\r\n\r\n")
                    .getBytes(StandardCharsets.US_ASCII));
            out.flush();
            InputStream in = s.getInputStream();
            String status = new String(in.readNBytes(12), StandardCharsets.US_ASCII);
            assertEquals("HTTP/1.1 403", status);
        }
    }

    private static HttpResponse<String> post(String path, String json) throws Exception {
        return HTTP.send(HttpRequest.newBuilder(studio.uri().resolve(path)).header("X-Studio-Token", studio.token())
                .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(json)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    @Test
    void theDeviceThisPcsModelsRunOnCanBeChosenAndIsKept() throws Exception {
        Map<String, Object> first = Json.object(Json.parse(get("/api/local/device", true).body()));
        assertEquals("best", first.get("choice"));
        String cpu = io.github.zzuegg.brainiac.tensor.Device.cpu().name();
        Map<String, Object> chosen = Json.object(Json.parse(post("/api/local/device", Json.write(Map.of("device", cpu))).body()));
        assertEquals(cpu, chosen.get("choice"));
        assertEquals(cpu, chosen.get("device"));
        assertEquals(400, post("/api/local/device", "{\"device\": \"no such card\"}").statusCode());
        assertEquals(cpu, Json.object(Json.parse(get("/api/local/device", true).body())).get("choice"));
        assertEquals(200, post("/api/local/device", "{\"device\": \"best\"}").statusCode());
    }

    @Test
    void listsCapabilitiesOfThePresentModules() throws Exception {
        Map<String, Object> body = Json.object(Json.parse(get("/api/capabilities", true).body()));
        List<Object> caps = Json.array(body.get("capabilities"));
        Map<String, Map<String, Object>> byKey = new java.util.HashMap<>();
        for (Object c : caps) {
            Map<String, Object> m = Json.object(c);
            byKey.put((String) m.get("key"), m);
        }
        assertEquals("available", byKey.get("text.chat").get("status"));
        assertEquals("text", byKey.get("text.chat").get("medium"));
        assertEquals("available", byKey.get("model.build").get("status"));
        assertEquals("available", byKey.get("speech.synthesize").get("status"));
        assertEquals("available", byKey.get("agent.run").get("status"));
        assertEquals("planned", byKey.get("vision.describe").get("status"));
    }

    @Test
    void describesTheMachine() throws Exception {
        Map<String, Object> body = Json.object(Json.parse(get("/api/system", true).body()));
        List<Object> devices = Json.array(body.get("devices"));
        assertTrue(devices.stream().anyMatch(d -> "cpu".equals(Json.object(d).get("kind"))), body.toString());
    }

    @Test
    void anAvailableCapabilityReplacesAPlannedOne() {
        CapabilityProvider planned = () -> List.of(Capability.planned("speech.synthesize", Medium.SPEECH, "TTS", "Speak"));
        CapabilityProvider real = () -> List.of(
                Capability.available("speech.synthesize", Medium.SPEECH, "TTS", "Speak", "my-tts"),
                Capability.available("text.chat", Medium.TEXT, "Chat", "Talk", "llm"));
        CapabilityProvider broken = () -> {
            throw new IllegalStateException("broken module");
        };
        List<Capability> all = CapabilityRegistry.of(List.of(planned, broken, real)).all();
        assertEquals(2, all.size());
        Capability tts = all.stream().filter(c -> c.key().equals("speech.synthesize")).findFirst().orElseThrow();
        assertEquals(Capability.Status.AVAILABLE, tts.status());
        assertEquals("my-tts", tts.source());
    }

    @Test
    void rejectsBadKeys() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> Capability.available("Chat", Medium.TEXT, "Chat", "Talk", "llm"));
    }

    @Test
    void uriIsLoopback() {
        URI u = studio.uri();
        assertEquals("127.0.0.1", u.getHost());
    }
}
