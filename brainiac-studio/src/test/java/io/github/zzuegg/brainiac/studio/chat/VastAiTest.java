package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The Vast.ai key is kept like the others, and the account and the offers are read with it. */
class VastAiTest {

    @TempDir
    Path dir;

    private HttpServer server;
    private final List<String> auth = new CopyOnWriteArrayList<>();
    private final List<String> queries = new CopyOnWriteArrayList<>();

    @BeforeEach
    void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/api/v0/", ex -> {
            String a = ex.getRequestHeaders().getFirst("Authorization");
            auth.add(String.valueOf(a));
            String path = ex.getRequestURI().getPath();
            String body;
            int status = 200;
            if (!"Bearer vast-key-1234".equals(a)) {
                status = 401;
                body = "{\"msg\":\"bad key\"}";
            } else if (path.equals("/api/v0/users/current/")) {
                body = "{\"balance\": 12.5, \"email\": \"someone@example.com\"}";
            } else if (path.equals("/api/v0/instances/")) {
                body = "{\"instances\": [{\"id\": 7, \"actual_status\": \"running\", \"gpu_name\": \"RTX_4090\", "
                        + "\"num_gpus\": 1, \"dph_total\": 0.41, \"geolocation\": \"Sweden, SE\"}]}";
            } else if (path.equals("/api/v0/bundles/") && ex.getRequestMethod().equals("POST")) {
                queries.add(new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
                body = "{\"offers\": [{\"id\": 1, \"gpu_name\": \"RTX_3090\", \"num_gpus\": 1, \"gpu_ram\": 24576, "
                        + "\"dph_total\": 0.18, \"reliability\": 0.991, \"geolocation\": \"Spain, ES\", \"cuda_max_good\": 12.4},"
                        + "{\"id\": 2, \"gpu_name\": \"RTX_4090\", \"num_gpus\": 2, \"gpu_ram\": 24564, \"dph_total\": 0.7,"
                        + " \"reliability2\": 0.98, \"cpu_ram\": 64000, \"inet_down\": 900.5}]}";
            } else {
                status = 404;
                body = "{}";
            }
            byte[] b = body.getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(status, b.length);
            ex.getResponseBody().write(b);
            ex.close();
        });
        server.start();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    private ChatService service() {
        URI base = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/api/v0/");
        return ChatService.at(new Keys(dir.resolve("keys.properties"), Map.of()), List.of(dir.resolve("models")),
                Map.of(VastAi.PROVIDER, base), dir.resolve("outputs"));
    }

    @Test
    void withoutAKeyNothingIsAsked() {
        ChatService chat = service();
        Map<String, Object> s = chat.vast().status();
        assertTrue(s.containsKey("key"));
        assertNull(s.get("key"));
        IOException e = assertThrows(IOException.class, () -> chat.vast().offers("", 0, 0, 0));
        assertEquals("add your Vast.ai API key first", e.getMessage());
        assertTrue(auth.isEmpty());
    }

    @Test
    void theAccountAndTheOffersAreReadWithTheSavedKey() throws Exception {
        ChatService chat = service();
        chat.addKey(VastAi.PROVIDER, "default", "vast-key-1234");
        Map<String, Object> s = chat.vast().status();
        assertEquals(Map.of("label", "default", "source", "saved", "hint", "…1234"), s.get("key"));
        assertEquals(12.5, s.get("balance"));
        assertTrue(!s.toString().contains("example.com"), "the account's email is not passed on");
        Map<String, Object> i = Json.object(Json.array(s.get("instances")).getFirst());
        assertEquals("RTX 4090", i.get("gpu"));
        assertEquals("running", i.get("status"));

        List<Map<String, Object>> all = chat.vast().offers("", 24, 1, 0.75);
        assertEquals(List.of("RTX 3090", "RTX 4090"), all.stream().map(o -> o.get("gpu")).toList());
        assertEquals(24L, all.getFirst().get("gpuRamGb"));
        assertEquals(0.98, all.get(1).get("reliability"));
        Map<String, Object> q = Json.object(Json.parse(queries.getFirst()));
        assertEquals(Map.of("gte", 24576.0), normalize(q.get("gpu_ram")));
        assertEquals(Map.of("lte", 0.75), normalize(q.get("dph_total")));
        assertEquals("ondemand", q.get("type"));
        assertEquals(Map.of("eq", true), q.get("verified"));

        assertEquals(List.of("RTX 4090"), chat.vast().offers("4090", 0, 0, 0).stream().map(o -> o.get("gpu")).toList());
        assertTrue(auth.stream().allMatch("Bearer vast-key-1234"::equals));
    }

    @Test
    void aRefusedKeyIsSaidPlainly() throws Exception {
        ChatService chat = service();
        chat.addKey(VastAi.PROVIDER, "default", "wrong-key-0000");
        Map<String, Object> s = chat.vast().status();
        assertTrue(String.valueOf(s.get("error")).contains("Vast.ai refused the key"), String.valueOf(s));
        assertThrows(IOException.class, () -> chat.vast().offers("", 0, 0, 0));
    }

    private static Map<String, Object> normalize(Object o) {
        Map<String, Object> m = Json.object(o);
        Map.Entry<String, Object> e = m.entrySet().iterator().next();
        return Map.of(e.getKey(), ((Number) e.getValue()).doubleValue());
    }
}
