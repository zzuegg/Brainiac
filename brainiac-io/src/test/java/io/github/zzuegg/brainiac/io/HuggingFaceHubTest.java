package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceHubTest {

    @Test
    void findsTheTokenInOrder(@TempDir Path home) throws IOException {
        assertEquals(Optional.empty(), HuggingFaceHub.findToken(home, Map.of()));
        Files.createDirectories(home.resolve(".cache/huggingface"));
        Files.writeString(home.resolve(".cache/huggingface/token"), "cache\n");
        assertEquals(Optional.of("cache"), HuggingFaceHub.findToken(home, Map.of()));
        Path hfHome = Files.createDirectories(home.resolve("hf"));
        Files.writeString(hfHome.resolve("token"), "hfhome");
        assertEquals(Optional.of("hfhome"), HuggingFaceHub.findToken(home, Map.of("HF_HOME", hfHome.toString())));
        assertEquals(Optional.of("hub"), HuggingFaceHub.findToken(home, Map.of("HUGGINGFACE_HUB_TOKEN", "hub")));
        assertEquals(Optional.of("env"), HuggingFaceHub.findToken(home,
                Map.of("HF_TOKEN", " env ", "HUGGINGFACE_HUB_TOKEN", "hub")));
        Files.createDirectories(home.resolve(".brainiac"));
        Files.writeString(home.resolve(".brainiac/hf_token"), "brainiac\r\n");
        assertEquals(Optional.of("brainiac"), HuggingFaceHub.findToken(home, Map.of("HF_TOKEN", "env")));
    }

    @Test
    void downloadsMatchingFilesWithTheToken(@TempDir Path dir) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        List<String> auth = new ArrayList<>();
        Map<String, String> files = Map.of("config.json", "{\"a\": 1}", "model.safetensors", "weights",
                "README.md", "readme", "sub/tokenizer.json", "{}");
        server.createContext("/", ex -> {
            String path = ex.getRequestURI().getPath();
            String a = ex.getRequestHeaders().getFirst("Authorization");
            auth.add(a);
            byte[] body;
            int code = 200;
            if (path.startsWith("/cdn/")) {
                body = (a == null ? "weights" : "leaked").getBytes(StandardCharsets.UTF_8);
            } else if (!"Bearer secret".equals(a)) {
                code = 401;
                body = new byte[0];
            } else if (path.equals("/api/models/org/gated/revision/main")) {
                List<Object> siblings = new ArrayList<>();
                files.keySet().stream().sorted().forEach(f -> siblings.add(Map.of("rfilename", f)));
                body = Json.write(Map.of("siblings", siblings)).getBytes(StandardCharsets.UTF_8);
            } else if (path.equals("/org/gated/resolve/main/model.safetensors")) {
                ex.getResponseHeaders().add("Location", "http://localhost:" + ex.getLocalAddress().getPort()
                        + "/cdn/model.safetensors");
                code = 302;
                body = new byte[0];
            } else if (path.startsWith("/org/gated/resolve/main/")) {
                String f = path.substring("/org/gated/resolve/main/".length());
                body = files.getOrDefault(f, "").getBytes(StandardCharsets.UTF_8);
                code = files.containsKey(f) ? 200 : 404;
            } else {
                code = 404;
                body = new byte[0];
            }
            ex.sendResponseHeaders(code, body.length == 0 ? -1 : body.length);
            if (body.length > 0) {
                ex.getResponseBody().write(body);
            }
            ex.close();
        });
        server.start();
        try {
            URI ep = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/");
            IOException e = assertThrows(IOException.class,
                    () -> new HuggingFaceHub(ep, null).download("org/gated", dir));
            assertTrue(e.getMessage().contains("needs a read token"), e.getMessage());
            assertFalse(e.getMessage().contains("secret"));

            HuggingFaceHub hub = new HuggingFaceHub(ep, "secret\n");
            assertTrue(hub.hasToken());
            assertFalse(hub.toString().contains("secret"));
            hub.download("org/gated", "main", List.of("*.json", "*.safetensors", "**/*.json"), dir);
            assertEquals("{\"a\": 1}", Files.readString(dir.resolve("config.json")));
            assertEquals("weights", Files.readString(dir.resolve("model.safetensors")));
            assertEquals("{}", Files.readString(dir.resolve("sub/tokenizer.json")));
            assertFalse(Files.exists(dir.resolve("README.md")));

            int before = auth.size();
            hub.downloadFile("org/gated", "main", "config.json", dir);   // kept, not fetched again
            assertEquals(before, auth.size());
            assertThrows(IOException.class, () -> hub.downloadFile("org/gated", "main", "../x", dir));
        } finally {
            server.stop(0);
        }
    }
}
