package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.github.zzuegg.brainiac.io.Json;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Models are found and downloaded from a stand-in for the Hugging Face Hub. */
class HubTest {

    @TempDir
    Path data;

    private HttpServer server;
    private HttpServer cdn;   // the file server the Hub redirects to, on another address
    private final Map<String, byte[]> files = new LinkedHashMap<>();   // "repo/path" -> bytes
    private final Map<String, String> configs = new LinkedHashMap<>();
    private final List<String> seen = new CopyOnWriteArrayList<>();      // "path auth range"
    private volatile int cutAfter = -1;                                  // drop the connection after this many bytes once
    private Keys keys;

    static byte[] gguf(String arch, int padding) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteBuffer b = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(0x46554747).putInt(3).putLong(0).putLong(2);
        byte[] k1 = "general.name".getBytes(StandardCharsets.UTF_8);
        b.putLong(k1.length).put(k1).putInt(8);
        byte[] v1 = "test".getBytes(StandardCharsets.UTF_8);
        b.putLong(v1.length).put(v1);
        byte[] k2 = "general.architecture".getBytes(StandardCharsets.UTF_8);
        b.putLong(k2.length).put(k2).putInt(8);
        byte[] v2 = arch.getBytes(StandardCharsets.UTF_8);
        b.putLong(v2.length).put(v2);
        out.write(b.array(), 0, b.position());
        out.writeBytes(new byte[padding]);
        return out.toByteArray();
    }

    @BeforeEach
    void start() throws IOException {
        keys = new Keys(data.resolve("keys.properties"), Map.of());
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/", this::handle);
        server.start();
        cdn = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        cdn.createContext("/", this::handle);
        cdn.start();
        files.put("Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf", gguf("qwen2", 300_000));
        files.put("Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-iq4_xs.gguf", gguf("qwen2", 10));
        files.put("Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-tq1_0.gguf", gguf("qwen2", 10));
        files.put("Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m-00001-of-00002.gguf", gguf("qwen2", 10));
        files.put("google/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf", gguf("gemma3", 10));
        files.put("Qwen/Qwen2.5-0.5B-Instruct/model.safetensors", new byte[5000]);
        files.put("Qwen/Qwen2.5-0.5B-Instruct/tokenizer.json", "{}".getBytes(StandardCharsets.UTF_8));
        files.put("Qwen/Qwen2.5-0.5B-Instruct/README.md", "# hi".getBytes(StandardCharsets.UTF_8));
        configs.put("Qwen/Qwen2.5-0.5B-Instruct", "{\"model_type\": \"qwen2\"}");
        files.put("meta-llama/Llama-3.2-1B/model.safetensors", new byte[10]);
    }

    @AfterEach
    void stop() {
        server.stop(0);
        cdn.stop(0);
    }

    private Hub hub() {
        return new Hub(keys, data.resolve("downloads"), URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/"));
    }

    private void handle(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getRawPath();
        seen.add(path + " " + ex.getRequestHeaders().getFirst("Authorization") + " "
                + ex.getRequestHeaders().getFirst("Range"));
        try (ex) {
            if (path.equals("/api/models")) {
                String q = ex.getRequestURI().getRawQuery();
                assertTrue(q.contains("expand%5B%5D=gguf") && q.contains("expand%5B%5D=config"), q);
                List<Object> out = new ArrayList<>();
                out.add(Map.of("id", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "downloads", 900, "likes", 40,
                        "tags", List.of("gguf", "qwen2", "conversational"), "config", Map.of(),
                        "gguf", Map.of("architecture", "qwen2", "total", 630_167_424, "context_length", 32768)));
                out.add(Map.of("id", "Qwen/Qwen2.5-0.5B-Instruct", "downloads", 2000, "likes", 300,
                        "tags", List.of("safetensors", "qwen2"), "config", Map.of("model_type", "qwen2"), "gated", false));
                out.add(Map.of("id", "google/gemma-3-1b-it", "downloads", 5000, "likes", 500,
                        "tags", List.of("safetensors", "gemma3"), "config", Map.of("model_type", "gemma3"), "gated", "manual"));
                out.add(Map.of("id", "google/gemma-3-1b-it-GGUF", "downloads", 4000, "likes", 50,
                        "tags", List.of("gguf"), "gguf", Map.of("architecture", "gemma3")));
                out.add(Map.of("id", "onnx-community/Qwen2.5-0.5B-Instruct", "downloads", 2500, "likes", 9,
                        "tags", List.of("onnx", "qwen2"), "config", Map.of("model_type", "qwen2")));
                out.add(Map.of("id", "meta-llama/Llama-3.2-1B", "downloads", 3000, "likes", 900,
                        "tags", List.of("safetensors", "llama"), "config", Map.of("model_type", "llama"), "gated", "auto"));
                send(ex, 200, Json.write(out).getBytes(StandardCharsets.UTF_8));
                return;
            }
            if (path.startsWith("/api/models/") && path.endsWith("/tree/main")) {
                String repo = path.substring("/api/models/".length(), path.length() - "/tree/main".length());
                if (repo.startsWith("meta-llama/")) {
                    send(ex, 401, "{\"error\":\"Access to model meta-llama/Llama-3.2-1B is restricted.\"}".getBytes());
                    return;
                }
                List<Object> out = new ArrayList<>();
                files.forEach((k, v) -> {
                    if (k.startsWith(repo + "/")) {
                        Map<String, Object> f = new LinkedHashMap<>();
                        f.put("type", "file");
                        f.put("path", k.substring(repo.length() + 1));
                        f.put("size", v.length);
                        if (!k.endsWith(".json") && !k.endsWith(".md")) {
                            f.put("lfs", Map.of("oid", sha(v), "size", v.length));
                        }
                        out.add(f);
                    }
                });
                if (configs.containsKey(repo)) {
                    out.add(Map.of("type", "file", "path", "config.json", "size", configs.get(repo).length()));
                }
                if (out.isEmpty()) {
                    send(ex, 404, "{\"error\":\"Repository not found\"}".getBytes());
                    return;
                }
                send(ex, 200, Json.write(out).getBytes(StandardCharsets.UTF_8));
                return;
            }
            int resolve = path.indexOf("/resolve/main/");
            if (resolve > 0 && !path.startsWith("/cdn/")) {
                // the Hub sends the file from another place, as the real one does
                ex.getResponseHeaders().set("Location", "http://127.0.0.1:" + cdn.getAddress().getPort() + "/cdn" + path);
                ex.sendResponseHeaders(302, -1);
                return;
            }
            if (path.startsWith("/cdn/")) {
                String p = java.net.URLDecoder.decode(path.substring(5), StandardCharsets.UTF_8);
                int r = p.indexOf("/resolve/main/");
                String repo = p.substring(0, r);
                String file = p.substring(r + "/resolve/main/".length());
                byte[] body = file.equals("config.json") ? configs.getOrDefault(repo, "").getBytes(StandardCharsets.UTF_8)
                        : files.get(repo + "/" + file);
                if (body == null) {
                    send(ex, 404, new byte[0]);
                    return;
                }
                String range = ex.getRequestHeaders().getFirst("Range");
                int from = 0;
                int to = body.length - 1;
                if (range != null) {
                    String[] ft = range.substring(6).split("-", -1);
                    from = Integer.parseInt(ft[0]);
                    if (!ft[1].isEmpty()) {
                        to = Math.min(to, Integer.parseInt(ft[1]));
                    }
                }
                byte[] part = java.util.Arrays.copyOfRange(body, from, to + 1);
                if (cutAfter > 0 && part.length > cutAfter && (range == null || range.endsWith("-"))) {
                    int n = cutAfter;
                    cutAfter = -1;
                    ex.sendResponseHeaders(range == null ? 200 : 206, part.length);
                    OutputStream o = ex.getResponseBody();
                    o.write(part, 0, n);
                    o.flush();
                    throw new IOException("cut");   // the connection drops halfway
                }
                send(ex, range == null ? 200 : 206, part);
                return;
            }
            send(ex, 404, new byte[0]);
        }
    }

    private static void send(HttpExchange ex, int status, byte[] body) throws IOException {
        ex.sendResponseHeaders(status, body.length == 0 ? -1 : body.length);
        if (body.length > 0) {
            ex.getResponseBody().write(body);
        }
    }

    private static String sha(byte[] b) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b));
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static Map<String, Object> option(Map<String, Object> listing, String name) {
        return Json.array(listing.get("options")).stream().map(Json::object).filter(o -> name.equals(o.get("name")))
                .findFirst().orElseThrow(() -> new AssertionError("no " + name + " in " + listing));
    }

    private static void await(Hub hub, String id) throws InterruptedException {
        for (int i = 0; i < 400; i++) {
            String st = (String) hub.downloads().stream().filter(d -> id.equals(d.get("id"))).findFirst().orElseThrow()
                    .get("state");
            if (!st.equals("queued") && !st.equals("downloading") && !st.equals("checking")) {
                return;
            }
            Thread.sleep(25);
        }
    }

    @Test
    void searchKeepsOnlyWhatTheCoreRuns() throws Exception {
        Map<String, Object> found = hub().search("1b");
        List<Map<String, Object>> models = Json.array(found.get("models")).stream().map(Json::object).toList();
        assertEquals(List.of("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Qwen/Qwen2.5-0.5B-Instruct", "google/gemma-3-1b-it-GGUF",
                "meta-llama/Llama-3.2-1B"), models.stream().map(m -> m.get("id")).toList());
        assertEquals(2, found.get("hidden"), "the multimodal gemma3 type, and ONNX weights the core does not load");
        assertEquals(true, models.getFirst().get("gguf"));
        assertEquals(630_167_424L, models.getFirst().get("parameters"));
        assertEquals(true, models.get(1).get("safetensors"));
        assertEquals(true, models.get(2).get("gated"));
        assertEquals(false, models.get(1).get("gated"));
        // what runs comes from the core
        assertEquals(io.github.zzuegg.brainiac.llm.LlamaConfig.GGUF_ARCHITECTURES.stream().sorted().toList(),
                hub().runs().get("architectures"));
        assertTrue(Json.array(hub().runs().get("quantizations")).contains("Q4_K"));
    }

    @Test
    void eachFileSaysWhetherTheStudioRunsIt() throws Exception {
        Map<String, Object> l = hub().files("Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assertEquals("qwen2", l.get("family"));
        Map<String, Object> q8 = option(l, "qwen2.5-0.5b-instruct-q8_0.gguf");
        assertEquals(true, q8.get("supported"), q8.toString());
        assertEquals("Q8_0", q8.get("quant"));
        // whether a quantization runs is the core's answer: IQ4_XS runs once Gguf reads it
        Map<String, Object> iq = option(l, "qwen2.5-0.5b-instruct-iq4_xs.gguf");
        assertEquals(io.github.zzuegg.brainiac.io.Gguf.formats().contains("IQ4_XS"), iq.get("supported"), iq.toString());
        Map<String, Object> tq = option(l, "qwen2.5-0.5b-instruct-tq1_0.gguf");
        assertEquals(false, tq.get("supported"));
        assertTrue(((String) tq.get("reason")).contains("TQ1_0"), tq.toString());
        Map<String, Object> split = option(l, "qwen2.5-0.5b-instruct-q4_k_m-00001-of-00002.gguf");
        assertTrue(((String) split.get("reason")).contains("split into 2 parts"), split.toString());

        Map<String, Object> gemma = option(hub().files("google/gemma-3-1b-it-GGUF"), "gemma-3-1b-it-Q4_K_M.gguf");
        assertEquals(false, gemma.get("supported"));
        assertTrue(((String) gemma.get("reason")).contains("gemma3"), gemma.toString());

        Map<String, Object> folder = option(hub().files("Qwen/Qwen2.5-0.5B-Instruct"), "Qwen2.5-0.5B-Instruct");
        assertEquals(true, folder.get("supported"), folder.toString());
        assertEquals(List.of("model.safetensors", "tokenizer.json", "config.json"), folder.get("files"));

        IOException missing = assertThrows(IOException.class, () -> hub().files("nobody/nothing"));
        assertTrue(missing.getMessage().contains("nothing at"), missing.getMessage());
        assertThrows(IOException.class, () -> hub().files("../etc"));
    }

    @Test
    void quantizationsAreReadFromFileNames() {
        assertEquals("Q4_K_M", Hub.quant("qwen2.5-7b-instruct-q4_k_m.gguf"));
        assertEquals("Q8_0", Hub.quant("Llama-3.2-1B-Instruct-Q8_0.gguf"));
        assertEquals("IQ2_XXS", Hub.quant("model-IQ2_XXS.gguf"));
        assertEquals("F16", Hub.quant("tinyllama-1.1b-chat-v1.0.f16.gguf"));
        assertEquals("BF16", Hub.quant("Qwen3-0.6B-BF16.gguf"));
        assertEquals("", Hub.quant("model.gguf"));
    }

    @Test
    void aGgufDownloadResumesIsCheckedAndBecomesAModel() throws Exception {
        keys.add("huggingface", "default", "hf_secret_token_value");
        ChatService chat = ChatService.at(keys, List.of(), Map.of("hub",
                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/")), data.resolve("outputs"));
        Hub hub = chat.hub();
        cutAfter = 100_000;
        Map<String, Object> d = hub.download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        await(hub, (String) d.get("id"));
        Map<String, Object> first = hub.downloads().getFirst();
        assertEquals("failed", first.get("state"), first.toString());
        assertTrue(((String) first.get("error")).contains("resume"), first.toString());
        Path part = data.resolve("downloads/qwen2.5-0.5b-instruct-q8_0.gguf.part");
        assertTrue(Files.size(part) > 0 && Files.size(part) < 300_000);

        Map<String, Object> again = hub.download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        await(hub, (String) again.get("id"));
        assertEquals("done", hub.downloads().getFirst().get("state"), hub.downloads().getFirst().toString());
        Path model = data.resolve("downloads/qwen2.5-0.5b-instruct-q8_0.gguf");
        assertTrue(Files.isRegularFile(model));
        assertFalse(Files.exists(part));
        assertTrue(seen.stream().anyMatch(s -> s.startsWith("/cdn/") && s.matches(".* bytes=[1-9]\\d*-$")),
                "the second try asked for the rest only: " + seen);
        // the key goes to the Hub, never to the file server it redirects to
        assertTrue(seen.stream().filter(s -> s.startsWith("/api/")).allMatch(s -> s.contains("Bearer hf_secret")));
        assertTrue(seen.stream().filter(s -> s.startsWith("/cdn/")).allMatch(s -> s.contains(" null ")), seen.toString());

        List<String> local = Json.array(chat.describe(false).get("models")).stream().map(Json::object)
                .filter(m -> "local".equals(m.get("provider"))).map(m -> (String) m.get("name")).toList();
        assertEquals(List.of("qwen2.5-0.5b-instruct-q8_0.gguf"), local);
        assertEquals(true, option(hub.files("Qwen/Qwen2.5-0.5B-Instruct-GGUF"), "qwen2.5-0.5b-instruct-q8_0.gguf")
                .get("have"));

        assertTrue(hub.delete("qwen2.5-0.5b-instruct-q8_0.gguf"));
        assertFalse(Files.exists(model));
        assertFalse(hub.delete("../keys.properties"));
    }

    @Test
    void aModelFolderDownloadsWhole() throws Exception {
        Hub hub = hub();
        Map<String, Object> d = hub.download("Qwen/Qwen2.5-0.5B-Instruct", "folder");
        await(hub, (String) d.get("id"));
        assertEquals("done", hub.downloads().getFirst().get("state"), hub.downloads().getFirst().toString());
        Path dir = data.resolve("downloads/Qwen2.5-0.5B-Instruct");
        assertEquals(5000, Files.size(dir.resolve("model.safetensors")));
        assertTrue(Files.exists(dir.resolve("config.json")));
        assertFalse(Files.exists(dir.resolve("README.md")), "only what the model needs");
        assertEquals("model", LocalProvider.kind(dir));
    }

    @Test
    void unsupportedAndGatedModelsAreRefusedWithAReason() throws Exception {
        IOException iq = assertThrows(IOException.class,
                () -> hub().download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-tq1_0.gguf"));
        assertTrue(iq.getMessage().contains("cannot run"), iq.getMessage());
        IOException gated = assertThrows(IOException.class, () -> hub().files("meta-llama/Llama-3.2-1B"));
        assertTrue(gated.getMessage().contains("gated"), gated.getMessage());
    }

    @Test
    void theArchitectureIsReadFromTheStartOfAGguf() throws Exception {
        assertEquals("qwen2", Hub.ggufArchitecture(gguf("qwen2", 0)));
        assertThrows(IOException.class, () -> Hub.ggufArchitecture("not gguf at all".getBytes()));
    }
}
