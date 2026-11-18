package io.github.zzuegg.brainiac.studio.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatListener;
import io.github.zzuegg.brainiac.studio.chat.ChatMessage;
import io.github.zzuegg.brainiac.studio.chat.ChatProvider;
import io.github.zzuegg.brainiac.studio.chat.ChatRequest;
import io.github.zzuegg.brainiac.studio.chat.ChatResult;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import io.github.zzuegg.brainiac.studio.chat.ModelInfo;
import io.github.zzuegg.brainiac.studio.server.WorkerServer;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A worker host serves its chat models, and the studio lists and uses them like any provider. */
class HostChatTest {

    private static final String TOKEN = "a-given-worker-token-0123456789abcdef";

    @TempDir
    Path studio;

    @TempDir
    Path lab;

    /** A model on the worker that answers in two pieces and remembers what it was asked. */
    static final class Echo implements ChatProvider {
        final List<ChatRequest> heard = new CopyOnWriteArrayList<>();

        @Override
        public String id() {
            return "local";
        }

        @Override
        public String name() {
            return "This PC";
        }

        @Override
        public boolean ready() {
            return true;
        }

        @Override
        public boolean local() {
            return true;
        }

        @Override
        public List<ModelInfo> models() {
            return List.of(new ModelInfo("local", "qwen-0.5b", "Qwen 0.5B", List.of("text"), 0, "safetensors · 1 GB"));
        }

        @Override
        public ChatResult chat(ChatRequest request, ChatListener listener) throws IOException {
            heard.add(request);
            if (request.model().equals("broken")) {
                throw new IOException("no model named broken");
            }
            listener.thinking("hmm");
            listener.text("you said ");
            listener.text(request.messages().getLast().text());
            return new ChatResult("end", 7, 3);
        }
    }

    @Test
    void aHostsModelsAreListedAndAnswerThroughTheStudio() throws Exception {
        Echo echo = new Echo();
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab,
                echo, TOKEN);
             TrainingService t = new TrainingService(studio)) {
            assertEquals(TOKEN, worker.token(), "a given token is used as it is");
            Hosts.Host h = t.hosts().add("lab", "127.0.0.1:" + worker.port(), TOKEN);
            ChatService chat = new ChatService(new Keys(studio.resolve("keys.properties"), Map.of()), List.of(),
                    studio.resolve("outputs"));
            chat.hosts(() -> t.hosts().list().stream()
                    .map(x -> (ChatProvider) new HostProvider(x, t.hosts().client(x))).toList());

            Map<String, Object> d = chat.describe(true);
            Map<String, Object> p = Json.object(Json.array(d.get("providers")).getFirst());
            assertEquals("host-" + h.id(), p.get("id"));
            assertEquals("lab", p.get("name"));
            assertEquals("host", p.get("kind"));
            assertEquals(1, ((Number) p.get("models")).intValue());
            Map<String, Object> m = Json.object(Json.array(d.get("models")).getFirst());
            assertEquals("host-" + h.id() + "/qwen-0.5b", m.get("key"));
            assertEquals("safetensors · 1 GB · on lab", m.get("detail"));
            assertFalse(d.toString().contains(TOKEN), "the token never goes to the page");

            List<String> said = new ArrayList<>();
            List<String> thought = new ArrayList<>();
            ChatResult r = chat.chat("host-" + h.id() + "/qwen-0.5b", "be brief",
                    List.of(new ChatMessage("user", "hello there")), 64, null, x -> { }, new ChatListener() {
                        @Override
                        public void text(String text) {
                            said.add(text);
                        }

                        @Override
                        public void thinking(String text) {
                            thought.add(text);
                        }
                    });
            assertEquals("you said hello there", String.join("", said));
            assertEquals(List.of("hmm"), thought);
            assertEquals(7, r.inputTokens());
            assertEquals(3, r.outputTokens());
            ChatRequest heard = echo.heard.getFirst();
            assertEquals("qwen-0.5b", heard.model());
            assertEquals("be brief", heard.system());
            assertEquals(64, heard.maxTokens());

            // a picture on a message reaches the host's model as it was sent
            byte[] png = {(byte) 0x89, 'P', 'N', 'G', 1, 2, 3};
            chat.chat("host-" + h.id() + "/qwen-0.5b", "", List.of(new ChatMessage("user", "what is this?",
                    List.of(new io.github.zzuegg.brainiac.studio.chat.Media(png, "image/png")))), 64, null, x -> { },
                    new ChatListener() {
                        @Override
                        public void text(String text) {
                        }
                    });
            ChatMessage seen = echo.heard.getLast().messages().getLast();
            assertEquals(1, seen.images().size());
            assertEquals("image/png", seen.images().getFirst().mime());
            org.junit.jupiter.api.Assertions.assertArrayEquals(png, seen.images().getFirst().data());

            HostProvider direct = new HostProvider(h, t.hosts().client(h));
            IOException e = assertThrows(IOException.class, () -> direct.chat(
                    new ChatRequest("broken", "", List.of(new ChatMessage("user", "hi")), 16, null), new ChatListener() {
                        @Override
                        public void text(String text) {
                        }
                    }));
            assertEquals("no model named broken", e.getMessage());

            t.hosts().remove(h.id());
            assertTrue(Json.array(chat.describe(true).get("providers")).isEmpty(), "a removed host is gone at once");
        }
    }

    @Test
    void theWorkerTakesAModelKeyAndContentAndRefusesEmptyMessages() throws Exception {
        Echo echo = new Echo();
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab,
                echo, TOKEN)) {
            java.net.http.HttpClient http = java.net.http.HttpClient.newHttpClient();
            java.util.function.Function<String, java.net.http.HttpResponse<String>> post = json -> {
                try {
                    return http.send(java.net.http.HttpRequest.newBuilder(
                                    java.net.URI.create("http://127.0.0.1:" + worker.port() + "/worker/chat"))
                            .header("Authorization", "Bearer " + TOKEN)
                            .POST(java.net.http.HttpRequest.BodyPublishers.ofString(json)).build(),
                            java.net.http.HttpResponse.BodyHandlers.ofString());
                } catch (IOException | InterruptedException e) {
                    throw new AssertionError(e);
                }
            };
            var ok = post.apply("{\"model\":\"local/qwen-0.5b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
            assertEquals(200, ok.statusCode());
            assertTrue(ok.body().contains("event: done"), ok.body());
            assertEquals("qwen-0.5b", echo.heard.getFirst().model());
            assertEquals("hi", echo.heard.getFirst().messages().getFirst().text());

            var empty = post.apply("{\"model\":\"qwen-0.5b\",\"messages\":[{\"role\":\"user\",\"body\":\"hi\"}]}");
            assertEquals(400, empty.statusCode());
            assertTrue(empty.body().contains("text"), empty.body());
            assertEquals(400, post.apply("{\"model\":\"qwen-0.5b\",\"messages\":[]}").statusCode());
            assertEquals(1, echo.heard.size(), "refused requests never reach the model");
        }
    }

    @Test
    void aShortGivenTokenIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> WorkerServer.start(
                new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab, new Echo(), "too-short"));
    }
}
