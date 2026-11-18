package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** One of this PC's chat models is served to other programs with the OpenAI API. */
class LocalServingTest {

    private static final HttpClient HTTP = HttpClient.newHttpClient();

    @TempDir
    Path models;

    private static HttpResponse<String> send(String url, String key, String body) throws Exception {
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(url)).header("Content-Type", "application/json");
        if (key != null) {
            b.header("Authorization", "Bearer " + key);
        }
        return HTTP.send(body == null ? b.GET().build() : b.POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    @Test
    void aLocalModelIsServedStoppedAndGuardedByAKeyOnTheNetwork() throws Exception {
        LocalVisionTest.tinyLlava(models, "tiny", "{% for m in messages %}{{ m['role'] | upper }}: {{ m['content'] }} "
                + "{% endfor %}ASSISTANT:");
        try (LocalServing serving = new LocalServing((LocalProvider) ChatService.localModels(List.of(models)))) {
            assertEquals(false, serving.state().get("running"));
            assertThrows(IOException.class, () -> serving.start("no-such-model", 0, false));
            assertThrows(IllegalArgumentException.class, () -> serving.start("tiny", 80, false));

            Map<String, Object> s = serving.start("tiny", 0, false);
            assertEquals(true, s.get("running"));
            assertEquals("tiny", s.get("model"));
            assertEquals("", s.get("key"), "on this PC only, no key is needed");
            String url = (String) s.get("url");
            assertTrue(url.startsWith("http://127.0.0.1:") && url.endsWith("/v1"), url);
            String listed = send(url + "/models", null, null).body();
            assertTrue(listed.contains("\"tiny\""), listed);
            HttpResponse<String> answer = send(url + "/chat/completions", null,
                    "{\"max_tokens\":3,\"messages\":[{\"role\":\"user\",\"content\":\"What is this?\"}]}");
            assertEquals(200, answer.statusCode(), answer.body());
            assertFalse(Json.array(Json.object(Json.parse(answer.body())).get("choices")).isEmpty());

            // the port it holds is taken for a second server, and says so
            int port = ((Number) s.get("port")).intValue();
            try (LocalServing other = new LocalServing((LocalProvider) ChatService.localModels(List.of(models)))) {
                IOException taken = assertThrows(IOException.class, () -> other.start("tiny", port, false));
                assertTrue(taken.getMessage().contains("port " + port), taken.getMessage());
            }

            // open to the network: a key made up for this run, and nothing without it
            Map<String, Object> open = serving.start("tiny", 0, true);
            String key = (String) open.get("key");
            assertTrue(key.startsWith("sk-brainiac-") && key.length() > 40, key);
            String local = "http://127.0.0.1:" + open.get("port") + "/v1/models";
            assertEquals(401, send(local, null, null).statusCode());
            assertEquals(200, send(local, key, null).statusCode());
            assertThrows(IOException.class, () -> send(url + "/models", null, null), "the first server is gone");

            // a model on this PC can only choose words that keep its answer in the format
            LocalProvider held = (LocalProvider) ChatService.localModels(List.of(models));
            for (String word : List.of("box.", "that?")) {
                StringBuilder said = new StringBuilder();
                held.chat(new ChatRequest("tiny", "", List.of(new ChatMessage("user", "What is this?")), 8, null, null,
                        Sampler.NONE, new Format("grammar", null, "root ::= \"" + word + "\"")), said::append);
                assertEquals(word, said.toString().strip());
            }
            IOException bad = assertThrows(IOException.class, () -> held.chat(new ChatRequest("tiny", "",
                    List.of(new ChatMessage("user", "What is this?")), 8, null, null, Sampler.NONE,
                    new Format("grammar", null, "root ::= (")), text -> { }));
            assertTrue(bad.getMessage().startsWith("the grammar cannot be used"), bad.getMessage());

            assertTrue(serving.stop());
            assertFalse(serving.stop());
            assertEquals(Map.of("running", false), serving.state());
        }
    }
}
