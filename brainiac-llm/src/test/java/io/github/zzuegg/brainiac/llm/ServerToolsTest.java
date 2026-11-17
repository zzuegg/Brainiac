package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Tool calls through the OpenAI-compatible server, with a model that writes scripted turns. */
class ServerToolsTest {

    private static final String TOOLS = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"weather\","
            + "\"description\":\"temperature\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]";

    private static String post(OpenAiServer server, String body) throws IOException {
        HttpURLConnection c = (HttpURLConnection) URI.create("http://127.0.0.1:" + server.port() + "/v1/chat/completions")
                .toURL().openConnection();
        c.setRequestMethod("POST");
        c.setDoOutput(true);
        try (OutputStream out = c.getOutputStream()) {
            out.write(body.getBytes(StandardCharsets.UTF_8));
        }
        int status = c.getResponseCode();
        try (InputStream in = status < 400 ? c.getInputStream() : c.getErrorStream()) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            assertEquals(200, status, text);
            return text;
        }
    }

    private static int status(OpenAiServer server, String body) throws IOException {
        HttpURLConnection c = (HttpURLConnection) URI.create("http://127.0.0.1:" + server.port() + "/v1/chat/completions")
                .toURL().openConnection();
        c.setRequestMethod("POST");
        c.setDoOutput(true);
        try (OutputStream out = c.getOutputStream()) {
            out.write(body.getBytes(StandardCharsets.UTF_8));
        }
        return c.getResponseCode();
    }

    @Test
    void toolCallsComeBackAsToolCalls() throws IOException {
        PretrainedTokenizer tok = AgentTest.tokenizer();
        AgentTest.Scripted model = new AgentTest.Scripted(List.of(
                "Checking.<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Vienna\"}}</tool_call>",
                "It is 21 C.",
                "<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>",
                "plain",
                "{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\"}}"));
        Generator gen = new Generator(model, tok);
        try (OpenAiServer server = OpenAiServer.start(new InetSocketAddress("127.0.0.1", 0), "s", gen, tok,
                Set.of(AgentTest.END))) {
            Map<String, Object> r = Json.object(Json.parse(post(server,
                    "{\"messages\":[{\"role\":\"user\",\"content\":\"How warm is Vienna?\"}]," + TOOLS + "}")));
            Map<String, Object> choice = Json.object(Json.array(r.get("choices")).getFirst());
            assertEquals("tool_calls", choice.get("finish_reason"));
            Map<String, Object> message = Json.object(choice.get("message"));
            assertEquals("Checking.", message.get("content"));
            Map<String, Object> call = Json.object(Json.array(message.get("tool_calls")).getFirst());
            assertEquals("function", call.get("type"));
            assertTrue(((String) call.get("id")).startsWith("call_"));
            Map<String, Object> function = Json.object(call.get("function"));
            assertEquals("weather", function.get("name"));
            assertEquals("{\"city\":\"Vienna\"}", function.get("arguments"));

            // the client runs the tool and sends the call and its result back
            String followUp = "{\"messages\":[{\"role\":\"user\",\"content\":\"How warm is Vienna?\"},"
                    + "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
                    + "\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Vienna\\\"}\"}}]},"
                    + "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"21 C\"}]," + TOOLS + "}";
            r = Json.object(Json.parse(post(server, followUp)));
            choice = Json.object(Json.array(r.get("choices")).getFirst());
            assertEquals("stop", choice.get("finish_reason"));
            assertEquals("It is 21 C.", Json.object(choice.get("message")).get("content"));

            // streamed: the call arrives as one delta
            String events = post(server, "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Rome?\"}],"
                    + TOOLS + "}");
            assertTrue(events.contains("\"tool_calls\":[{\"index\":0,\"id\":\"call_"), events);
            assertTrue(events.contains("\"finish_reason\":\"tool_calls\""), events);
            assertTrue(events.endsWith("data: [DONE]\n\n"), events);

            // tool_choice none: the tools are not offered and the text is left as it is
            r = Json.object(Json.parse(post(server, "{\"tool_choice\":\"none\",\"messages\":[{\"role\":\"user\","
                    + "\"content\":\"x\"}]," + TOOLS + "}")));
            choice = Json.object(Json.array(r.get("choices")).getFirst());
            assertEquals("plain", Json.object(choice.get("message")).get("content"));

            // a named function is forced: the reply is held to the call's schema
            r = Json.object(Json.parse(post(server, "{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":"
                    + "\"weather\"}},\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Oslo?\"}]," + TOOLS + "}")));
            choice = Json.object(Json.array(r.get("choices")).getFirst());
            assertEquals("tool_calls", choice.get("finish_reason"));
            Map<String, Object> forced = Json.object(Json.object(Json.array(Json.object(choice.get("message"))
                    .get("tool_calls")).getFirst()).get("function"));
            assertEquals("{\"city\":\"Oslo\"}", forced.get("arguments"));
            assertTrue(Json.object(choice.get("message")).containsKey("content"));
            assertEquals(null, Json.object(choice.get("message")).get("content"));
            assertEquals(400, status(server, "{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"nope\"}},"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," + TOOLS + "}"));
            assertEquals(400, status(server, "{\"tool_choice\":\"sometimes\",\"messages\":[{\"role\":\"user\","
                    + "\"content\":\"x\"}]," + TOOLS + "}"));
            // the script has run out, so the model fails: the client still gets an answer
            assertEquals(500, status(server, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}"));
        }
    }
}
