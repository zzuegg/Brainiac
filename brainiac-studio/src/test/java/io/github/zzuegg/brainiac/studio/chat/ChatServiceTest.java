package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ChatServiceTest {

    @TempDir
    Path dir;
    private FakeApis apis;

    @BeforeEach
    void start() throws IOException {
        apis = new FakeApis();
    }

    @AfterEach
    void stop() {
        apis.close();
    }

    private Keys keys(Map<String, String> env) {
        return new Keys(dir.resolve("keys.properties"), env);
    }

    private static final List<ChatMessage> HELLO = List.of(new ChatMessage("user", "Say hello"));

    private record Heard(StringBuilder text, StringBuilder thinking, List<ModelInfo> routes) implements ChatListener {
        Heard() {
            this(new StringBuilder(), new StringBuilder(), new ArrayList<>());
        }

        @Override
        public void text(String t) {
            text.append(t);
        }

        @Override
        public void thinking(String t) {
            thinking.append(t);
        }
    }

    @Test
    void anthropicStreamsTextAndThinkingWithTheKeyInItsHeader() throws Exception {
        ChatService chat = apis.service(keys(Map.of("ANTHROPIC_API_KEY", "sk-ant-test")), List.of());
        Heard heard = new Heard();
        ChatResult r = chat.chat("anthropic/claude-opus-5-5", "Be brief.", HELLO, 100, null, heard.routes()::add, heard);
        assertEquals("Hello there", heard.text().toString());
        assertEquals("Hmm.", heard.thinking().toString());
        assertEquals(new ChatResult("end", 12, 3), r);
        assertEquals("sk-ant-test", apis.headers.get("a").getFirst("x-api-key"));
        assertEquals(AnthropicProvider.VERSION, apis.headers.get("a").getFirst("anthropic-version"));
        Map<String, Object> body = Json.object(Json.parse(apis.bodies.get("a POST")));
        assertEquals("claude-opus-5-5", body.get("model"));
        assertEquals("Be brief.", body.get("system"));
        assertEquals(true, body.get("stream"));
        assertFalse(body.containsKey("temperature"), "the current API version rejects sampling parameters");
    }

    @Test
    void openAiStreamsUntilDoneAndReportsTheLengthStop() throws Exception {
        ChatService chat = apis.service(keys(Map.of("OPENAI_API_KEY", "sk-test")), List.of());
        Heard heard = new Heard();
        ChatResult r = chat.chat("openai/gpt-5", "", HELLO, 50, null, heard.routes()::add, heard);
        assertEquals("Hi!", heard.text().toString());
        assertEquals(new ChatResult("length", 9, 2), r);
        assertEquals("Bearer sk-test", apis.headers.get("o").getFirst("Authorization"));
        Map<String, Object> body = Json.object(Json.parse(apis.bodies.get("o POST")));
        assertEquals(50L, ((Number) body.get("max_completion_tokens")).longValue());
    }

    @Test
    void geminiSeparatesThoughtsAndMapsSafetyToRefusal() throws Exception {
        ChatService chat = apis.service(keys(Map.of("GEMINI_API_KEY", "g-test")), List.of());
        Heard heard = new Heard();
        ChatResult r = chat.chat("google/gemini-3-pro", "", HELLO, 50, null, heard.routes()::add, heard);
        assertEquals("Bonjour.", heard.text().toString());
        assertEquals("Planning", heard.thinking().toString());
        assertEquals(new ChatResult("refusal", 5, 6), r);
        assertEquals("g-test", apis.headers.get("g").getFirst("x-goog-api-key"));
        Map<String, Object> body = Json.object(Json.parse(apis.bodies.get("g POST")));
        assertTrue(body.containsKey("contents"));
    }

    @Test
    void modelListsAreSortedIntoMediaAndAutoPicksTheFirstReadyProvider() {
        ChatService chat = apis.service(keys(Map.of("OPENAI_API_KEY", "o", "GEMINI_API_KEY", "g")), List.of(dir));
        Map<String, Object> d = chat.describe(false);
        List<Map<String, Object>> models = Json.array(d.get("models")).stream().map(Json::object).toList();
        Map<String, List<Object>> media = new java.util.HashMap<>();
        models.forEach(m -> media.put((String) m.get("key"), Json.array(m.get("media"))));
        assertEquals(List.of("speech"), media.get("openai/tts-1"));
        assertEquals(List.of("transcribe"), media.get("openai/whisper-1"));
        assertEquals(List.of("image"), media.get("openai/gpt-image-1"));
        assertEquals(List.of("image"), media.get("google/imagen-4.0-generate-001"));
        assertEquals(List.of("video"), media.get("google/veo-3.0-generate-001"));
        assertFalse(media.containsKey("openai/text-embedding-3-large"));
        assertFalse(media.containsKey("openai/gpt-4o-realtime-preview"));
        // no local models and no Anthropic key: OpenAI comes next, and its newest full-size GPT
        assertEquals(Map.of("text", "openai/gpt-5"), d.get("auto"));
        Map<String, Object> anthropic = Json.array(d.get("providers")).stream().map(Json::object)
                .filter(p -> p.get("id").equals("anthropic")).findFirst().orElseThrow();
        assertEquals(false, anthropic.get("ready"));
        assertEquals(List.of(), anthropic.get("keys"));
    }

    @Test
    void autoPrefersClaudeOpusWhenAnthropicHasAKey() throws Exception {
        ChatService chat = apis.service(keys(Map.of("ANTHROPIC_API_KEY", "a", "OPENAI_API_KEY", "o")), List.of(dir));
        Heard heard = new Heard();
        chat.chat(ChatService.AUTO, "", HELLO, 10, null, heard.routes()::add, heard);
        assertEquals("anthropic/claude-opus-5-5", heard.routes().getFirst().key());
    }

    @Test
    void autoWithNothingAvailableSaysWhatToDo() {
        ChatService chat = apis.service(keys(Map.of()), List.of(dir));
        Heard heard = new Heard();
        IOException e = assertThrows(IOException.class,
                () -> chat.chat(ChatService.AUTO, "", HELLO, 10, null, heard.routes()::add, heard));
        assertTrue(e.getMessage().contains("API key"), e.getMessage());
    }

    @Test
    void aProviderErrorCarriesItsOwnMessageAndAHint() {
        ChatService chat = apis.service(keys(Map.of("ANTHROPIC_API_KEY", "bad")), List.of());
        apis.failWith = 401;
        Heard heard = new Heard();
        IOException e = assertThrows(IOException.class,
                () -> chat.chat("anthropic/claude-opus-5-5", "", HELLO, 10, null, heard.routes()::add, heard));
        assertEquals("Anthropic answered 401 (check the API key): invalid x-api-key", e.getMessage());
        Map<String, Object> listed = Json.array(chat.describe(true).get("providers")).stream().map(Json::object)
                .filter(p -> p.get("id").equals("anthropic")).findFirst().orElseThrow();
        assertTrue(String.valueOf(listed.get("error")).contains("401"));
    }

    @Test
    void eachProviderHoldsSeveralNamedKeysOnlyTheOwnerCanRead() throws IOException {
        Keys keys = keys(Map.of("OPENAI_API_KEY", "sk-from-environment"));
        keys.add("openai", "work", " sk-work-000000001111 ");
        keys.add("openai", "personal", "sk-personal-00002222");
        assertEquals(List.of(new Keys.Entry("work", "saved", "…1111"), new Keys.Entry("personal", "saved", "…2222"),
                new Keys.Entry("OPENAI_API_KEY", "environment", "…ment")), keys.list("openai"));
        assertEquals("sk-work-000000001111", keys.get("openai", "work").orElseThrow());
        keys.makeFirst("openai", "personal");
        keys.add("openai", "work", "sk-work-new-00003333");
        assertEquals(List.of("personal", "work", "OPENAI_API_KEY"),
                keys.list("openai").stream().map(Keys.Entry::label).toList(), "replacing a key keeps its place");
        Path file = dir.resolve("keys.properties");
        if (Files.getFileStore(file).supportsFileAttributeView("posix")) {
            assertEquals("rw-------", PosixFilePermissions.toString(Files.getPosixFilePermissions(file)));
        }
        Keys reopened = new Keys(file, Map.of());
        assertEquals(List.of("personal", "work"), reopened.list("openai").stream().map(Keys.Entry::label).toList(),
                "kept across restarts, in order");
        assertTrue(keys.remove("openai", "personal"));
        assertFalse(keys.remove("openai", "OPENAI_API_KEY"), "an environment key is removed in the environment");
        assertTrue(keys.list("anthropic").isEmpty());
        assertThrows(IllegalArgumentException.class, () -> keys.add("openai", "x", "two words"));
        assertThrows(IllegalArgumentException.class, () -> keys.add("openai", "", "k"));
        assertThrows(IllegalArgumentException.class, () -> keys.add("elsewhere", "x", "k"));
    }

    @Test
    void aKeyThatIsOutOfCreditHandsTheChatToTheNextKey() throws Exception {
        Keys keys = keys(Map.of());
        keys.add("anthropic", "empty", "sk-ant-empty");
        keys.add("anthropic", "team", "sk-ant-team");
        ChatService chat = apis.service(keys, List.of());
        apis.failKey = "sk-ant-empty";
        Heard heard = new Heard();
        chat.chat("anthropic/claude-opus-5-5", "", HELLO, 10, null, heard.routes()::add, heard);
        assertEquals("Hello there", heard.text().toString());
        assertEquals("sk-ant-team", apis.headers.get("a").getFirst("x-api-key"));
        // asked for by name, the empty key is not swapped for another
        IOException e = assertThrows(IOException.class, () -> chat.chat("anthropic/claude-opus-5-5", "", HELLO, 10,
                "empty", heard.routes()::add, heard));
        assertTrue(e.getMessage().contains("credit balance too low"), e.getMessage());
        // the model list joins what each key sees, and names the key that failed
        Map<String, Object> anthropic = Json.array(chat.describe(true).get("providers")).stream().map(Json::object)
                .filter(p -> p.get("id").equals("anthropic")).findFirst().orElseThrow();
        assertEquals(2L, ((Number) anthropic.get("models")).longValue());
        List<Map<String, Object>> ks = Json.array(anthropic.get("keys")).stream().map(Json::object).toList();
        assertTrue(String.valueOf(ks.get(0).get("error")).contains("429"), ks.toString());
        assertFalse(ks.get(1).containsKey("error"));
    }

    @Test
    void localModelFoldersAndGgufFilesAreFound() throws IOException {
        Path qwen = Files.createDirectories(dir.resolve("Qwen2.5-0.5B-Instruct"));
        Files.writeString(qwen.resolve("config.json"), "{\"model_type\":\"qwen2\"}");
        Files.writeString(qwen.resolve("tokenizer.json"), "{}");
        Path other = Files.createDirectories(dir.resolve("bert-base"));
        Files.writeString(other.resolve("config.json"), "{\"model_type\":\"bert\"}");
        Files.writeString(other.resolve("tokenizer.json"), "{}");
        Files.write(dir.resolve("small-q8_0.gguf"), new byte[2 << 20]);
        Files.writeString(dir.resolve("notes.txt"), "");
        LocalProvider local = new LocalProvider(List.of(dir, dir.resolve("missing")));
        assertTrue(local.ready());
        List<String> names = local.models().stream().map(ModelInfo::id).toList();
        assertEquals(List.of("Qwen2.5-0.5B-Instruct", "small-q8_0.gguf"), names);
        assertEquals("GGUF · 2 MB", local.models().get(1).detail());
        ChatService chat = apis.service(keys(Map.of("ANTHROPIC_API_KEY", "a")), List.of(dir));
        assertEquals(Map.of("text", "local/Qwen2.5-0.5B-Instruct"), chat.describe(false).get("auto"),
                "auto keeps the conversation on this PC when it can");
    }

    @Test
    void sseJoinsDataLinesAndSkipsComments() throws IOException {
        String stream = ": keep-alive\nevent: a\ndata: one\ndata: two\n\ndata: three\n\ndata: tail";
        List<String> seen = new ArrayList<>();
        Sse.read(new ByteArrayInputStream(stream.getBytes(StandardCharsets.UTF_8)), (e, d) -> seen.add(e + "=" + d));
        assertEquals(List.of("a=one\ntwo", "message=three", "message=tail"), seen);
    }
}
