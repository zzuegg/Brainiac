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
        assertEquals(Map.of("type", "adaptive", "display", "summarized"), body.get("thinking"),
                "without a display the thinking blocks arrive empty");
        chat.describe(true);   // the list says only Opus takes adaptive thinking
        chat.chat("anthropic/claude-haiku-5-5", "", HELLO, 100, null, m -> { }, new Heard());
        assertFalse(Json.object(Json.parse(apis.bodies.get("a POST"))).containsKey("thinking"),
                "a model the list does not mark as adaptive gets no thinking setting");
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
        assertFalse(Json.object(body.get("generationConfig")).containsKey("temperature"), "left to the model unless asked");
    }

    @Test
    @SuppressWarnings("unchecked")
    void picturesGoToEachProviderInItsOwnFormAndNotToModelsThatCannotSee() throws Exception {
        ChatService chat = apis.service(keys(Map.of("ANTHROPIC_API_KEY", "a", "OPENAI_API_KEY", "o",
                "GEMINI_API_KEY", "g")), List.of(dir));
        byte[] png = {(byte) 0x89, 'P', 'N', 'G'};
        String b64 = java.util.Base64.getEncoder().encodeToString(png);
        List<ChatMessage> shown = List.of(new ChatMessage("user", "What is this?", List.of(new Media(png, "image/png"))));

        chat.chat("anthropic/claude-opus-5-5", "", shown, 50, null, m -> { }, new Heard());
        Map<String, Object> a = Json.object(Json.array(Json.object(Json.parse(apis.bodies.get("a POST"))).get("messages")).getFirst());
        List<Object> parts = Json.array(a.get("content"));
        assertEquals(Map.of("type", "base64", "media_type", "image/png", "data", b64), Json.object(parts.get(0)).get("source"));
        assertEquals("What is this?", Json.object(parts.get(1)).get("text"));

        chat.chat("openai/gpt-5", "", shown, 50, null, m -> { }, new Heard());
        Map<String, Object> o = Json.object(Json.array(Json.object(Json.parse(apis.bodies.get("o POST"))).get("messages")).getFirst());
        assertEquals(Map.of("url", "data:image/png;base64," + b64), Json.object(Json.array(o.get("content")).get(1)).get("image_url"));

        chat.chat("google/gemini-3-pro", "", shown, 50, null, m -> { }, new Heard());
        Map<String, Object> g = Json.object(Json.array(Json.object(Json.parse(apis.bodies.get("g POST"))).get("contents")).getFirst());
        assertEquals(Map.of("mimeType", "image/png", "data", b64), Json.object(Json.array(g.get("parts")).get(0)).get("inlineData"));

        assertThrows(IllegalArgumentException.class, () -> new ChatMessage("assistant", "x", List.of(new Media(png, "image/png"))));
        assertThrows(IllegalArgumentException.class, () -> new ChatMessage("user", "x", List.of(new Media(png, "application/pdf"))));
    }

    @Test
    void aTemperatureGoesOnlyToModelsThatTakeOne() throws Exception {
        ChatService chat = apis.service(keys(Map.of("ANTHROPIC_API_KEY", "a", "OPENAI_API_KEY", "o",
                "GEMINI_API_KEY", "g")), List.of());
        chat.chat("google/gemini-3-pro", "", HELLO, 50, null, 0.2f, m -> { }, new Heard());
        Map<String, Object> config = Json.object(Json.object(Json.parse(apis.bodies.get("g POST"))).get("generationConfig"));
        assertEquals(0.2, ((Number) config.get("temperature")).doubleValue(), 1e-6);
        chat.chat("openai/gpt-5", "", HELLO, 50, null, 0.2f, m -> { }, new Heard());
        assertFalse(Json.object(Json.parse(apis.bodies.get("o POST"))).containsKey("temperature"),
                "OpenAI's reasoning models answer only at their own temperature");
        chat.chat("anthropic/claude-opus-5-5", "", HELLO, 50, null, 0.2f, m -> { }, new Heard());
        assertFalse(Json.object(Json.parse(apis.bodies.get("a POST"))).containsKey("temperature"));
        assertThrows(IllegalArgumentException.class,
                () -> chat.chat("google/gemini-3-pro", "", HELLO, 50, null, 3f, m -> { }, new Heard()));

        // top-p and top-k go where the API has them, and a repeat penalty only to models on this PC
        Sampler sampler = new Sampler(0.8f, 40, 1.1f);
        chat.chat("google/gemini-3-pro", "", HELLO, 50, null, null, sampler, m -> { }, new Heard());
        config = Json.object(Json.object(Json.parse(apis.bodies.get("g POST"))).get("generationConfig"));
        assertEquals(0.8, ((Number) config.get("topP")).doubleValue(), 1e-6);
        assertEquals(40, ((Number) config.get("topK")).intValue());
        assertFalse(config.toString().contains("epeat"), config.toString());
        chat.chat("anthropic/claude-opus-5-5", "", HELLO, 50, null, null, sampler, m -> { }, new Heard());
        Map<String, Object> anthropic = Json.object(Json.parse(apis.bodies.get("a POST")));
        assertFalse(anthropic.containsKey("top_p") || anthropic.containsKey("top_k"), anthropic.keySet().toString());
        assertThrows(IllegalArgumentException.class, () -> new Sampler(1.5f, null, null));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(null, 0, null));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(null, null, 0.5f));
        // on this PC: the usual top-p unless one is given, and greedy stays greedy
        assertEquals(0.9f, LocalProvider.sampling(null, Sampler.NONE).topP());
        io.github.zzuegg.brainiac.llm.Sampling local = LocalProvider.sampling(0.5f, sampler);
        assertEquals(0.8f, local.topP());
        assertEquals(40, local.topK());
        assertEquals(1.1f, local.repetitionPenalty());
        assertEquals(0f, LocalProvider.sampling(0f, new Sampler(0.8f, 40, null)).temperature());
    }

    @Test
    void anAnswersFormatIsAskedOfEachProviderInItsOwnWay() throws Exception {
        ChatService chat = apis.service(keys(Map.of("ANTHROPIC_API_KEY", "a", "OPENAI_API_KEY", "o",
                "GEMINI_API_KEY", "g")), List.of());
        Map<String, Object> open = Json.object(Json.parse(
                "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},"
                        + "\"where\":{\"type\":\"object\",\"properties\":{\"lat\":{\"type\":\"number\"}}}}}"));
        Format schema = new Format("schema", open, null);

        // a JSON object: OpenAI's JSON mode, and the instructions say so for the providers that can only be asked
        chat.chat("openai/gpt-5", "Be brief.", HELLO, 50, null, null, Sampler.NONE, Format.JSON, m -> { }, new Heard());
        Map<String, Object> o = Json.object(Json.parse(apis.bodies.get("o POST")));
        assertEquals(Map.of("type", "json_object"), o.get("response_format"));
        assertTrue(o.get("messages").toString().contains("Be brief.\n\n" + Format.JSON_HINT), o.get("messages").toString());
        chat.chat("anthropic/claude-opus-5-5", "", HELLO, 50, null, null, Sampler.NONE, Format.JSON, m -> { }, new Heard());
        Map<String, Object> a = Json.object(Json.parse(apis.bodies.get("a POST")));
        assertEquals(Format.JSON_HINT, a.get("system"));
        assertFalse(a.containsKey("output_config"));

        // a schema: each API's structured output; strict at OpenAI only for a schema that is closed already
        chat.chat("openai/gpt-5", "", HELLO, 50, null, null, Sampler.NONE, schema, m -> { }, new Heard());
        Map<String, Object> format = Json.object(Json.object(Json.parse(apis.bodies.get("o POST"))).get("response_format"));
        assertEquals("json_schema", format.get("type"));
        assertEquals(open, Json.object(format.get("json_schema")).get("schema"));
        assertFalse(Json.object(format.get("json_schema")).containsKey("strict"));
        Format closed = new Format("schema", schema.closedSchema(), null);
        chat.chat("openai/gpt-5", "", HELLO, 50, null, null, Sampler.NONE, closed, m -> { }, new Heard());
        format = Json.object(Json.object(Json.parse(apis.bodies.get("o POST"))).get("response_format"));
        assertEquals(true, Json.object(format.get("json_schema")).get("strict"));

        chat.chat("google/gemini-3-pro", "", HELLO, 50, null, null, Sampler.NONE, schema, m -> { }, new Heard());
        Map<String, Object> config = Json.object(Json.object(Json.parse(apis.bodies.get("g POST"))).get("generationConfig"));
        assertEquals("application/json", config.get("responseMimeType"));
        assertEquals(open, config.get("responseJsonSchema"));

        // Claude wants every object closed, so the studio closes them
        chat.chat("anthropic/claude-opus-5-5", "", HELLO, 50, null, null, Sampler.NONE, schema, m -> { }, new Heard());
        a = Json.object(Json.parse(apis.bodies.get("a POST")));
        Map<String, Object> sent = Json.object(Json.object(Json.object(a.get("output_config")).get("format")).get("schema"));
        assertEquals("json_schema", Json.object(Json.object(a.get("output_config")).get("format")).get("type"));
        assertEquals(false, sent.get("additionalProperties"));
        assertEquals(false, Json.object(Json.object(sent.get("properties")).get("where")).get("additionalProperties"));
        assertFalse(a.containsKey("system"), "a schema needs no asking in words");

        // no provider takes a grammar, and says so before anything is sent
        int before = apis.calls.values().stream().mapToInt(Integer::intValue).sum();
        IOException e = assertThrows(IOException.class, () -> chat.chat("openai/gpt-5", "", HELLO, 50, null, null,
                Sampler.NONE, new Format("grammar", null, "root ::= \"yes\" | \"no\""), m -> { }, new Heard()));
        assertTrue(e.getMessage().contains("takes no grammar"), e.getMessage());
        assertEquals(before, apis.calls.values().stream().mapToInt(Integer::intValue).sum());

        // reading a request's format
        assertTrue(Format.of(Map.of()).none());
        assertEquals(Format.JSON, Format.of(Map.of("format", Map.of("type", "json"))));
        assertEquals(open, Format.of(Map.of("format", Map.of("type", "schema", "schema", Json.write(open)))).schema());
        assertEquals(schema, Format.of(Map.of("format", schema.toJson())));
        assertThrows(IllegalArgumentException.class, () -> Format.of(Map.of("format", Map.of("type", "schema"))));
        assertThrows(IllegalArgumentException.class,
                () -> Format.of(Map.of("format", Map.of("type", "schema", "schema", "{not json"))));
        assertThrows(IllegalArgumentException.class, () -> Format.of(Map.of("format", Map.of("type", "grammar"))));
        assertThrows(IllegalArgumentException.class, () -> Format.of(Map.of("format", Map.of("type", "yaml"))));
        assertThrows(IllegalArgumentException.class, () -> new Format("grammar", null, "root ::= (").constraint());
        assertTrue(Format.JSON.constraint().matches("{\"a\": 1}"));
        assertFalse(Format.JSON.constraint().matches("[1]"));
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
        assertEquals(Map.of("text", "openai/gpt-5", "image", "openai/gpt-image-1", "speech", "openai/gpt-4o-mini-tts"),
                d.get("auto"));
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
        assertTrue(e.getMessage().contains("(out of credit): Your credit balance is too low"), e.getMessage());
        // the model list joins what each key sees, and names the key that failed
        Map<String, Object> anthropic = Json.array(chat.describe(true).get("providers")).stream().map(Json::object)
                .filter(p -> p.get("id").equals("anthropic")).findFirst().orElseThrow();
        assertEquals(2L, ((Number) anthropic.get("models")).longValue());
        List<Map<String, Object>> ks = Json.array(anthropic.get("keys")).stream().map(Json::object).toList();
        assertTrue(String.valueOf(ks.get(0).get("error")).contains("400 (out of credit)"), ks.toString());
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

    @Test
    void openAiDrawsAndSpeaksAndEverythingIsSaved() throws Exception {
        ChatService chat = apis.service(keys(Map.of("OPENAI_API_KEY", "o")), List.of(dir));
        ChatService.Generated pics = chat.images("openai/gpt-image-1", "a lighthouse", "16:9", 2, null);
        assertEquals(2, pics.media().size());
        Map<String, Object> body = Json.object(Json.parse(apis.bodies.get("o POST")));
        assertEquals("1536x1024", body.get("size"));
        assertFalse(body.containsKey("response_format"), "GPT image models reject response_format");
        assertEquals("PNG1", Files.readString(pics.files().get(1)));
        assertTrue(pics.files().getFirst().getFileName().toString().matches("image-\\d{8}-\\d{6}-1\\.png"));

        chat.images("openai/dall-e-3", "a lighthouse", "9:16", 2, null);
        assertEquals(2, apis.calls.get("o /v1/images/generations") - 1, "DALL-E 3 draws one picture per call");
        assertEquals("1024x1792", Json.object(Json.parse(apis.bodies.get("o POST"))).get("size"));

        chat.describe(false);
        ChatService.Generated voice = chat.speech(ChatService.AUTO, "Hello", "coral", 1.1, "calm", "mp3", null);
        assertEquals("openai/gpt-4o-mini-tts", voice.model().key());
        assertEquals("audio/mpeg", voice.media().getFirst().mime());
        assertTrue(voice.files().getFirst().toString().endsWith(".mp3"));
        Map<String, Object> said = Json.object(Json.parse(apis.bodies.get("o POST")));
        assertEquals("calm", said.get("instructions"));
        assertEquals(1.1, ((Number) said.get("speed")).doubleValue(), 1e-9);
    }

    @Test
    void googleDrawsWithImagenAndSpeaksAsWav() throws Exception {
        ChatService chat = apis.service(keys(Map.of("GEMINI_API_KEY", "g")), List.of(dir));
        Map<String, Object> d = chat.describe(false);
        assertEquals(Map.of("text", "google/gemini-3-pro", "image", "google/imagen-4.0-generate-001",
                "speech", "google/gemini-2.5-flash-preview-tts"), d.get("auto"));
        Map<String, Object> tts = Json.array(d.get("models")).stream().map(Json::object)
                .filter(m -> m.get("key").equals("google/gemini-2.5-flash-preview-tts")).findFirst().orElseThrow();
        Map<String, Object> options = Json.object(tts.get("speech"));
        assertTrue(Json.array(options.get("voices")).contains("Kore"));
        assertEquals(false, options.get("speed"));

        ChatService.Generated pics = chat.images(ChatService.AUTO, "a lighthouse", "9:16", 2, null);
        assertEquals(List.of("IMG-A", "IMG-B"), pics.media().stream().map(m -> new String(m.data(), StandardCharsets.UTF_8)).toList());
        Map<String, Object> params = Json.object(Json.object(Json.parse(apis.bodies.get("g POST"))).get("parameters"));
        assertEquals("9:16", params.get("aspectRatio"));

        ChatService.Generated voice = chat.speech(ChatService.AUTO, "Hello", "Kore", 1, "cheerfully", "mp3", null);
        byte[] wav = voice.media().getFirst().data();
        assertEquals("audio/wav", voice.media().getFirst().mime());
        assertEquals("RIFF", new String(wav, 0, 4, StandardCharsets.US_ASCII));
        assertEquals(24000, java.nio.ByteBuffer.wrap(wav, 24, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt());
        assertEquals("abcd", new String(wav, 44, 4, StandardCharsets.US_ASCII));
        String text = String.valueOf(Json.object(Json.array(Json.object(Json.array(Json.object(Json.parse(
                apis.bodies.get("g POST"))).get("contents")).getFirst()).get("parts")).getFirst()).get("text"));
        assertEquals("cheerfully: Hello", text, "Gemini speech takes its directions in the text");
    }

    @Test
    void picturesNeedARemoteProviderAndSaneSettings() {
        ChatService chat = apis.service(keys(Map.of()), List.of(dir));
        IOException e = assertThrows(IOException.class, () -> chat.images(ChatService.AUTO, "x", "1:1", 1, null));
        assertTrue(e.getMessage().contains("OpenAI or Google"), e.getMessage());
        ChatService withKey = apis.service(keys(Map.of("OPENAI_API_KEY", "o")), List.of(dir));
        assertThrows(IllegalArgumentException.class, () -> withKey.images("openai/gpt-image-1", "x", "2:1", 1, null));
        assertThrows(IllegalArgumentException.class, () -> withKey.images("openai/gpt-image-1", "x", "1:1", 9, null));
        assertThrows(IllegalArgumentException.class, () -> withKey.speech("openai/tts-1", "", "alloy", 1, "", "mp3", null));
    }
}
