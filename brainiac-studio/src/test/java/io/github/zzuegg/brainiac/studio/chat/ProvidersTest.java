package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The OpenAI-format providers, balances and the usage log. */
class ProvidersTest {

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

    private ChatService service(Map<String, String> env) {
        return apis.service(new Keys(dir.resolve("keys.properties"), env), List.of(dir));
    }

    private static final List<ChatMessage> HELLO = List.of(new ChatMessage("user", "Say hello"));

    private static final ChatListener QUIET = t -> { };

    @Test
    void everyPresetHasItsOwnIdAndAnEnvironmentVariable() {
        List<String> ids = Preset.COMPATIBLE.stream().map(Preset::id).toList();
        assertEquals(ids.size(), ids.stream().distinct().count());
        assertTrue(ids.containsAll(List.of("huggingface", "deepseek", "qwen", "moonshot", "zhipu", "minimax")), ids.toString());
        for (Preset p : Preset.COMPATIBLE) {
            assertFalse(p.env().isEmpty(), p.id());
            assertTrue(Keys.ENV.get(p.id()).containsAll(p.env()), p.id());
            assertTrue(p.base().toString().startsWith("https://") && p.base().toString().endsWith("/"), p.id());
        }
    }

    @Test
    void deepSeekStreamsWithMaxTokensAndAsksForUsage() throws Exception {
        ChatService chat = service(Map.of("DEEPSEEK_API_KEY", "sk-ds"));
        StringBuilder text = new StringBuilder();
        ChatResult r = chat.chat("deepseek/deepseek-chat", "Be brief.", HELLO, 64, null, m -> { }, text::append);
        assertEquals("Ni hao", text.toString());
        assertEquals(new ChatResult("end", 7, 2), r);
        assertEquals("Bearer sk-ds", apis.headers.get("d").getFirst("Authorization"));
        Map<String, Object> body = Json.object(Json.parse(apis.bodies.get("d POST")));
        assertEquals(64L, ((Number) body.get("max_tokens")).longValue());
        assertFalse(body.containsKey("max_completion_tokens"));
        assertEquals(Map.of("include_usage", true), body.get("stream_options"));
        assertEquals("system", Json.object(Json.array(body.get("messages")).getFirst()).get("role"));
    }

    @Test
    void openRouterListsChatModelsWithTheirContextAndSkipsEmbeddings() throws Exception {
        ChatService chat = service(Map.of("OPENROUTER_API_KEY", "sk-or"));
        Map<String, Object> d = chat.describe(true);
        Map<String, Object> or = Json.array(d.get("providers")).stream().map(Json::object)
                .filter(p -> p.get("id").equals("openrouter")).findFirst().orElseThrow();
        assertEquals(1L, ((Number) or.get("models")).longValue(), or.toString());
        Map<String, Object> qwen = Json.array(d.get("models")).stream().map(Json::object)
                .filter(m -> "openrouter/qwen/qwen3-vl-235b".equals(m.get("key"))).findFirst().orElseThrow();
        assertEquals(List.of("text", "vision"), qwen.get("media"));
        assertEquals("262k context", qwen.get("detail"));
    }

    @Test
    void balancesComeFromProvidersThatReportThem() throws Exception {
        ChatService chat = service(Map.of("DEEPSEEK_API_KEY", "sk-ds", "OPENROUTER_API_KEY", "sk-or",
                "ANTHROPIC_API_KEY", "sk-ant"));
        Map<String, Object> b = chat.balances();
        assertEquals(List.of("deepseek", "openrouter"), List.copyOf(b.keySet()), "Anthropic has no balance API");
        Map<String, Object> ds = Json.object(Json.object(((List<?>) b.get("deepseek")).getFirst()).get("balance"));
        assertEquals(110.0, ((Number) ds.get("amount")).doubleValue());
        assertEquals("CNY", ds.get("currency"));
        Map<String, Object> or = Json.object(Json.object(((List<?>) b.get("openrouter")).getFirst()).get("balance"));
        assertEquals(12.5, ((Number) or.get("amount")).doubleValue());
        assertTrue(((String) or.get("detail")).contains("7.50"), or.toString());

        apis.failWith = 401;
        Map<String, Object> failed = Json.object(((List<?>) chat.balances().get("deepseek")).getFirst());
        assertTrue(String.valueOf(failed.get("error")).contains("401"), failed.toString());
    }

    @Test
    void everyCallIsLoggedWithItsKeyTokensAndOutcome() throws Exception {
        ChatService chat = service(Map.of("DEEPSEEK_API_KEY", "sk-ds"));
        chat.chat("deepseek/deepseek-chat", "", HELLO, 64, null, m -> { }, QUIET);
        apis.failWith = 500;
        assertThrows(IOException.class, () -> chat.chat("deepseek/deepseek-chat", "", HELLO, 64, null, m -> { }, QUIET));

        Map<String, Object> s = chat.usage().summary(ZoneOffset.UTC);
        Map<String, Object> ds = Json.object(Json.object(s.get("providers")).get("deepseek"));
        assertEquals(2L, ((Number) ds.get("calls")).longValue());
        assertEquals(1L, ((Number) ds.get("failed")).longValue());
        assertEquals(7L, ((Number) ds.get("in")).longValue());
        assertEquals(2L, ((Number) ds.get("out")).longValue());
        assertTrue(Json.object(s.get("keys")).containsKey("deepseek/DEEPSEEK_API_KEY"), s.get("keys").toString());
        assertTrue(Json.object(s.get("models")).containsKey("deepseek/deepseek-chat"));
        List<Object> daily = Json.array(s.get("daily"));
        assertEquals(30, daily.size());
        assertTrue(Json.object(Json.object(daily.getLast()).get("providers")).containsKey("deepseek"));
        assertEquals(false, Json.object(Json.array(s.get("recent")).getFirst()).get("ok"));

        String log = Files.readString(dir.resolve("usage.jsonl"));
        assertFalse(log.contains("sk-ds"), "keys appear by name only");
        assertFalse(log.contains("Say hello"), "prompts are not kept");
    }

    @Test
    void theLogSurvivesARestartAndSkipsALineCutShort() throws Exception {
        Path file = dir.resolve("usage.jsonl");
        UsageLog log = new UsageLog(file);
        log.add(new UsageLog.Entry(System.currentTimeMillis(), "openai", "work", "gpt-image-1", "image", -1, -1, 2, 3.5, true));
        Files.writeString(file, "{\"at\":1,\"prov", StandardCharsets.UTF_8, java.nio.file.StandardOpenOption.APPEND);
        Map<String, Object> s = new UsageLog(file).summary(ZoneOffset.UTC);
        Map<String, Object> openai = Json.object(Json.object(s.get("providers")).get("openai"));
        assertEquals(1L, ((Number) openai.get("calls")).longValue());
        assertEquals(2L, ((Number) openai.get("units")).longValue());
        assertEquals(0L, ((Number) openai.get("in")).longValue(), "unknown counts add nothing");
    }
}
