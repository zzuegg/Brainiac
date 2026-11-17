package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** What the person sets per provider: models shown, Auto, address, options; and Hugging Face's list. */
class ProviderSettingsTest {

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

    private static Map<String, Object> provider(Map<String, Object> d, String id) {
        return Json.array(d.get("providers")).stream().map(Json::object).filter(p -> p.get("id").equals(id))
                .findFirst().orElseThrow();
    }

    private static Map<String, Object> model(Map<String, Object> d, String key) {
        return Json.array(d.get("models")).stream().map(Json::object).filter(m -> key.equals(m.get("key")))
                .findFirst().orElse(null);
    }

    @Test
    void huggingFaceListsLiveHostsWithTheCheapestPriceAndSkipsModelsNobodyServes() {
        Map<String, Object> d = service(Map.of("HF_TOKEN", "hf_x")).describe(true);
        Map<String, Object> oss = model(d, "huggingface/openai/gpt-oss-120b");
        assertEquals("131k context · 2 hosts · from $0.15 in / $0.60 out per 1M", oss.get("detail"),
                "the offline host is neither counted nor priced");
        assertEquals("131k context · 1 host", model(d, "huggingface/meta-llama/Llama-3.3-70B-Instruct").get("detail"));
        assertNull(model(d, "huggingface/old/retired-model"));
        assertEquals(List.of(ProviderSettings.BILL_TO),
                List.copyOf(Json.object(Json.object(provider(d, "huggingface").get("settings")).get("options")).keySet()));
    }

    @Test
    void hiddenModelsAndProvidersOffForAutoAreSkipped() throws Exception {
        ChatService chat = service(Map.of("ANTHROPIC_API_KEY", "sk-ant", "DEEPSEEK_API_KEY", "sk-ds"));
        assertEquals("anthropic/claude-opus-5-5", Json.object(chat.describe(true).get("auto")).get("text"));

        chat.configure("anthropic", null, Optional.of(List.of("claude-haiku-5-5")), null, null);
        Map<String, Object> d = chat.describe(false);
        assertEquals(false, model(d, "anthropic/claude-opus-5-5").get("shown"));
        assertEquals(true, model(d, "anthropic/claude-haiku-5-5").get("shown"));
        assertEquals("anthropic/claude-haiku-5-5", Json.object(d.get("auto")).get("text"));

        chat.configure("anthropic", false, null, null, null);
        assertEquals("deepseek/deepseek-chat", Json.object(chat.describe(false).get("auto")).get("text"));
        // a hidden model can still be asked for by name
        StringBuilder text = new StringBuilder();
        chat.chat("anthropic/claude-opus-5-5", "", List.of(new ChatMessage("user", "Hi")), 10, null, m -> { }, text::append);
        assertEquals("Hello there", text.toString());

        chat.configure("anthropic", true, Optional.empty(), null, null);
        Map<String, Object> s = Json.object(provider(chat.describe(false), "anthropic").get("settings"));
        assertEquals(true, s.get("auto"));
        assertNull(s.get("models"));
    }

    @Test
    void settingsAreSavedAndReadBackAfterARestart() throws Exception {
        ChatService chat = service(Map.of("HF_TOKEN", "hf_x"));
        chat.configure("huggingface", false, Optional.of(List.of("openai/gpt-oss-120b")), null,
                Map.of(ProviderSettings.BILL_TO, "my-lab"));
        String file = Files.readString(dir.resolve("providers.properties"));
        assertFalse(file.contains("hf_x"), "no keys in the settings file");

        ChatService again = service(Map.of("HF_TOKEN", "hf_x"));
        Map<String, Object> s = Json.object(provider(again.describe(true), "huggingface").get("settings"));
        assertEquals(false, s.get("auto"));
        assertEquals(List.of("openai/gpt-oss-120b"), s.get("models"));
        assertEquals("my-lab", Json.object(s.get("options")).get(ProviderSettings.BILL_TO));

        StringBuilder text = new StringBuilder();
        again.chat("huggingface/openai/gpt-oss-120b", "", List.of(new ChatMessage("user", "Hi")), 10, null, m -> { },
                text::append);
        assertEquals("Hello from the hub", text.toString());
        assertEquals("my-lab", apis.headers.get("h").getFirst("X-HF-Bill-To"));
        assertEquals("Bearer hf_x", apis.headers.get("h").getFirst("Authorization"));
    }

    @Test
    void anotherAddressIsUsedAndCanBeReset() throws Exception {
        ChatService chat = service(Map.of("DEEPSEEK_API_KEY", "sk-ds"));
        chat.configure("deepseek", null, null, apis.base("d") + "gateway", null);
        Map<String, Object> s = Json.object(provider(chat.describe(true), "deepseek").get("settings"));
        assertEquals(apis.base("d") + "gateway/", s.get("url"));
        assertEquals(1, apis.calls.getOrDefault("d /gateway/models", 0));
        chat.configure("deepseek", null, null, "", null);
        chat.describe(true);
        assertEquals(1, apis.calls.getOrDefault("d /models", 0));
        assertEquals("", Json.object(provider(chat.describe(false), "deepseek").get("settings")).get("url"));
    }

    @Test
    void badSettingsAreRefused() {
        ChatService chat = service(Map.of());
        assertThrows(IllegalArgumentException.class, () -> chat.configure("deepseek", null, null, "ftp://x/", null));
        assertThrows(IllegalArgumentException.class, () -> chat.configure("deepseek", null, null, "https://u:p@x/", null));
        assertThrows(IllegalArgumentException.class, () -> chat.configure("deepseek", null, null, null,
                Map.of(ProviderSettings.BILL_TO, "org")), "only Hugging Face bills an organization");
        assertThrows(IllegalArgumentException.class, () -> chat.configure("huggingface", null, null, null,
                Map.of(ProviderSettings.BILL_TO, "bad org\nX: y")));
        assertThrows(IllegalArgumentException.class, () -> chat.configure("local", true, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> chat.configure("nobody", true, null, null, null));
        assertTrue(Files.notExists(dir.resolve("providers.properties")));
    }
}
