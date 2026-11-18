package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Bart;
import io.github.zzuegg.brainiac.llm.T5;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** BART, Marian and T5 models on this PC summarize and translate. */
class LocalTextsTest {

    private static final Path FIXTURES = Path.of("../brainiac-llm/src/test/resources/io/github/zzuegg/brainiac/llm");

    @TempDir
    Path models;

    // the fixtures have no tokenizer: a word-level one over their 64 tokens
    private Path tiny(String fixture, String name, String configEdit) throws IOException {
        Path dir = Files.createDirectories(models.resolve(name));
        for (String f : List.of("model.safetensors", "generation_config.json")) {
            Files.copy(FIXTURES.resolve(fixture).resolve(f), dir.resolve(f));
        }
        String config = Files.readString(FIXTURES.resolve(fixture).resolve("config.json"));
        Files.writeString(dir.resolve("config.json"), configEdit == null ? config : config.replaceFirst("\\{", "{" + configEdit + ","));
        Map<String, Object> vocab = new LinkedHashMap<>();
        for (int i = 0; i < 64; i++) {
            vocab.put("w" + i, i);
        }
        Map<String, Object> tokenizer = new LinkedHashMap<>();
        tokenizer.put("model", Map.of("type", "WordLevel", "vocab", vocab, "unk_token", "w3"));
        tokenizer.put("pre_tokenizer", Map.of("type", "WhitespaceSplit"));
        Files.writeString(dir.resolve("tokenizer.json"), Json.write(tokenizer));
        return dir;
    }

    @Test
    void aBartModelIsListedAndWritesWhatTheModelWrites() throws Exception {
        Path dir = tiny("bart", "tiny-bart", null);
        LocalProvider local = new LocalProvider(List.of(models));
        ModelInfo listed = local.models().stream().filter(m -> m.id().equals("tiny-bart")).findFirst().orElseThrow();
        assertEquals(List.of(LocalTexts.MEDIUM), listed.media());
        assertFalse(listed.chats(), "a summarizer is not a chat model");

        String text = "w10 w11 w12 w13";
        String expected = Bart.load(dir, Device.cpu(), DType.FLOAT32).generate(PretrainedTokenizer.load(dir), text, 6).strip();
        LocalTexts.Rewritten r = local.texts().rewrite("tiny-bart", text, "", 6);
        assertEquals(expected, r.text());
        assertFalse(r.text().isEmpty());
    }

    @Test
    void aT5TaskPutsItsPrefixBeforeTheText() throws Exception {
        Path dir = tiny("t5-relu", "tiny-t5",
                "\"task_specific_params\": {\"summarization\": {\"prefix\": \"w5 w6 \", \"max_length\": 5}}");
        LocalProvider local = new LocalProvider(List.of(models));
        List<LocalTexts.Task> tasks = local.texts().tasks("tiny-t5");
        assertEquals(List.of(new LocalTexts.Task("summarization", "w5 w6 ", 5)), tasks);
        assertTrue(local.models().stream().anyMatch(m -> m.id().equals("tiny-t5") && m.detail().contains("summarization")));

        String expected = T5.load(dir, Device.cpu(), DType.FLOAT32).generate(PretrainedTokenizer.load(dir), "w5 w6 w20 w21", 5)
                .strip();
        assertEquals(expected, local.texts().rewrite("tiny-t5", "w20 w21", "summarization", 0).text());
        assertThrows(IllegalArgumentException.class, () -> local.texts().rewrite("tiny-t5", "w20", "translation", 0));
        assertThrows(IOException.class, () -> local.texts().rewrite("missing", "w20", "", 0));
    }
}
