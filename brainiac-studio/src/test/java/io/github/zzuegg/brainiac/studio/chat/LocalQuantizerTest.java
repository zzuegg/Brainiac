package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A chat model on this PC gets a smaller copy: a quantized GGUF file next to it, listed and usable. */
class LocalQuantizerTest {

    private static final Path LLAMA = Path.of("../brainiac-llm/src/test/resources/io/github/zzuegg/brainiac/llm/llama");

    @TempDir
    Path models;

    private static Map<String, Object> job(LocalQuantizer q) {
        return Json.object(q.state().get("job"));
    }

    /**
     * Puts a tiny Llama with a byte-level BPE tokenizer of single letters, the kind a GGUF file can
     * carry, into a model folder.
     */
    static Path tinyLlama(Path models, String name) throws IOException {
        Path dir = Files.createDirectories(models.resolve(name));
        for (String f : List.of("config.json", "model.safetensors")) {
            Files.copy(LLAMA.resolve(f), dir.resolve(f));
        }
        // a byte-level BPE tokenizer of single letters: the kind a GGUF file can carry
        Map<String, Object> vocab = new LinkedHashMap<>();
        List<String> pieces = new java.util.ArrayList<>(List.of("<unk>", "<s>", "</s>", "\u0120"));
        for (char c : "Whatisto?Abx.enrdlmucgpyfvkw,I!TSj-qz1023456789".toCharArray()) {
            if (!pieces.contains(String.valueOf(c))) {
                pieces.add(String.valueOf(c));
            }
        }
        for (int i = 0; i < 48; i++) {
            vocab.put(pieces.get(i), i);
        }
        Files.writeString(dir.resolve("tokenizer.json"), Json.write(Map.of(
                "added_tokens", List.of(Map.of("id", 0, "content", "<unk>", "special", true),
                        Map.of("id", 1, "content", "<s>", "special", true),
                        Map.of("id", 2, "content", "</s>", "special", true)),
                "model", Map.of("type", "BPE", "vocab", vocab, "merges", List.of(), "unk_token", "<unk>"),
                "pre_tokenizer", Map.of("type", "ByteLevel", "add_prefix_space", false),
                "decoder", Map.of("type", "ByteLevel"))));
        return dir;
    }

    @Test
    void aSmallerCopyIsWrittenNextToTheModelAndAnswers() throws Exception {
        Path dir = tinyLlama(models, "tiny-llama");
        LocalVisionTest.tinyLlava(models, "tiny-eyes", null);

        LocalProvider local = (LocalProvider) ChatService.localModels(List.of(models));
        LocalQuantizer q = new LocalQuantizer(local);
        assertFalse(q.state().containsKey("job"));
        assertEquals("Q4_K_M", Json.object(Json.array(q.state().get("sizes")).getFirst()).get("id"));

        assertThrows(IOException.class, () -> q.start("no-such-model", "Q8_0"));
        assertThrows(IllegalArgumentException.class, () -> q.start("tiny-llama", "Q3_XXS"));
        IOException eyes = assertThrows(IOException.class, () -> q.start("tiny-eyes", "Q8_0"));
        assertTrue(eyes.getMessage().contains("looks at pictures"), eyes.getMessage());

        q.start("tiny-llama", "Q8_0");
        for (int i = 0; i < 600 && q.running(); i++) {
            Thread.sleep(50);
        }
        Map<String, Object> done = job(q);
        assertEquals("done", done.get("state"), String.valueOf(done.get("error")));
        Path copy = models.resolve("tiny-llama-Q8_0.gguf");
        assertEquals(copy.toString(), done.get("file"));
        assertTrue(Files.size(copy) > 1000 && Files.size(copy) < Files.size(dir.resolve("model.safetensors")),
                "8 bits a weight is smaller than 32");
        assertEquals(Files.size(copy), ((Number) done.get("bytes")).longValue());
        try (var files = Files.list(models)) {
            assertTrue(files.noneMatch(p -> p.toString().endsWith(".part")));
        }

        // the copy is one of this PC's models now, and answers
        assertTrue(local.models().stream().anyMatch(m -> m.id().equals("tiny-llama-Q8_0.gguf")),
                local.models().toString());
        ChatResult r = local.chat(new ChatRequest("tiny-llama-Q8_0.gguf", "",
                List.of(new ChatMessage("user", "What is this?")), 3, null), text -> { });
        assertTrue(r.outputTokens() > 0);

        IOException again = assertThrows(IOException.class, () -> q.start("tiny-llama", "Q8_0"));
        assertTrue(again.getMessage().contains("there already"), again.getMessage());

        // a copy of the copy, in another size
        q.start("tiny-llama-Q8_0.gguf", "F16");
        for (int i = 0; i < 600 && q.running(); i++) {
            Thread.sleep(50);
        }
        assertEquals("done", job(q).get("state"), String.valueOf(job(q).get("error")));
        assertTrue(Files.exists(models.resolve("tiny-llama-Q8_0-F16.gguf")));
    }
}
