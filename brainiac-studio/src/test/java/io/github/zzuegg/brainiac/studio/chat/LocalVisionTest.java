package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A model on this PC that looks at pictures answers about the ones attached to a message. */
class LocalVisionTest {

    /** The tiny LLaVA the language-model tests check against transformers. */
    private static final Path LLAVA = Path.of("../brainiac-llm/src/test/resources/io/github/zzuegg/brainiac/llm/vlm-llava");

    @TempDir
    Path models;

    private static byte[] png(int rgb) throws IOException {
        BufferedImage image = new BufferedImage(40, 24, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 24; y++) {
            for (int x = 0; x < 40; x++) {
                image.setRGB(x, y, x < 20 ? rgb : 0xffffff);
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, "png", out);
        return out.toByteArray();
    }

    // the fixture has no tokenizer: a word-level one over its 64 tokens, with <image> where its config says
    private Path tinyLlava(String name, String template) throws IOException {
        Path dir = Files.createDirectories(models.resolve(name));
        for (String f : List.of("config.json", "model.safetensors", "preprocessor_config.json", "generation_config.json")) {
            Files.copy(LLAVA.resolve(f), dir.resolve(f));
        }
        List<String> words = new ArrayList<>(List.of("<pad>", "<s>", "</s>", "USER:", "ASSISTANT:", "What", "is", "this?", "And",
                "that?", "A", "box."));
        Map<String, Object> vocab = new LinkedHashMap<>();
        for (int i = 0; i < 64; i++) {
            vocab.put(i == 60 ? "<image>" : i < words.size() ? words.get(i) : "w" + i, i);
        }
        Map<String, Object> tokenizer = new LinkedHashMap<>();
        tokenizer.put("model", Map.of("type", "WordLevel", "vocab", vocab, "unk_token", "<pad>"));
        tokenizer.put("pre_tokenizer", Map.of("type", "WhitespaceSplit"));
        tokenizer.put("added_tokens", List.of(Map.of("id", 60, "content", "<image>", "special", true)));
        Files.writeString(dir.resolve("tokenizer.json"), Json.write(tokenizer));
        if (template != null) {
            Files.writeString(dir.resolve("tokenizer_config.json"), Json.write(Map.of("chat_template", template)));
        }
        return dir;
    }

    private static ChatResult ask(ChatProvider p, String model, List<ChatMessage> messages, StringBuilder heard)
            throws Exception {
        return p.chat(new ChatRequest(model, "", messages, 4, null, 0f), heard::append);
    }

    @Test
    void aVisionModelIsListedAndAnswersAboutAnAttachedPicture() throws Exception {
        tinyLlava("tiny-llava", null);
        ChatProvider local = ChatService.localModels(List.of(models));
        ModelInfo listed = local.models().stream().filter(m -> m.id().equals("tiny-llava")).findFirst().orElseThrow();
        assertTrue(listed.detail().contains("looks at pictures"), listed.detail());

        Media red = new Media(png(0xcc2222), "image/png");
        ChatResult words = ask(local, "tiny-llava", List.of(new ChatMessage("user", "What is this?")), new StringBuilder());
        ChatResult seen = ask(local, "tiny-llava", List.of(new ChatMessage("user", "What is this?", List.of(red))),
                new StringBuilder());
        // a 32 px picture in 8 px patches is 16 tokens where the one marker stood
        assertTrue(seen.inputTokens() >= words.inputTokens() + 16, words.inputTokens() + " -> " + seen.inputTokens());
        assertTrue(seen.outputTokens() > 0);

        // a picture earlier in the conversation is still in the prompt of a later turn
        ChatResult later = ask(local, "tiny-llava", List.of(new ChatMessage("user", "What is this?", List.of(red)),
                new ChatMessage("assistant", "A box."), new ChatMessage("user", "And that?",
                        List.of(new Media(png(0x2222cc), "image/png")))), new StringBuilder());
        assertTrue(later.outputTokens() > 0);

        IOException unreadable = assertThrows(IOException.class, () -> ask(local, "tiny-llava",
                List.of(new ChatMessage("user", "What is this?", List.of(new Media(new byte[] {1, 2, 3}, "image/png")))),
                new StringBuilder()));
        assertTrue(unreadable.getMessage().contains("cannot be read"), unreadable.getMessage());
    }

    @Test
    void theChatTemplatePlacesEveryPictureOfTheConversation() throws Exception {
        tinyLlava("templated", "{% for m in messages %}{{ m['role'] | upper }}: {% if m['content'] is string %}"
                + "{{ m['content'] }}{% else %}{% for p in m['content'] %}{% if p['type'] == 'image' %}<image> "
                + "{% else %}{{ p['text'] }}{% endif %}{% endfor %}{% endif %} {% endfor %}ASSISTANT:");
        ChatProvider local = ChatService.localModels(List.of(models));
        Media red = new Media(png(0xcc2222), "image/png");
        ChatResult one = ask(local, "templated", List.of(new ChatMessage("user", "What is this?", List.of(red))),
                new StringBuilder());
        ChatResult two = ask(local, "templated", List.of(new ChatMessage("user", "What is this?", List.of(red)),
                new ChatMessage("assistant", "A box."), new ChatMessage("user", "And that?", List.of(red))),
                new StringBuilder());
        assertTrue(two.inputTokens() >= one.inputTokens() + 16, one.inputTokens() + " -> " + two.inputTokens());
    }

    @Test
    void aProjectorFileGoesWithTheModelBesideItAndIsNotAModelItself() throws Exception {
        Path gemma = Files.writeString(models.resolve("gemma-3-4b-it-Q4_K_M.gguf"), "");
        Path qwen = Files.writeString(models.resolve("qwen2.5-0.5b-instruct-q4_k_m.gguf"), "");
        assertNull(LocalProvider.mmproj(gemma), "no projector yet");
        Path forGemma = Files.writeString(models.resolve("mmproj-gemma-3-4b-it-f16.gguf"), "");
        assertEquals(forGemma, LocalProvider.mmproj(gemma));
        // one projector in a folder is taken for the models there; with two, only the one named for the model
        Path forLlava = Files.writeString(models.resolve("mmproj-llava-1.5-7b-f16.gguf"), "");
        assertEquals(forGemma, LocalProvider.mmproj(gemma));
        assertNull(LocalProvider.mmproj(qwen), "neither projector is named for it");
        assertNull(LocalProvider.mmproj(forLlava), "a projector is not a model");
        List<String> listed = ChatService.localModels(List.of(models)).models().stream().map(ModelInfo::id).toList();
        assertEquals(List.of("gemma-3-4b-it-Q4_K_M.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"), listed);
    }
}
