package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

/**
 * Compares with the libraries the fixtures were made with (make_tokenizers.py): Hugging
 * Face tokenizers for the {@code .json} files, sentencepiece for the {@code .model} files.
 */
class PretrainedTokenizerTest {

    @TempDir
    Path dir;

    private Path fixture(String name) throws IOException {
        Path p = dir.resolve(name);
        if (!Files.exists(p)) {
            try (InputStream in = getClass().getResourceAsStream("tokenizers/" + name)) {
                Files.copy(in, p);
            }
        }
        return p;
    }

    private List<Map<String, Object>> cases(String name) throws IOException {
        return Json.array(Json.read(fixture(name + ".expected.json"))).stream().map(Json::object).toList();
    }

    private static int[] ints(Object list) {
        return Json.array(list).stream().mapToInt(o -> ((Number) o).intValue()).toArray();
    }

    @ParameterizedTest
    @ValueSource(strings = {"bytelevel-bpe", "gpt2-bpe", "metaspace-bpe", "unigram", "wordpiece"})
    void matchesHuggingFaceTokenizers(String name) throws IOException {
        PretrainedTokenizer tok = PretrainedTokenizer.load(fixture(name + ".json"));
        for (Map<String, Object> c : cases(name)) {
            String text = (String) c.get("text");
            int[] expected = ints(c.get("ids"));
            int[] ids = tok.encode(text, true);
            assertArrayEquals(expected, ids, name + ": \"" + text + "\"");
            assertEquals(c.get("decoded"), tok.decode(ids), name + " decode: \"" + text + "\"");
            assertEquals(c.get("decodedSkip"), tok.decode(ids, true), name + " decode, skipping specials");
        }
        // saved and loaded again
        Path saved = dir.resolve(name + "-saved.json");
        tok.save(saved);
        PretrainedTokenizer again = PretrainedTokenizer.load(saved);
        assertArrayEquals(tok.encode("Grüße aus München! 🙂"), again.encode("Grüße aus München! 🙂"));
    }

    /** transformers rebuilds a Llama tokenizer's pipeline: with "legacy": false no "▁" after special tokens. */
    @ParameterizedTest
    @ValueSource(strings = {"llama-legacy-false", "llama-legacy-true"})
    void buildsLlamaTokenizersAsTransformersDoes(String name) throws IOException {
        Path model = dir.resolve(name);
        Files.createDirectories(model);
        for (String f : List.of("tokenizer.json", "tokenizer_config.json", "expected.json")) {
            try (InputStream in = getClass().getResourceAsStream("tokenizers/" + name + "/" + f)) {
                Files.copy(in, model.resolve(f));
            }
        }
        PretrainedTokenizer tok = PretrainedTokenizer.load(model);
        for (Object o : Json.array(Json.read(model.resolve("expected.json")))) {
            Map<String, Object> c = Json.object(o);
            String text = (String) c.get("text");
            assertArrayEquals(ints(c.get("ids")), tok.encode(text, true), name + ": \"" + text + "\"");
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"sp-bpe", "sp-unigram"})
    void matchesSentencePiece(String name) throws IOException {
        PretrainedTokenizer tok = PretrainedTokenizer.load(fixture(name + ".model"));
        for (Map<String, Object> c : cases(name)) {
            String text = (String) c.get("text");
            int[] ids = tok.encode(text);
            assertArrayEquals(ints(c.get("ids")), ids, name + ": \"" + text + "\"");
            assertEquals(c.get("decoded"), tok.decode(ids), name + " decode: \"" + text + "\"");
        }
        assertEquals(1, tok.bos().orElseThrow());
        assertEquals(2, tok.eos().orElseThrow());
        assertEquals(1, tok.encode("Hello", true)[0]);
        Path saved = dir.resolve(name + ".json");
        tok.save(saved);
        assertArrayEquals(tok.encode("Hello  world"), PretrainedTokenizer.load(saved).encode("Hello  world"));
    }

    @Test
    void readsByteLevelVocabularyFromGguf() throws IOException {
        PretrainedTokenizer tok = PretrainedTokenizer.load(fixture("vocab.gguf"));
        for (Map<String, Object> c : cases("bytelevel-bpe")) {
            String text = (String) c.get("text");
            // the HF fixture also normalizes to NFC; these texts are already NFC
            assertArrayEquals(ints(c.get("ids")), tok.encode(text, true), "gguf: \"" + text + "\"");
            assertEquals(c.get("decodedSkip"), tok.decode(tok.encode(text, true), true));
        }
        assertEquals(0, tok.bos().orElseThrow());
        assertTrue(tok.chatTemplate().contains("messages"));
        assertEquals(1, tok.id("<|im_start|>").orElseThrow());
    }

    @Test
    void readsSentencePieceVocabularyFromGguf() throws IOException {
        PretrainedTokenizer tok = PretrainedTokenizer.load(fixture("vocab-llama.gguf"));
        for (Map<String, Object> c : cases("sp-bpe")) {
            String text = (String) c.get("text");
            if (text.contains("  ") || text.startsWith(" ") || text.endsWith(" ") || text.contains("\n ")) {
                continue;   // llama.cpp keeps runs of spaces that sentencepiece collapses
            }
            int[] ids = tok.encode(text);
            assertArrayEquals(ints(c.get("ids")), ids, "gguf spm: \"" + text + "\"");
            assertEquals(c.get("decoded"), tok.decode(ids));
        }
        assertEquals(tok.bos().orElseThrow(), tok.encode("Hi", true)[0]);
    }

    @Test
    void addsTokensDeclaredOnlyInTheTokenizerConfig() throws IOException {
        // Qwen2-VL-2B: <|image_pad|> is in tokenizer_config.json's added_tokens_decoder only
        String json = Files.readString(fixture("bytelevel-bpe.json"));
        int id = PretrainedTokenizer.fromJson(json, Map.of()).size();
        PretrainedTokenizer plain = PretrainedTokenizer.fromJson(json, Map.of());
        assertTrue(plain.encode("a<|image_pad|>b", false).length > 3);
        PretrainedTokenizer tok = PretrainedTokenizer.fromJson(json, Map.of("added_tokens_decoder",
                Map.of(String.valueOf(id), Map.of("content", "<|image_pad|>", "special", true))));
        int[] ids = tok.encode("a<|image_pad|>b", false);
        assertEquals(3, ids.length);
        assertEquals(id, ids[1]);
        assertEquals(plain.encode("a", false)[0], ids[0]);
    }
}
