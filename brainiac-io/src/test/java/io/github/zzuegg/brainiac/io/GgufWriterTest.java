package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class GgufWriterTest {

    @TempDir
    Path dir;

    private Path fixture(String name) throws IOException {
        Path p = dir.resolve(name);
        try (InputStream in = getClass().getResourceAsStream("tokenizers/" + name)) {
            Files.copy(in, p);
        }
        return p;
    }

    /** A SentencePiece-style BPE tokenizer.json (as TinyLlama's) with its tokenizer_config.json. */
    private Path folder(String name) throws IOException {
        Path d = Files.createDirectories(dir.resolve(name));
        for (String file : new String[] {"tokenizer.json", "tokenizer_config.json"}) {
            try (InputStream in = getClass().getResourceAsStream("tokenizers/" + name + "/" + file)) {
                Files.copy(in, d.resolve(file));
            }
        }
        return d;
    }

    @Test
    void metadataAndTensorsReadBack() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("general.architecture", "llama");
        m.put("llama.block_count", 2L);
        m.put("llama.rope.freq_base", 10000.0f);
        m.put("flag", true);
        m.put("signed", -3);
        m.put("names", List.of("a", "bé", "c"));
        m.put("scores", List.of(0.5f, -1f));
        m.put("ids", new GgufWriter.Typed(GgufWriter.INT32, List.of(1, 2, 3)));
        Map<String, Tensor> t = new LinkedHashMap<>();
        t.put("vec", Tensor.of(Shape.of(3), 1, 2, 3));
        t.put("mat", Tensor.of(Shape.of(2, 3), 1, 2, 3, 4, 5, 6.5f));
        Path f = dir.resolve("x.gguf");
        GgufWriter.write(f, m, t, true);
        try (Gguf g = Gguf.open(f)) {
            assertEquals("llama", g.architecture());
            assertEquals(2L, g.metadata().get("llama.block_count"));
            assertEquals(10000.0, (Double) g.metadata().get("llama.rope.freq_base"), 0);
            assertEquals(true, g.metadata().get("flag"));
            assertEquals(-3L, g.metadata().get("signed"));
            assertEquals(List.of("a", "bé", "c"), g.metadata().get("names"));
            assertEquals(List.of(1L, 2L, 3L), g.metadata().get("ids"));
            assertEquals("F32", g.format("vec"));
            assertEquals("F16", g.format("mat"));
            assertArrayEquals(new int[] {2, 3}, g.shape("mat").toArray());
            assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6.5f},
                    g.tensor("mat", io.github.zzuegg.brainiac.tensor.Device.cpu(),
                            io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray(), 0);
        }
    }

    @ParameterizedTest
    // not gpt2-bpe.json: GGUF has no add_prefix_space; and texts without leading or doubled
    // spaces, which GGUF SentencePiece models keep (sp-bpe.model removes them)
    @ValueSource(strings = {"bytelevel-bpe.json", "sp-bpe.model", "vocab.gguf", "vocab-llama.gguf", "metaspace-bpe.json",
        "llama-legacy-true"})
    void tokenizersSurviveAGgufRoundTrip(String name) throws IOException {
        PretrainedTokenizer tok = PretrainedTokenizer.load(name.contains(".") ? fixture(name) : folder(name));
        Path f = dir.resolve("tok.gguf");
        GgufWriter.write(f, tok.ggufMetadata(), Map.of(), false);
        PretrainedTokenizer back;
        try (Gguf g = Gguf.open(f)) {
            back = PretrainedTokenizer.fromGguf(g);
        }
        assertEquals(tok.size(), back.size());
        for (String text : new String[] {"Hello world", "The quick brown fox, 1234!", "naïve café spaces\n\nnew",
            "<|im_start|>user hi", "Why? Because 🙂 works."}) {
            assertArrayEquals(tok.encode(text), back.encode(text), name + ": " + text);
            assertArrayEquals(tok.encode(text, false), back.encode(text, false), name + ": " + text);
            assertEquals(tok.decode(tok.encode(text)), back.decode(back.encode(text)), name + ": " + text);
        }
    }
}
