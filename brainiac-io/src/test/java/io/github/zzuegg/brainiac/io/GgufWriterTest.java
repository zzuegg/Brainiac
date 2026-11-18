package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    private void shard(String name, int no, int count, int total, Map<String, Tensor> t) {
        Map<String, Object> m = new LinkedHashMap<>();
        if (no == 0) {
            m.put("general.architecture", "llama");
        }
        m.put("split.no", new GgufWriter.Typed(GgufWriter.UINT16, no));
        m.put("split.count", new GgufWriter.Typed(GgufWriter.UINT16, count));
        m.put("split.tensors.count", new GgufWriter.Typed(GgufWriter.INT32, total));
        GgufWriter.write(dir.resolve(String.format("%s-%05d-of-%05d.gguf", name, no + 1, count)), m, t, false);
    }

    @Test
    void quantizedTensorsReadBackWithinTheirPrecision() throws IOException {
        java.util.SplittableRandom r = new java.util.SplittableRandom(4);
        float[] v = new float[4 * 512];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * (1 + (i / 64) % 3)) + (i % 97 == 0 ? 6 : 0);
        }
        Tensor w = Tensor.of(Shape.of(4, 512), v);
        Tensor odd = Tensor.of(Shape.of(2, 40), java.util.Arrays.copyOf(v, 80));   // rows not whole blocks
        double rms = 0;
        for (float f : v) {
            rms += f * f;
        }
        rms = Math.sqrt(rms / v.length);
        // relative RMS error each format should stay under
        Map<GgufWriter.Type, Double> bound = Map.of(GgufWriter.Type.F16, 1e-3, GgufWriter.Type.Q8_0, 0.01,
                GgufWriter.Type.Q6_K, 0.03, GgufWriter.Type.Q4_K, 0.12, GgufWriter.Type.Q4_0, 0.15);
        var cpu = io.github.zzuegg.brainiac.tensor.Device.cpu();
        var f32 = io.github.zzuegg.brainiac.tensor.DType.FLOAT32;
        for (Map.Entry<GgufWriter.Type, Double> e : bound.entrySet()) {
            Path f = dir.resolve(e.getKey() + ".gguf");
            Map<String, Tensor> t = new LinkedHashMap<>();
            t.put("w", w);
            t.put("odd", odd);
            t.put("norm", Tensor.of(Shape.of(3), 1, 2, 3));
            GgufWriter.write(f, Map.of("general.architecture", "llama"), t, name -> e.getKey());
            try (Gguf g = Gguf.open(f)) {
                float[] back = g.tensor("w", cpu, f32).toArray();
                double err = 0;
                for (int i = 0; i < v.length; i++) {
                    err += (back[i] - v[i]) * (back[i] - v[i]);
                }
                err = Math.sqrt(err / v.length) / rms;
                assertTrue(err < e.getValue(), e.getKey() + ": relative error " + err);
                assertArrayEquals(odd.toArray(), g.tensor("odd", cpu, f32).toArray(), 0.01f);
                assertArrayEquals(new float[] {1, 2, 3}, g.tensor("norm", cpu, f32).toArray(), 0f);
            }
        }
        float[] zeros = new float[256];
        assertArrayEquals(zeros, decodeVia(GgufWriter.Type.Q4_K, zeros), 0f);
        assertArrayEquals(zeros, decodeVia(GgufWriter.Type.Q6_K, zeros), 0f);
    }

    private float[] decodeVia(GgufWriter.Type type, float[] values) throws IOException {
        Path f = dir.resolve("z" + type + ".gguf");
        GgufWriter.write(f, Map.of(), Map.of("z", Tensor.of(Shape.of(1, values.length), values)), name -> type);
        try (Gguf g = Gguf.open(f)) {
            return g.tensor("z", io.github.zzuegg.brainiac.tensor.Device.cpu(),
                    io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray();
        }
    }

    @Test
    void splitModelsOpenWholeFromAnyShard() throws IOException {
        shard("m", 0, 3, 3, Map.of("a", Tensor.of(Shape.of(2), 1, 2)));
        shard("m", 1, 3, 3, Map.of("b", Tensor.of(Shape.of(2, 2), 3, 4, 5, 6)));
        shard("m", 2, 3, 3, Map.of("c", Tensor.of(Shape.of(1), 7)));
        for (int i = 1; i <= 3; i++) {
            try (Gguf g = Gguf.open(dir.resolve("m-0000" + i + "-of-00003.gguf"))) {
                assertEquals(3, g.files());
                assertEquals(dir.resolve("m-00001-of-00003.gguf"), g.path());
                assertEquals("llama", g.architecture());
                assertEquals(List.of("a", "b", "c"), g.names());
                var cpu = io.github.zzuegg.brainiac.tensor.Device.cpu();
                var f32 = io.github.zzuegg.brainiac.tensor.DType.FLOAT32;
                assertArrayEquals(new float[] {1, 2}, g.tensor("a", cpu, f32).toArray(), 0);
                assertArrayEquals(new float[] {3, 4, 5, 6}, g.tensor("b", cpu, f32).toArray(), 0);
                assertArrayEquals(new float[] {7}, g.tensor("c", cpu, f32).toArray(), 0);
            }
        }
        Files.delete(dir.resolve("m-00002-of-00003.gguf"));
        IllegalArgumentException e = org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> Gguf.open(dir.resolve("m-00001-of-00003.gguf")));
        org.junit.jupiter.api.Assertions.assertTrue(e.getMessage().contains("m-00002-of-00003.gguf is missing"),
                e.getMessage());
        shard("n", 0, 2, 2, Map.of("a", Tensor.of(Shape.of(1), 1)));
        shard("n", 1, 2, 2, Map.of("a", Tensor.of(Shape.of(1), 2)));
        e = org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> Gguf.open(dir.resolve("n-00001-of-00002.gguf")));
        org.junit.jupiter.api.Assertions.assertTrue(e.getMessage().contains("in two shards"), e.getMessage());
    }

    @ParameterizedTest
    // not gpt2-bpe.json: GGUF has no add_prefix_space; and texts without leading or doubled
    // spaces, which GGUF SentencePiece models keep (sp-bpe.model removes them)
    @ValueSource(strings = {"bytelevel-bpe.json", "sp-bpe.model", "vocab.gguf", "vocab-llama.gguf", "metaspace-bpe.json",
        "llama-legacy-true", "llama-legacy-false"})
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

    @Test
    void ggufSentencePieceTokenizersPrefixOnlyTheStart() throws IOException {
        // as transformers' non-legacy Llama tokenizers (Mistral's chat template, say): no "▁"
        // before a leading <s> or after a control token, none added to a leading space
        PretrainedTokenizer tok = PretrainedTokenizer.load(folder("llama-legacy-false"));
        Path f = dir.resolve("tok.gguf");
        GgufWriter.write(f, tok.ggufMetadata(), Map.of(), false);
        PretrainedTokenizer back;
        try (Gguf g = Gguf.open(f)) {
            back = PretrainedTokenizer.fromGguf(g);
        }
        for (String text : new String[] {"Hello world", " Hello", "<s>[INST] Hi[/INST]",
            "<s>[INST] Hi [/INST] Ok</s>[INST] Again[/INST]"}) {
            assertArrayEquals(tok.encode(text), back.encode(text), text);
        }
        int[] withBos = back.encode("<s>[INST] Hi[/INST]", true);
        assertEquals(1, withBos[0]);
        assertEquals(314, withBos[1], "no second BOS");
        // transformers' ids for the same tokenizer
        assertArrayEquals(new int[] {355, 295, 321, 466, 331}, back.encode(" Hello"));
        assertArrayEquals(new int[] {1, 314, 627, 306, 307, 315, 355, 295, 325, 314, 272, 627, 306, 307, 315},
                back.encode("<s>[INST] Hi[/INST]"));
        assertArrayEquals(new int[] {1, 314, 627, 306, 307, 315, 355, 295, 325, 355, 314, 272, 627, 306, 307, 315, 355,
            302, 327, 2, 314, 627, 306, 307, 315, 355, 288, 323, 317, 378, 314, 272, 627, 306, 307, 315},
                back.encode("<s>[INST] Hi [/INST] Ok</s>[INST] Again[/INST]"));
        // and decodes as transformers does, writing control tokens unless asked to skip them
        int[] chat = back.encode("<s>[INST] Hi [/INST] Ok</s>[INST] Again[/INST]");
        assertEquals("<s>[INST] Hi [/INST] Ok</s>[INST] Again[/INST]", back.decode(chat));
        assertEquals("[INST] Hi [/INST] Ok[INST] Again[/INST]", back.decode(chat, true));
    }
}
