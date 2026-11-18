package io.github.zzuegg.brainiac.llm;

import static io.github.zzuegg.brainiac.llm.LlamaTest.argmax;
import static io.github.zzuegg.brainiac.llm.LlamaTest.expected;
import static io.github.zzuegg.brainiac.llm.LlamaTest.ints;
import static io.github.zzuegg.brainiac.llm.LlamaTest.resource;
import static io.github.zzuegg.brainiac.llm.LlamaTest.rows;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * DeepSeek-V2 and V3 (and Kimi K2, V3 by another name) against tiny random models:
 * transformers' logits and answers, and llama.cpp's for its GGUF conversion (make_deepseek.py).
 * They cover latent attention with and without a query latent, YaRN, the dense first block,
 * shared experts and both group-limited routings.
 */
class DeepSeekTest {

    static final Device DEVICE = LlamaTest.DEVICE;

    @ParameterizedTest
    @ValueSource(strings = {"deepseek-v2", "deepseek-v3"})
    void matchesTransformersLogits(String name) throws IOException {
        Map<String, Object> e = expected(name + "/expected.json");
        Llama model = Llama.load(resource(name), DEVICE, DType.FLOAT32);
        float[] logits = model.forward(new int[][] {ints(e.get("ids"))}).value().toArray();
        assertArrayEquals(rows(e.get("logits")), logits, 1e-4f, name);
    }

    @ParameterizedTest
    @ValueSource(strings = {"deepseek-v2", "deepseek-v3"})
    void readsTheConfiguration(String name) {
        LlamaConfig c = Llama.load(resource(name), DEVICE, DType.FLOAT32).config();
        LlamaConfig.Mla mla = c.extras().mla();
        assertTrue(mla.present());
        assertEquals(name.equals("deepseek-v2") ? 0 : 12, mla.qRank());
        assertEquals(16, mla.kvRank());
        assertEquals(12, c.headDim());
        assertEquals(4, c.rope().dim());
        assertEquals("yarn", c.rope().type());
        LlamaConfig.Moe moe = c.extras().moe();
        assertEquals(8, moe.experts());
        assertEquals(24, moe.sharedFfn());
        assertFalse(moe.in(0));
        assertTrue(moe.in(1));
        assertEquals(name.equals("deepseek-v3"), moe.routing().sigmoid());
        assertEquals(4, moe.routing().groups());
        assertEquals(2, moe.routing().topGroups());
    }

    @ParameterizedTest
    @ValueSource(strings = {"deepseek-v2", "deepseek-v3"})
    void decodingFromTheLatentCacheMatchesTransformers(String name) throws IOException {
        Map<String, Object> e = expected(name + "/expected.json");
        Llama model = Llama.load(resource(name), DEVICE, DType.FLOAT32);
        int[] ids = ints(e.get("ids"));
        int[] greedy = ints(e.get("greedy"));
        LanguageModel.State state = model.newState(1);
        Tensor logits = model.next(state, new int[][] {ids}, null);
        float[] all = rows(e.get("logits"));
        int v = model.vocabSize();
        assertArrayEquals(java.util.Arrays.copyOfRange(all, all.length - v, all.length), logits.toArray(), 1e-4f);
        int[] out = new int[greedy.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = argmax(logits.toArray());
            logits = model.next(state, new int[][] {{out[i]}}, null);
        }
        assertArrayEquals(greedy, out, name);
        assertEquals(ids.length + greedy.length, state.length());
    }

    @ParameterizedTest
    @ValueSource(strings = {"deepseek-v2", "deepseek-v3"})
    void readsAPromptInPiecesAsAtOnce(String name) throws IOException {
        Map<String, Object> e = expected(name + "/expected.json");
        Llama model = Llama.load(resource(name), DEVICE, DType.FLOAT32).prefillChunk(3);
        int[] ids = ints(e.get("ids"));
        LanguageModel.State state = model.newState(1);
        float[] logits = model.nextAll(state, new int[][] {ids}).toArray();
        assertArrayEquals(rows(e.get("logits")), logits, 1e-4f, name);
    }

    @ParameterizedTest
    @ValueSource(strings = {"deepseek-v2", "deepseek-v3"})
    void beamSearchMatchesTransformers(String name) throws IOException {
        Map<String, Object> e = expected(name + "/expected.json");
        Llama model = Llama.load(resource(name), DEVICE, DType.FLOAT32);
        Generator g = new Generator(model, null);
        int[] beam = g.beamSearch(ints(e.get("ids")), 3, 6, 1.0, java.util.Set.of());
        assertArrayEquals(ints(e.get("beam3")), beam, name);
    }

    @ParameterizedTest
    @ValueSource(strings = {"deepseek-v2", "deepseek-v3"})
    void readsLlamaCppConversions(String name) throws IOException {
        // written by llama.cpp's convert_hf_to_gguf.py: kv_b_proj split into k_b and v_b; V3's
        // logits are llama.cpp's own (it keeps the cache in float16, hence the tolerance)
        Map<String, Object> e = expected(name + ".gguf.expected.json");
        Llama model = Llama.load(resource(name + ".gguf"), DEVICE, DType.FLOAT32);
        int[] ids = ints(e.get("ids"));
        float[] logits = model.forward(new int[][] {ids}).value().toArray();
        assertArrayEquals(rows(e.get("logits")), logits, name.equals("deepseek-v3") ? 5e-3f : 1e-4f, name);
        Llama hf = Llama.load(resource(name), DEVICE, DType.FLOAT32);
        assertArrayEquals(hf.forward(new int[][] {ids}).value().toArray(), logits, 1e-4f, name);
        LanguageModel.State state = model.newState(1);
        float[] cached = model.next(state, new int[][] {ids}, null).toArray();
        int v = model.vocabSize();
        assertArrayEquals(java.util.Arrays.copyOfRange(logits, logits.length - v, logits.length), cached, 1e-4f);
    }

    @Test
    void readsBlockScaledFp8Weights() throws IOException {
        // DeepSeek-V3 as released: projections in F8_E4M3 with a scale per 128 x 128 block
        Map<String, Object> e = expected("deepseek-v3-fp8/expected.json");
        Llama model = Llama.load(resource("deepseek-v3-fp8"), DEVICE, DType.FLOAT32);
        float[] logits = model.forward(new int[][] {ints(e.get("ids"))}).value().toArray();
        assertArrayEquals(rows(e.get("logits")), logits, 1e-4f);
    }

    @Test
    void readsKimiK2AsDeepSeekV3(@TempDir Path dir) throws IOException {
        // Kimi K2's config.json says model_type kimi_k2 for DeepSeek-V3's architecture
        Map<String, Object> cfg = Json.object(Json.read(resource("deepseek-v3").resolve("config.json")));
        cfg.put("model_type", "kimi_k2");
        Files.writeString(dir.resolve("config.json"), Json.write(cfg));
        Files.copy(resource("deepseek-v3").resolve("model.safetensors"), dir.resolve("model.safetensors"));
        Map<String, Object> e = expected("deepseek-v3/expected.json");
        Llama model = Llama.load(dir, DEVICE, DType.FLOAT32);
        assertEquals("kimi_k2", model.config().architecture());
        float[] logits = model.forward(new int[][] {ints(e.get("ids"))}).value().toArray();
        assertArrayEquals(rows(e.get("logits")), logits, 1e-4f);
    }

    @ParameterizedTest
    @ValueSource(strings = {"deepseek-v2", "deepseek-v3"})
    void savesAndLoads(String name, @TempDir Path dir) {
        Llama model = Llama.load(resource(name), DEVICE, DType.FLOAT32);
        model.save(dir);
        Llama back = Llama.load(dir, DEVICE, DType.FLOAT32);
        assertEquals(model.config().extras(), back.config().extras());
        int[][] ids = {{1, 2, 3, 4, 5, 6, 7}};
        assertArrayEquals(model.forward(ids).value().toArray(), back.forward(ids).value().toArray(), 1e-5f, name);
    }

    @Test
    void theCacheKeepsOnlyTheLatentAndTheRotaryKey() {
        Llama model = Llama.load(resource("deepseek-v3"), DEVICE, DType.FLOAT32);
        LanguageModel.State state = model.newState(1);
        model.next(state, new int[][] {{1, 5, 9}}, null);
        Path file = null;
        try {
            file = Files.createTempFile("deepseek-state", ".safetensors");
            model.saveState(state, file);
            try (io.github.zzuegg.brainiac.io.SafeTensors st = io.github.zzuegg.brainiac.io.SafeTensors.open(file)) {
                // 16 latent features and 4 rotary ones per step for all 4 heads, not 2 x 4 x 12
                assertEquals(io.github.zzuegg.brainiac.tensor.Shape.of(1, 1, 3, 16), st.shape("keys.0"));
                assertEquals(io.github.zzuegg.brainiac.tensor.Shape.of(1, 1, 3, 4), st.shape("values.0"));
            }
            LanguageModel.State back = model.loadState(file);
            float[] a = model.next(state, new int[][] {{3}}, null).toArray();
            float[] b = model.next(back, new int[][] {{3}}, null).toArray();
            assertArrayEquals(a, b, 1e-5f);
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        } finally {
            if (file != null) {
                file.toFile().delete();
            }
        }
    }

    @Test
    void leftPaddedBatchMatchesSingleSequences() {
        Llama model = Llama.load(resource("deepseek-v2"), DEVICE, DType.FLOAT32);
        int[] a = {5, 9, 3, 7, 2};
        int[] b = {8, 1, 4};
        LanguageModel.State s = model.newState(2);
        float[] both = model.next(s, new int[][] {a, {0, 0, 8, 1, 4}},
                new boolean[][] {new boolean[5], {true, true, false, false, false}}).toArray();
        float[] both2 = model.next(s, new int[][] {{6}, {6}}, null).toArray();
        int v = model.vocabSize();
        for (int[] seq : new int[][] {a, b}) {
            LanguageModel.State one = model.newState(1);
            float[] first = model.next(one, new int[][] {seq}, null).toArray();
            float[] second = model.next(one, new int[][] {{6}}, null).toArray();
            int row = seq == a ? 0 : 1;
            assertArrayEquals(first, java.util.Arrays.copyOfRange(both, row * v, row * v + v), 1e-4f);
            assertArrayEquals(second, java.util.Arrays.copyOfRange(both2, row * v, row * v + v), 1e-4f);
        }
    }

    @Test
    void splitAcrossDevicesGivesTheSameLogits() {
        int[][] ids = {{1, 5, 9, 3, 7}};
        Llama whole = Llama.load(resource("deepseek-v3"), DEVICE, DType.FLOAT32);
        Llama split = Llama.load(resource("deepseek-v3"), Device.cpu(), DType.FLOAT32).offload(DEVICE, DType.FLOAT32, 2);
        assertArrayEquals(whole.forward(ids).value().toArray(), split.forward(ids).value().toArray(), 1e-4f);
        LanguageModel.State s = split.newState(1);
        LanguageModel.State w = whole.newState(1);
        assertArrayEquals(whole.next(w, ids, null).toArray(), split.next(s, ids, null).toArray(), 1e-4f);
    }

    @Test
    void trainsThroughTheLatentAttentionAndTheExperts() {
        Llama model = Llama.load(resource("deepseek-v3"), DEVICE, DType.FLOAT32);
        int[][] ids = {{1, 5, 9, 3, 7, 2, 11, 4}};
        Var loss = model.loss(ids);
        loss.backward();
        for (String n : new String[] {"model.layers.1.self_attn.kv_b_proj.weight", "model.layers.1.self_attn.q_a_proj.weight",
                "model.layers.1.mlp.gate.weight", "model.layers.1.mlp.shared_experts.up_proj.weight"}) {
            assertNotNull(model.namedParameters().get(n).grad(), n);
        }
        assertFalse(model.parameters().contains(model.namedParameters().get(
                "model.layers.1.mlp.gate.e_score_correction_bias")));
    }

    @Test
    void randomModelsHaveTheRightShapes() {
        LlamaConfig c = Llama.load(resource("deepseek-v3"), DEVICE, DType.FLOAT32).config();
        Llama model = Llama.random(c, 1, DEVICE, DType.FLOAT32);
        float[] logits = model.forward(new int[][] {{1, 2, 3}}).value().toArray();
        assertEquals(3 * c.vocabSize(), logits.length);
    }

    @Test
    void routingPicksWithinTheBestGroups() {
        // 8 experts in 4 groups of 2, 2 groups per token, 3 experts per token
        float[] scores = {0.9f, 0.1f, 0.5f, 0.5f, 0.8f, 0.0f, 0.2f, 0.3f};
        // V2: groups ranked by their best expert: groups 0 (0.9) and 2 (0.8)
        int[] v2 = Llama.choose(scores, null, 1, 8, 3, new LlamaConfig.Routing(false, 4, 2, 1, 0, false));
        assertArrayEquals(new int[] {0, 4, 1}, v2);
        // V3: ranked by the sum of the two best: group 1 (1.0) and 0 (1.0) beat 2 (0.8)
        int[] v3 = Llama.choose(scores, null, 1, 8, 3, new LlamaConfig.Routing(true, 4, 2, 1, 0, false));
        assertArrayEquals(new int[] {0, 2, 3}, v3);
        // a selection bias moves the choice, but not the weights (those stay the scores)
        float[] bias = {0, 0, 0, 0, 0, 0, 1, 1};
        int[] biased = Llama.choose(scores, bias, 1, 8, 3, new LlamaConfig.Routing(true, 4, 2, 1, 0, false));
        assertArrayEquals(new int[] {7, 6, 0}, biased);
    }
}
