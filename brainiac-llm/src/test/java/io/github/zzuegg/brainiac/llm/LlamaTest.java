package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Compares tiny random models with transformers (make_models.py wrote the weights and the
 * logits transformers computed from them).
 */
class LlamaTest {

    static final Device DEVICE = TestDevice.get();

    static Path resource(String name) {
        try {
            return Path.of(LlamaTest.class.getResource(name).toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
    }

    static Map<String, Object> expected(String name) throws IOException {
        return Json.object(Json.read(resource(name)));
    }

    static int[] ints(Object list) {
        return Json.array(list).stream().mapToInt(o -> ((Number) o).intValue()).toArray();
    }

    static float[] rows(Object list) {
        List<Object> rows = Json.array(list);
        int n = Json.array(rows.get(0)).size();
        float[] out = new float[rows.size() * n];
        for (int i = 0; i < rows.size(); i++) {
            List<Object> r = Json.array(rows.get(i));
            for (int j = 0; j < n; j++) {
                out[i * n + j] = ((Number) r.get(j)).floatValue();
            }
        }
        return out;
    }

    static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }

    @ParameterizedTest
    @ValueSource(strings = {"llama", "llama3-rope", "qwen2"})
    void matchesTransformersLogits(String name) throws IOException {
        Map<String, Object> e = expected(name + "/expected.json");
        Llama model = Llama.load(resource(name), DEVICE, DType.FLOAT32);
        int[] ids = ints(e.get("ids"));
        float[] logits = model.forward(new int[][] {ids}).value().toArray();
        assertArrayEquals(rows(e.get("logits")), logits, 1e-4f, name);
    }

    @ParameterizedTest
    @ValueSource(strings = {"llama", "llama3-rope", "qwen2"})
    void greedyDecodingWithTheCacheMatchesTransformers(String name) throws IOException {
        Map<String, Object> e = expected(name + "/expected.json");
        Llama model = Llama.load(resource(name), DEVICE, DType.FLOAT32);
        int[] ids = ints(e.get("ids"));
        int[] greedy = ints(e.get("greedy"));
        LanguageModel.State state = model.newState(1);
        Tensor logits = model.next(state, new int[][] {ids}, null);
        int[] out = new int[greedy.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = argmax(logits.toArray());
            logits = model.next(state, new int[][] {{out[i]}}, null);
        }
        assertArrayEquals(greedy, out, name);
        assertEquals(ids.length + greedy.length, state.length());
    }

    @Test
    void readsQuantizedGguf() throws IOException {
        Map<String, Object> e = expected("llama.gguf.expected.json");
        Llama model = Llama.load(resource("llama.gguf"), DEVICE, DType.FLOAT32);
        float[] logits = model.forward(new int[][] {ints(e.get("ids"))}).value().toArray();
        // the reference used the dequantized weights; activations are rounded differently
        assertArrayEquals(rows(e.get("logits")), logits, 2e-3f);
    }

    @Test
    void leftPaddedBatchMatchesSingleSequences() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 3, DEVICE, DType.FLOAT32);
        int[] a = {5, 9, 3, 7, 2};
        int[] b = {8, 1, 4};
        int[][] batch = {a, {0, 0, 8, 1, 4}};
        boolean[][] pad = {new boolean[5], {true, true, false, false, false}};
        LanguageModel.State s = model.newState(2);
        float[] both = model.next(s, batch, pad).toArray();
        float[] both2 = model.next(s, new int[][] {{6}, {6}}, null).toArray();
        int v = model.vocabSize();
        for (int[] seq : List.of(a, b)) {
            LanguageModel.State one = model.newState(1);
            float[] first = model.next(one, new int[][] {seq}, null).toArray();
            float[] second = model.next(one, new int[][] {{6}}, null).toArray();
            int row = seq == a ? 0 : 1;
            assertArrayEquals(first, java.util.Arrays.copyOfRange(both, row * v, row * v + v), 1e-4f);
            assertArrayEquals(second, java.util.Arrays.copyOfRange(both2, row * v, row * v + v), 1e-4f);
        }
    }

    @Test
    void cacheThatGrowsMatchesTheFullForwardPass() {
        // 150 steps: the cache's buffers grow from 64 to 128 to 256 on the way
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 5, DEVICE, DType.FLOAT32);
        int[] seq = new int[150];
        java.util.SplittableRandom r = new java.util.SplittableRandom(6);
        for (int i = 0; i < seq.length; i++) {
            seq[i] = r.nextInt(40);
        }
        LanguageModel.State s = model.newState(1);
        model.next(s, new int[][] {java.util.Arrays.copyOf(seq, 10)}, null);
        float[] last = null;
        for (int i = 10; i < seq.length; i++) {
            last = model.next(s, new int[][] {{seq[i]}}, null).toArray();
        }
        float[] full = model.forward(new int[][] {seq}).value().toArray();
        int v = model.vocabSize();
        assertArrayEquals(java.util.Arrays.copyOfRange(full, full.length - v, full.length), last, 1e-4f);
        assertEquals(150, s.length());
    }

    @Test
    void selectReordersRows() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 4, DEVICE, DType.FLOAT32);
        LanguageModel.State s = model.newState(2);
        model.next(s, new int[][] {{1, 2, 3}, {4, 5, 6}}, null);
        s = s.select(new int[] {1, 1});
        float[] out = model.next(s, new int[][] {{7}, {7}}, null).toArray();
        LanguageModel.State one = model.newState(1);
        model.next(one, new int[][] {{4, 5, 6}}, null);
        float[] ref = model.next(one, new int[][] {{7}}, null).toArray();
        int v = model.vocabSize();
        assertArrayEquals(ref, java.util.Arrays.copyOfRange(out, 0, v), 1e-4f);
        assertArrayEquals(ref, java.util.Arrays.copyOfRange(out, v, 2 * v), 1e-4f);
    }

    @Test
    void savesAndLoads(@TempDir Path dir) {
        Llama model = Llama.load(resource("qwen2"), DEVICE, DType.FLOAT32);
        model.save(dir);
        Llama back = Llama.load(dir, DEVICE, DType.FLOAT32);
        int[][] ids = {{1, 2, 3, 4}};
        assertEquals(model.config(), back.config());
        assertArrayEquals(model.forward(ids).value().toArray(), back.forward(ids).value().toArray());
        assertTrue(back.parameters().size() > 0);
    }
}
