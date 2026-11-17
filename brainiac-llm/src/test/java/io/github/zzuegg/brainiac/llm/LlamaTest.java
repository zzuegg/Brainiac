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
    void chunkedLossEqualsCrossEntropyOfTheLogits() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 7, DEVICE, DType.FLOAT32);
        int[][] tokens = {{1, 5, 9, 3, 7, 2, 30}, {4, 4, 8, 0, 39, 12, 6}};
        float[] next = new float[14];
        for (int b = 0; b < 2; b++) {
            for (int t = 0; t < 7; t++) {
                next[b * 7 + t] = t < 6 ? tokens[b][t + 1] : -1;
            }
        }
        var logits = model.forward(tokens).reshape(14, 40);
        var expected = io.github.zzuegg.brainiac.nn.Functional.crossEntropy(logits,
                io.github.zzuegg.brainiac.tensor.Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(14), next, DEVICE));
        expected.backward();
        java.util.List<float[]> grads = model.parameters().stream().map(p -> p.grad().toArray()).toList();
        model.parameters().forEach(p -> p.zeroGrad());
        var loss = model.loss(tokens, 3);
        loss.backward();
        assertEquals(expected.value().item(), loss.value().item(), 1e-5f);
        for (int i = 0; i < grads.size(); i++) {
            assertArrayEquals(grads.get(i), model.parameters().get(i).grad().toArray(), 1e-5f, "parameter " + i);
        }
    }

    /** Plain gradient steps on the adapters; returns the losses before and after. */
    private static float[] trainAdapters(Llama model, java.util.List<io.github.zzuegg.brainiac.autograd.Var> params,
            int[][] tokens, int steps) {
        float first = 0;
        float last = 0;
        for (int i = 0; i < steps; i++) {
            var loss = model.loss(tokens);
            loss.backward();
            last = loss.value().item();
            if (i == 0) {
                first = last;
            }
            for (var p : params) {
                p.assign(p.value().sub(p.grad().mul(0.5f)));
                p.zeroGrad();
            }
        }
        return new float[] {first, last};
    }

    @Test
    void loraStartsUnchangedTrainsOnlyTheAdaptersAndMerges() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 8, DEVICE, DType.FLOAT32);
        int[][] tokens = {{1, 5, 9, 3, 7, 2, 30, 4}, {4, 4, 8, 0, 39, 12, 6, 6}};
        float[] before = model.forward(tokens).value().toArray();
        Map<String, float[]> weights = new java.util.HashMap<>();
        model.namedParameters().forEach((n, v) -> weights.put(n, v.value().toArray()));
        var adapters = model.addLora(4, 8, Llama.attentionAndMlp(), 9);
        assertEquals(2 * 2 * 7, adapters.size());
        assertEquals(adapters.size(), model.loraParameters().size());
        assertArrayEquals(before, model.forward(tokens).value().toArray(), 1e-6f);
        float[] losses = trainAdapters(model, adapters, tokens, 20);
        assertTrue(losses[1] < losses[0] - 0.1f, java.util.Arrays.toString(losses));
        model.namedParameters().forEach((n, v) -> {
            assertArrayEquals(weights.get(n), v.value().toArray(), 0f, n + " is frozen");
            assertTrue(v.grad() == null || v.grad().toArray().length == 0 || java.util.Arrays.equals(v.grad().toArray(),
                    new float[v.grad().toArray().length]), n + " has no gradient");
        });
        float[] adapted = model.forward(tokens).value().toArray();
        model.mergeLora();
        assertTrue(model.loraParameters().isEmpty());
        assertArrayEquals(adapted, model.forward(tokens).value().toArray(), 1e-4f);
    }

    @Test
    void qloraTrainsAdaptersOnQuantizedWeights() {
        Llama model = Llama.random(LlamaConfig.small(40, 64, 2, 4, 2), 10, DEVICE, DType.FLOAT32);
        int[][] tokens = {{1, 5, 9, 3, 7, 2, 30, 4}, {4, 4, 8, 0, 39, 12, 6, 6}};
        float[] full = model.forward(tokens).value().toArray();
        model.quantizeWeights(8, 32);
        float[] quantized = model.forward(tokens).value().toArray();
        assertArrayEquals(full, quantized, 2e-2f);
        var adapters = model.addLora(4, 8, Llama.attentionAndMlp(), 11);
        float[] losses = trainAdapters(model, adapters, tokens, 20);
        assertTrue(losses[1] < losses[0] - 0.1f, java.util.Arrays.toString(losses));
        // decoding with the adapters matches the full pass
        LanguageModel.State s = model.newState(2);
        float[] last = model.next(s, tokens, null).toArray();
        float[] all = model.forward(tokens).value().toArray();
        int v = model.vocabSize();
        for (int b = 0; b < 2; b++) {
            int row = (b * 8 + 7) * v;
            assertArrayEquals(java.util.Arrays.copyOfRange(all, row, row + v),
                    java.util.Arrays.copyOfRange(last, b * v, b * v + v), 1e-4f);
        }
    }

    @Test
    void aWindowedStateForgetsTheMiddleAndStaysSmall() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 12, DEVICE, DType.FLOAT32);
        LanguageModel.State full = model.newState(1);
        LanguageModel.State windowed = model.newState(1, 4, 16);
        java.util.SplittableRandom r = new java.util.SplittableRandom(13);
        int[] first = new int[10];
        for (int i = 0; i < first.length; i++) {
            first[i] = r.nextInt(40);
        }
        float[] a = model.next(full, new int[][] {first}, null).toArray();
        float[] b = model.next(windowed, new int[][] {first}, null).toArray();
        int maxLength = 0;
        boolean differed = false;
        for (int step = 0; step < 100; step++) {
            int t = r.nextInt(40);
            if (step < 10) {
                // nothing forgotten yet: the same as keeping everything
                assertArrayEquals(a, b, 1e-5f, "step " + step);
            } else {
                differed |= !java.util.Arrays.equals(a, b);
            }
            a = model.next(full, new int[][] {{t}}, null).toArray();
            b = model.next(windowed, new int[][] {{t}}, null).toArray();
            maxLength = Math.max(maxLength, windowed.length());
            for (float f : b) {
                assertTrue(Float.isFinite(f));
            }
        }
        assertTrue(differed);
        assertTrue(maxLength <= 4 + 16 + 4, "length " + maxLength);
        assertEquals(110, full.length());
    }

    @Test
    void anEightBitCacheStaysCloseToTheFullOne() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 14, DEVICE, DType.FLOAT32);
        LanguageModel.State full = model.newState(2);
        LanguageModel.State small = model.quantizeCache(true).newState(2);
        model.quantizeCache(false);
        java.util.SplittableRandom r = new java.util.SplittableRandom(15);
        int[][] prompt = new int[2][12];
        for (int[] row : prompt) {
            for (int i = 0; i < row.length; i++) {
                row[i] = r.nextInt(40);
            }
        }
        boolean[][] padding = new boolean[2][12];
        padding[1][0] = padding[1][1] = true;
        float[] a = model.next(full, prompt, padding).toArray();
        float[] b = model.next(small, prompt, padding).toArray();
        org.junit.jupiter.api.Assertions.assertFalse(java.util.Arrays.equals(a, b));   // the levels are read, not the values
        for (int step = 0; step < 80; step++) {
            double largest = 0;
            double error = 0;
            for (int i = 0; i < a.length; i++) {
                largest = Math.max(largest, Math.abs(a[i]));
                error = Math.max(error, Math.abs(a[i] - b[i]));
            }
            // levels of 1/127 of each row's largest value: errors well under a percent
            assertTrue(error < 0.01 * largest, "step " + step + ": " + error + " of " + largest);
            if (step == 40) {
                small = small.select(new int[] {1, 0});
                full = full.select(new int[] {1, 0});
            }
            int[][] next = {{r.nextInt(40)}, {r.nextInt(40)}};
            a = model.next(full, next, null).toArray();
            b = model.next(small, next, null).toArray();
        }
        assertEquals(full.length(), small.length());
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
