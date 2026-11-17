package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class LongContextTest {

    private static final Device DEVICE = TestDevice.get();

    @Test
    void yarnMatchesTransformers() {
        // reference values from transformers' _compute_yarn_parameters (dim 64, factor 4, 2048)
        Rope r = new Rope(new Rope.Settings(64, 10000, "yarn", 4, 1, 4, 2048, null));
        double[] f = r.frequencies();
        double[][] expected = {{0, 1.0}, {5, 0.23713737056616555}, {10, 0.049745578766838564},
            {15, 0.00794983930712751}, {20, 0.0009730085108210396}, {31, 3.33380358040831e-05}};
        for (double[] e : expected) {
            assertEquals(e[1], f[(int) e[0]], e[1] * 1e-5, "pair " + (int) e[0]);
        }
        assertEquals(0.1 * Math.log(4) + 1, r.magnitude(), 1e-6);
        // the attention factor scales cos and sin: at position 0 the rotation is the identity
        Tensor x = Tensor.normal(Shape.of(1, 1, 1, 64), 0, 1, new SplittableRandom(1)).to(DEVICE);
        float[] y = r.apply(Var.of(x), new int[][] {{0}}).value().toArray();
        float[] in = x.toArray();
        for (int i = 0; i < in.length; i++) {
            assertEquals(in[i] * r.magnitude(), y[i], 1e-5);
        }
    }

    @Test
    void dynamicNtkGrowsTheBaseOnlyPastTheOriginalContext() {
        Rope.Settings s = new Rope.Settings(16, 10000, "dynamic", 2, 1, 4, 32, null);
        Rope r = new Rope(s);
        double[] plain = new Rope(new Rope.Settings(16, 10000, "default", 1, 1, 4, 32, null)).frequencies();
        Tensor x = Tensor.normal(Shape.of(1, 1, 64, 16), 0, 1, new SplittableRandom(2)).to(DEVICE);
        r.apply(Var.of(x.slice(2, 0, 32)), new int[][] {range(0, 32)});
        assertArrayEquals(plain, r.frequencies(), 0);
        r.apply(Var.of(x), new int[][] {range(0, 64)});
        double base = 10000 * Math.pow(2.0 * 64 / 32 - 1, 16 / 14.0);
        assertEquals(1 / Math.pow(base, 14 / 16.0), r.frequencies()[7], 1e-9);
        assertEquals(1, r.frequencies()[0], 1e-9);
    }

    @Test
    void readsYarnFromAHuggingFaceConfig() {
        Map<String, Object> config = Json.object(Json.parse("""
                {"model_type": "qwen2", "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64,
                 "num_hidden_layers": 2, "vocab_size": 40, "max_position_embeddings": 32768,
                 "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                                  "beta_fast": 16}}
                """));
        LlamaConfig c = LlamaConfig.fromHuggingFace(config);
        assertEquals("yarn", c.rope().type());
        assertEquals(16, c.rope().betaFast());
        assertEquals(1, c.rope().betaSlow());
        LlamaConfig back = LlamaConfig.fromHuggingFace(c.toHuggingFace());
        assertEquals(16, back.rope().betaFast());
        assertEquals(4, back.rope().factor());
    }

    @Test
    void chunkedPrefillGivesTheSameLogits() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 3, DEVICE, DType.FLOAT32);
        int[][] prompt = {tokens(30, 4), tokens(30, 5)};
        float[] whole = model.prefillChunk(0).nextAll(model.newState(2), prompt).toArray();
        LanguageModel.State s = model.newState(2);
        float[] pieces = model.prefillChunk(7).nextAll(s, prompt).toArray();
        assertArrayEquals(whole, pieces, 1e-4f);
        assertEquals(30, s.length());
        float[] nextWhole = model.prefillChunk(0).next(model.newState(2), prompt, null).toArray();
        Tensor[] withHidden = model.prefillChunk(8).nextWithHidden(model.newState(2), prompt, null);
        assertArrayEquals(nextWhole, withHidden[0].toArray(), 1e-4f);
        assertArrayEquals(new int[] {2, 30, 32}, withHidden[1].shape().toArray());
    }

    @Test
    void slidingWindowMatchesBetweenForwardAndTheCache() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 4, DEVICE, DType.FLOAT32);
        int[] seq = tokens(24, 6);
        float[] full = model.forward(new int[][] {seq}).value().toArray();
        model.slidingWindow(24);   // as long as the text: no change
        assertArrayEquals(full, model.forward(new int[][] {seq}).value().toArray(), 1e-5f);
        model.slidingWindow(5);
        float[] windowed = model.forward(new int[][] {seq}).value().toArray();
        int v = model.vocabSize();
        // the first five steps see everything either way, later ones do not
        assertArrayEquals(Arrays.copyOf(full, 5 * v), Arrays.copyOf(windowed, 5 * v), 1e-5f);
        assertFalse(Arrays.equals(full, windowed));
        // through a state, one prompt piece then token by token
        model.prefillChunk(4);
        LanguageModel.State s = model.newState(1);
        float[] first = model.nextAll(s, new int[][] {Arrays.copyOf(seq, 10)}).toArray();
        assertArrayEquals(Arrays.copyOf(windowed, 10 * v), first, 1e-4f);
        float[] last = null;
        for (int i = 10; i < seq.length; i++) {
            last = model.next(s, new int[][] {{seq[i]}}, null).toArray();
        }
        assertArrayEquals(Arrays.copyOfRange(windowed, windowed.length - v, windowed.length), last, 1e-4f);
        assertEquals(seq.length, s.length());
    }

    private static int[] range(int from, int to) {
        int[] r = new int[to - from];
        for (int i = 0; i < r.length; i++) {
            r[i] = from + i;
        }
        return r;
    }

    private static int[] tokens(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = r.nextInt(40);
        }
        return t;
    }
}
