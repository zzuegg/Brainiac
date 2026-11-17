package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.DType;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class EmbeddingTest {

    @Test
    void poolsTheFinalHiddenStatesAsTransformers() {
        Llama model = Llama.load(LlamaTest.resource("qwen2"), LlamaTest.DEVICE, DType.FLOAT32);
        int[] ids = {1, 5, 9, 3};
        // transformers: hidden_states[-1] of the last token and the mean, scaled to length 1
        float[] last = model.embed(new int[][] {ids}, Llama.Pooling.LAST, true)[0];
        assertArrayEquals(new float[] {0.16959143f, 0.10296189f, 0.025714802f, -0.22317117f},
                Arrays.copyOf(last, 4), 1e-4f);
        float[] mean = model.embed(new int[][] {ids}, Llama.Pooling.MEAN, true)[0];
        assertArrayEquals(new float[] {0.22782434f, 0.037882596f, 0.13636792f, -0.1193987f},
                Arrays.copyOf(mean, 4), 1e-4f);
        double sq = 0;
        for (float v : last) {
            sq += v * v;
        }
        assertEquals(1, sq, 1e-4);

        // sequences of different lengths in one batch embed as they do alone
        int[] shorter = {2, 7};
        float[][] both = model.embed(new int[][] {shorter, ids}, Llama.Pooling.MEAN, true);
        assertArrayEquals(model.embed(new int[][] {shorter}, Llama.Pooling.MEAN, true)[0], both[0], 1e-4f);
        assertArrayEquals(mean, both[1], 1e-4f);
    }
}
