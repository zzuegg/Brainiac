package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Tiny BERT and XLM-R models against transformers (make_encoders.py). */
class EncoderTest {

    static int[][] grid(Object list) {
        return Json.array(list).stream().map(LlamaTest::ints).toArray(int[][]::new);
    }

    static float[] floats(Object list) {
        List<Object> l = Json.array(list);
        float[] out = new float[l.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) l.get(i)).floatValue();
        }
        return out;
    }

    @Test
    void bertHiddenStatesMatchTransformers() throws IOException {
        Map<String, Object> e = LlamaTest.expected("bert/expected.json");
        Encoder bert = Encoder.load(LlamaTest.resource("bert"), LlamaTest.DEVICE, DType.FLOAT32);
        assertFalse(bert.hasClassifier());
        int[][] ids = grid(e.get("ids"));
        int[][] mask = grid(e.get("mask"));
        float[] hidden = bert.hidden(ids, mask, grid(e.get("types"))).toArray();
        float[] expected = floats(e.get("hidden"));
        // padded positions are not compared: transformers computes them too, but they mean nothing
        for (int b = 0; b < 2; b++) {
            for (int s = 0; s < 8; s++) {
                if (mask[b][s] == 1) {
                    for (int j = 0; j < 32; j++) {
                        int i = (b * 8 + s) * 32 + j;
                        assertEquals(expected[i], hidden[i], 1e-4f, "row " + b + " step " + s);
                    }
                }
            }
        }
        // the tokenizer's pair template: [CLS] a [SEP] b [SEP], b in segment 1
        PretrainedTokenizer tok = PretrainedTokenizer.load(LlamaTest.resource("bert"));
        int[] a = tok.encode("the cat sat", true);
        int[] b = tok.encode("on the mat", false);
        int[] pair = java.util.stream.IntStream.concat(java.util.Arrays.stream(a),
                java.util.stream.IntStream.concat(java.util.Arrays.stream(b), java.util.stream.IntStream.of(3))).toArray();
        assertArrayEquals(LlamaTest.ints(e.get("pair_ids")), pair);
    }

    @Test
    void poolingAndNormalizing() throws IOException {
        Map<String, Object> e = LlamaTest.expected("bert/expected.json");
        Encoder bert = Encoder.load(LlamaTest.resource("bert"), LlamaTest.DEVICE, DType.FLOAT32);
        int[][] ids = grid(e.get("ids"));
        int[][] mask = grid(e.get("mask"));
        float[] expected = bert.hidden(ids, mask, null).toArray();   // embed reads one segment
        float[][] cls = bert.embed(ids, mask, Encoder.Pooling.CLS, false);
        assertArrayEquals(java.util.Arrays.copyOfRange(expected, 8 * 32, 8 * 32 + 32), cls[1], 1e-4f);
        float[][] mean = bert.embed(ids, mask, Encoder.Pooling.MEAN, true);
        float[] m = new float[32];
        for (int s = 0; s < 5; s++) {
            for (int j = 0; j < 32; j++) {
                m[j] += expected[(8 + s) * 32 + j] / 5;
            }
        }
        double n = 0;
        for (float f : m) {
            n += f * f;
        }
        for (int j = 0; j < 32; j++) {
            m[j] /= (float) Math.sqrt(n);
        }
        assertArrayEquals(m, mean[1], 1e-4f);
        PretrainedTokenizer tok = PretrainedTokenizer.load(LlamaTest.resource("bert"));
        float[][] v = bert.embed(tok, List.of("the cat sat", "a dog ran fast ."), Encoder.Pooling.MEAN, true);
        assertEquals(2, v.length);
        double len = 0;
        for (float f : v[1]) {
            len += f * f;
        }
        assertEquals(1, len, 1e-4);
        assertThrows(IllegalStateException.class, () -> bert.classify(ids, mask, null));
    }

    @Test
    void rerankersScoreAsTransformers() throws IOException {
        for (String name : new String[] {"bert-rerank", "xlmr-rerank"}) {
            Map<String, Object> e = LlamaTest.expected(name + "/expected.json");
            Encoder r = Encoder.load(LlamaTest.resource(name), LlamaTest.DEVICE, DType.FLOAT32);
            assertTrue(r.hasClassifier());
            float[][] logits = r.classify(grid(e.get("ids")), grid(e.get("mask")),
                    e.containsKey("types") ? grid(e.get("types")) : null);
            assertArrayEquals(floats(e.get("logits")), new float[] {logits[0][0], logits[1][0]}, 1e-4f, name);
        }
        Encoder r = Encoder.load(LlamaTest.resource("bert-rerank"), LlamaTest.DEVICE, DType.FLOAT32);
        PretrainedTokenizer tok = PretrainedTokenizer.load(LlamaTest.resource("bert-rerank"));
        float[] scores = r.rerank(tok, "where is the cat ?", List.of("the cat sat on the mat .", "stock market"));
        // the same pairs through classify, built as the tokenizer's pair template does
        int[] q = tok.encode("where is the cat ?", false);
        int[] d = tok.encode("stock market", false);
        int[] ids = new int[q.length + d.length + 3];
        int[] types = new int[ids.length];
        ids[0] = 2;
        System.arraycopy(q, 0, ids, 1, q.length);
        ids[q.length + 1] = 3;
        System.arraycopy(d, 0, ids, q.length + 2, d.length);
        ids[ids.length - 1] = 3;
        java.util.Arrays.fill(types, q.length + 2, ids.length, 1);
        int[] mask = new int[ids.length];
        java.util.Arrays.fill(mask, 1);
        assertEquals(r.classify(new int[][] {ids}, new int[][] {mask}, new int[][] {types})[0][0], scores[1], 1e-4f);
    }
}
