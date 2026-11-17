package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelMergeTest {

    private static float[] merge(ModelMerge.Method m, float[] base, float[]... models) {
        return ModelMerge.merge(m, "w", base, List.of(models));
    }

    @Test
    void linearAndTaskArithmetic() {
        assertArrayEquals(new float[] {3, 1}, merge(new ModelMerge.Linear(new double[] {1, 3}), null,
                new float[] {0, 4}, new float[] {4, 0}), 1e-6f);
        float[] base = {1, 1};
        float[] a = {2, 1};
        float[] b = {1, 3};
        assertArrayEquals(new float[] {2, 2}, merge(new ModelMerge.TaskArithmetic(new double[] {1, 0.5}), base, a, b),
                1e-6f);
        // DARE that keeps everything is task arithmetic
        assertArrayEquals(new float[] {2, 2}, merge(new ModelMerge.Dare(new double[] {1, 0.5}, 1, 3), base, a, b),
                1e-6f);
    }

    @Test
    void slerpFollowsTheArc() {
        float h = (float) Math.sqrt(0.5);
        assertArrayEquals(new float[] {h, h}, merge(new ModelMerge.Slerp(0.5), null, new float[] {1, 0},
                new float[] {0, 1}), 1e-6f);
        // nearly parallel: a straight line
        assertArrayEquals(new float[] {1.5f, 3}, merge(new ModelMerge.Slerp(0.5), null, new float[] {1, 2},
                new float[] {2, 4}), 1e-6f);
        assertThrows(IllegalArgumentException.class, () -> merge(new ModelMerge.Slerp(0.5), null, new float[] {1}));
    }

    @Test
    void tiesTrimsElectsAndAverages() {
        float[] base = {0, 0, 0, 0};
        float[] a = {3, -1, 2, 0.5f};    // keeps 3 and 2
        float[] b = {-1, -4, 1, 0.2f};   // keeps -4 and the first of the two 1s
        assertArrayEquals(new float[] {3, -4, 2, 0},
                merge(new ModelMerge.Ties(new double[] {1, 1}, 0.5), base, a, b), 1e-6f);
        // agreeing values are averaged
        assertArrayEquals(new float[] {2, 0}, merge(new ModelMerge.Ties(new double[] {1, 1}, 1), new float[] {0, 0},
                new float[] {1, 1}, new float[] {3, -1}), 1e-6f);
    }

    @Test
    void dareDropsAndRescales() {
        float[] base = new float[10_000];
        float[] m = new float[10_000];
        java.util.Arrays.fill(m, 1);
        float[] out = merge(new ModelMerge.Dare(new double[] {1}, 0.25, 7), base, m);
        int kept = 0;
        double sum = 0;
        for (float v : out) {
            assertTrue(v == 0 || v == 4, "kept values are scaled by 1 / density");
            kept += v != 0 ? 1 : 0;
            sum += v;
        }
        assertEquals(2500, kept, 150);
        assertEquals(1, sum / out.length, 0.06);
    }

    @Test
    void mergesModelFolders(@TempDir Path dir) throws IOException {
        Path qwen = LlamaTest.resource("qwen2");
        Path same = dir.resolve("same");
        // a model merged with itself is itself, here split into small parts
        ModelMerge.merge(new ModelMerge.Linear(new double[] {1, 2}), null, List.of(qwen, qwen), same, 4096);
        assertTrue(Files.exists(same.resolve("config.json")));
        assertTrue(Files.exists(same.resolve("model.safetensors.index.json")));
        Llama a = Llama.load(qwen, Device.cpu(), DType.FLOAT32);
        Llama b = Llama.load(same, Device.cpu(), DType.FLOAT32);
        int[][] ids = {{1, 5, 9, 3}};
        assertArrayEquals(a.forward(ids).value().toArray(), b.forward(ids).value().toArray(), 1e-5f);

        // the base plus half of (model - base), where the model is the base shifted by 2
        Path shifted = dir.resolve("shifted");
        Files.createDirectories(shifted);
        Files.copy(qwen.resolve("config.json"), shifted.resolve("config.json"));
        Map<String, Tensor> w = new java.util.LinkedHashMap<>();
        try (SafeTensors s = SafeTensors.open(qwen.resolve("model.safetensors"))) {
            for (String n : s.names()) {
                w.put(n, s.tensor(n, Device.cpu(), DType.FLOAT32).add(2f));
            }
            SafeTensors.write(shifted.resolve("model.safetensors"), w);
        }
        Path half = dir.resolve("half");
        ModelMerge.merge(new ModelMerge.TaskArithmetic(new double[] {0.5}), qwen, List.of(shifted), half);
        assertTrue(Files.exists(half.resolve("model.safetensors")));
        try (SafeTensors orig = SafeTensors.open(qwen.resolve("model.safetensors"));
                SafeTensors merged = SafeTensors.open(half.resolve("model.safetensors"))) {
            for (String n : orig.names()) {
                float[] o = orig.tensor(n, Device.cpu(), DType.FLOAT32).toArray();
                float[] m = merged.tensor(n, Device.cpu(), DType.FLOAT32).toArray();
                for (int i = 0; i < o.length; i++) {
                    assertEquals(o[i] + 1, m[i], 1e-5f, n);
                }
            }
        }
    }
}
