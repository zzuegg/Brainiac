package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Tiny BART, mBART and Marian models against transformers (make_bart.py). */
class BartTest {

    private static final int[][] IDS = {{5, 17, 42, 9, 2, 1}, {7, 3, 2, 1, 1, 1}};
    private static final int[][] MASK = {{1, 1, 1, 1, 1, 0}, {1, 1, 1, 0, 0, 0}};
    private static final int[][] DECODER = {{2, 11, 23, 4}, {2, 8, 8, 30}};

    private static float[] expected(String model, String name) {
        try (WeightSource w = WeightSource.open(LlamaTest.resource(model).resolve("expected.safetensors"))) {
            return w.tensor(name, Device.cpu(), DType.FLOAT32).toArray();
        }
    }

    private static void close(float[] expected, Tensor actual, float tolerance, String what) {
        float[] a = actual.to(DType.FLOAT32).toArray();
        assertEquals(expected.length, a.length, what);
        for (int i = 0; i < a.length; i++) {
            assertEquals(expected[i], a[i], tolerance * Math.max(1, Math.abs(expected[i])), what + " [" + i + "]");
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"bart", "mbart", "marian"})
    void matchesTransformers(String name) {
        Bart model = Bart.load(LlamaTest.resource(name), LlamaTest.DEVICE, DType.FLOAT32);
        assertEquals(32, model.hiddenSize());
        Tensor encoded = model.encode(IDS, MASK);
        close(expected(name, "encoded"), encoded, 1e-4f, "encoded");
        close(expected(name, "logits"), model.logits(encoded, MASK, DECODER), 1e-4f, "logits");
        float[] g = expected(name, "generated");
        int steps = g.length / IDS.length;
        int[][] generated = model.generate(IDS, MASK, steps - 1);
        for (int i = 0; i < IDS.length; i++) {
            // transformers keeps the start token and pads after the end token
            int row = i;
            int[] want = IntStream.range(1, steps).map(s -> (int) g[row * steps + s]).takeWhile(t -> t != 2).toArray();
            assertArrayEquals(want, generated[i], name + " row " + i);
        }
        assertTrue(model.toString().contains("2+3 layers"));
    }

    @Test
    void forcesTheFirstToken(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) throws java.io.IOException {
        // as bart-large-cnn: forced_bos_token_id 0; transformers generates [2, 0, 40, 40, 40, 40, 40]
        // and [2, 0, 55, ...] for this model, and the <s> is not text
        java.nio.file.Path from = LlamaTest.resource("bart");
        for (String f : new String[] {"model.safetensors", "generation_config.json"}) {
            java.nio.file.Files.copy(from.resolve(f), dir.resolve(f));
        }
        String config = java.nio.file.Files.readString(from.resolve("config.json"))
                .replaceFirst("\\{", "{\"forced_bos_token_id\": 0,");
        java.nio.file.Files.writeString(dir.resolve("config.json"), config);
        int[][] generated = Bart.load(dir, LlamaTest.DEVICE, DType.FLOAT32).generate(IDS, MASK, 6);
        assertArrayEquals(new int[] {40, 40, 40, 40, 40}, generated[0]);
        assertArrayEquals(new int[] {55, 55, 55, 55, 55}, generated[1]);
    }
}
