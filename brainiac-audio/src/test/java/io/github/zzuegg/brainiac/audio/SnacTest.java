package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** SNAC decoding against the snac package with deterministic noise (see make_snac.py). */
class SnacTest {

    @Test
    void decodesAsSnac() throws Exception {
        check("snac");
    }

    @Test
    void decodesWithLocalAttention() throws Exception {
        check("snac-attention");
    }

    private static void check(String name) throws Exception {
        Path dir = WhisperTest.resource(name);
        Map<String, Object> expected = Json.object(Json.read(dir.resolve("expected.json")));
        int[][] codes = Json.array(expected.get("codes")).stream()
                .map(l -> Json.array(l).stream().mapToInt(v -> ((Number) v).intValue()).toArray())
                .toArray(int[][]::new);
        float[] want = WhisperTest.floats(expected.get("audio"));
        try (Snac snac = Snac.load(dir, TestDevice.get(), DType.FLOAT32)) {
            float[] got = snac.decode(codes, i -> KokoroTest.pseudoNoise());
            assertEquals(want.length, got.length, "samples");
            for (int i = 0; i < want.length; i++) {
                assertEquals(want[i], got[i], 2e-3, "sample " + i);
            }
        }
    }
}
