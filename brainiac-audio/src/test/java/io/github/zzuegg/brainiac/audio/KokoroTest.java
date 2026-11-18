package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Map;
import java.util.random.RandomGenerator;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Kokoro against the kokoro package with deterministic source noise (see make_kokoro.py). */
class KokoroTest {

    @Test
    void audioMatchesKokoro() throws Exception {
        Path dir = WhisperTest.resource("kokoro");
        Map<String, Object> expected = Json.object(Json.read(dir.resolve("expected.json")));
        try (Kokoro model = Kokoro.load(dir, TestDevice.get(), DType.FLOAT32)) {
            int[] ids = model.tokenize((String) expected.get("phonemes"));
            assertArrayEquals(ints(expected.get("ids")), ids);
            float[] style;
            try (WeightSource voice = WeightSource.open(dir.resolve("voices/test_voice.pt"))) {
                float[] pack = voice.tensor("", Device.cpu(), DType.FLOAT32).toArray();
                int row = Math.min(ids.length, 12) - 1;
                style = java.util.Arrays.copyOfRange(pack, row * 256, row * 256 + 256);
            }
            Kokoro.Output out = model.synthesize(ids, style, 1, KokoroTest.pseudoNoise());
            assertArrayEquals(ints(expected.get("durations")), out.durations());
            float[] ref = WhisperTest.floats(expected.get("audio"));
            assertEquals(ref.length, out.audio().length, "length");
            double scale = 0;
            for (float r : ref) {
                scale = Math.max(scale, Math.abs(r));
            }
            for (int i = 0; i < ref.length; i++) {
                assertEquals(ref[i], out.audio()[i], 2e-3 * scale, "sample " + i);
            }
            assertArrayEquals(ints(expected.get("fast_durations")), model.synthesize(ids, style, 1.5f, KokoroTest.pseudoNoise())
                    .durations());
        }
    }

    /** The noise make_kokoro.py uses: value k is {@code frac(sin(k) * 43758.5453) * 2 - 1}. */
    static RandomGenerator pseudoNoise() {
        return new RandomGenerator() {
            private long k;

            @Override
            public long nextLong() {
                throw new UnsupportedOperationException();
            }

            @Override
            public double nextGaussian() {
                double v = Math.sin(k++) * 43758.5453;
                return (float) ((v - Math.floor(v)) * 2 - 1);
            }
        };
    }

    private static int[] ints(Object o) {
        return Json.array(o).stream().mapToInt(v -> ((Number) v).intValue()).toArray();
    }
}
