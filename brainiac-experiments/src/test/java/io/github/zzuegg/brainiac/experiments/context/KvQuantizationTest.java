package io.github.zzuegg.brainiac.experiments.context;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.QuantizedKvCache;

import java.util.Map;

import org.junit.jupiter.api.Test;

class KvQuantizationTest {

    @Test
    void kiviBeatsPerTokenKeysAndEvictionAtEqualMemory() {
        Map<String, KvQuantization.Outcome> r = KvQuantization.run(384, 1);
        StringBuilder line = new StringBuilder("KV quantization (rel. error, KB):");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%d;", k, v.relativeError(), v.bytes() / 1024)));
        System.out.println(line);
        KvQuantization.Outcome kivi2 = r.get("KIVI 2-bit");
        assertTrue(kivi2.bytes() * 5 < r.get("float32").bytes(), "2-bit must hold a fifth of float32 or less");
        assertTrue(r.get("KIVI 4-bit").relativeError() < kivi2.relativeError());
        assertTrue(kivi2.relativeError() < 0.7 * r.get("2-bit, per-token keys").relativeError());
        assertTrue(kivi2.relativeError() < 0.7 * r.get("H2O at KIVI-2 bytes").relativeError());
    }

    @Test
    void residualWindowStaysExactAndGroupsQuantizeWhole() {
        QuantizedKvCache c = new QuantizedKvCache(1, 4, 2, 8, 8);
        for (int t = 0; t < 15; t++) {
            c.append(new float[] {t, -t, 1, 2}, new float[] {1, 2, 3, 4});
        }
        assertEquals(15, c.fullPrecisionTokens(), "nothing is quantized before residual + group tokens");
        c.append(new float[] {15, -15, 1, 2}, new float[] {1, 2, 3, 4});
        assertEquals(8, c.fullPrecisionTokens());
        assertEquals(16, c.size());
    }
}
