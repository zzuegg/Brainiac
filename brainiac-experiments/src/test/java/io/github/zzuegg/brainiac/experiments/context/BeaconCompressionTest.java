package io.github.zzuegg.brainiac.experiments.context;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class BeaconCompressionTest {

    @Test
    void learnedBeaconsBeatPoolingAtEqualMemory() {
        Map<String, BeaconCompression.Cell[]> r = BeaconCompression.run(200, new int[] {32, 128}, 300, 1);
        r.forEach((k, v) -> System.out.printf("8.4.10 %-28s accuracy at 32 / 128 tokens: %.2f / %.2f (memory %.0f / %.0f)%n",
                k, v[0].accuracy(), v[1].accuracy(), v[0].memory(), v[1].memory()));
        String all = java.util.Arrays.deepToString(r.values().toArray());
        assertTrue(r.get("Activation Beacon 4x")[0].accuracy() > r.get("sliding window")[0].accuracy() + 0.2, all);
        assertTrue(r.get("Activation Beacon 8x")[1].accuracy() > r.get("gist tokens")[1].accuracy() + 0.2, all);
        assertTrue(r.get("Activation Beacon 4x")[1].accuracy() > 0.8, all);
    }
}
