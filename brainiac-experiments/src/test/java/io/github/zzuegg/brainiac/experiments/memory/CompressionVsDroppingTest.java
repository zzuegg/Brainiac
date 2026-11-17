package io.github.zzuegg.brainiac.experiments.memory;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CompressionVsDroppingTest {

    @Test
    void foldingRecallsWhatDroppingLoses() {
        Map<String, CompressionVsDropping.Result> r = CompressionVsDropping.run(48, 8, 128, 1);
        double drop = r.get("DROP").recall();
        double linear = r.get("LINEAR").recall();
        double delta = r.get("DELTA").recall();
        assertTrue(Math.abs(drop) < 0.15, "drop " + drop);
        assertTrue(linear > drop + 0.05, "linear " + linear + " vs drop " + drop);
        assertTrue(delta > drop + 0.03, "delta " + delta + " vs drop " + drop);
        assertTrue(r.get("LINEAR_DPFP").recall() > drop + 0.2, "dpfp " + r.get("LINEAR_DPFP"));
        assertTrue(r.get("DELTA_DPFP").recall() > drop + 0.2, "dpfp delta " + r.get("DELTA_DPFP"));
        assertTrue(r.get("POOL").recall() > drop, "pool " + r.get("POOL").recall());
        assertEquals(0, r.get("DROP").memoryFloats());
        // constant memory, however many pairs stream through
        assertEquals(r.get("LINEAR").memoryFloats(),
                CompressionVsDropping.run(200, 8, 128, 1).get("LINEAR").memoryFloats());
    }
}
