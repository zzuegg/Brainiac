package io.github.zzuegg.brainiac.experiments.algorithms;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.KeyStreams;
import io.github.zzuegg.brainiac.experiments.algorithms.SketchShootout.FreqRow;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Item 11.2.14: Count-Min, conservative update, Count-Sketch, Misra-Gries, Space-Saving and HyperLogLog. */
class SketchShootoutTest {

    static FreqRow find(List<FreqRow> rows, String method) {
        return rows.stream().filter(r -> r.method().equals(method)).findFirst().orElseThrow();
    }

    @Test
    void sketchesOnZipfStreams() {
        int universe = 20_000;
        int[] skewed = KeyStreams.zipf(200_000, universe, 1.2, 1);
        int[] flat = KeyStreams.zipf(200_000, universe, 0.8, 2);
        int[] drifting = KeyStreams.driftingZipf(200_000, universe, 1.2, 4, 0.2, 3);
        List<FreqRow> s12 = SketchShootout.frequency(skewed, universe, 1024, 50, 1);
        List<FreqRow> s08 = SketchShootout.frequency(flat, universe, 1024, 50, 2);
        List<FreqRow> dz = SketchShootout.frequency(drifting, universe, 1024, 50, 3);
        Map<Integer, double[]> hll = SketchShootout.distinct(skewed, new int[] {6, 10, 14}, 4);
        System.out.println("Zipf 1.2, 1024 words: " + s12);
        System.out.println("Zipf 0.8, 1024 words: " + s08);
        System.out.println("drifting Zipf 1.2: " + dz);
        hll.forEach((p, v) -> System.out.printf("HLL p=%d (%d registers): rel. error %.4f, bound 1.04/sqrt(m) %.4f%n", p,
                (int) v[0], v[1], v[2]));
        System.out.printf("11.2.14 Zipf-1.2 weighted err: CM %.2e, CU %.2e, CS %.2e, FeatureHasher %.2e, reservoir %.2e;"
                + " HH recall MG %.2f, SS %.2f, CM %.2f; HLL p=10 err %.3f (bound %.3f)%n",
                find(s12, "COUNT_MIN").weightedError(), find(s12, "CONSERVATIVE_UPDATE").weightedError(),
                find(s12, "COUNT_SKETCH").weightedError(), find(s12, "FeatureHasher").weightedError(),
                find(s12, "reservoir sample").weightedError(), find(s12, "MISRA_GRIES").recall(),
                find(s12, "SPACE_SAVING").recall(), find(s12, "COUNT_MIN").recall(), hll.get(10)[1], hll.get(10)[2]);
        assertTrue(find(s12, "CONSERVATIVE_UPDATE").meanError() <= find(s12, "COUNT_MIN").meanError(),
                "conservative update never raises Count-Min's error");
        assertTrue(hll.get(10)[1] < 3 * hll.get(10)[2], "HLL stays near its bound");
    }
}
