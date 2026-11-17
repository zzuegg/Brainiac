package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MemorizationCapacityTest {

    @Test
    void capacityGrowsWithParameters() {
        List<MemorizationCapacity.Size> sizes = MemorizationCapacity.sizes(new int[] {4, 8, 12, 16}, 12, 400, 1);
        for (MemorizationCapacity.Size s : sizes) {
            System.out.printf("Memorization hidden %d: %d params, %d strings (%.0f bits): %.0f bits memorised"
                    + " = %.2f bits/param;"
                    + " int8 %.0f, int4 %.0f bits%n", s.hidden(), s.parameters(), s.strings(), s.datasetBits(),
                    s.memorizedBits(), s.bitsPerParameter(), s.int8Bits(), s.int4Bits());
        }
        assertTrue(sizes.get(3).memorizedBits() > sizes.get(0).memorizedBits(), sizes.toString());
        for (MemorizationCapacity.Size s : sizes) {
            assertTrue(s.bitsPerParameter() > 0.1 && s.bitsPerParameter() < 8, s.toString());
            assertTrue(s.int4Bits() <= s.memorizedBits() * 1.05, s.toString());
        }
    }

    @Test
    void memorisationSaturatesWithDatasetSize() {
        List<MemorizationCapacity.Point> pts = MemorizationCapacity.datasetSweep(8, new int[] {10, 30, 90, 270, 540},
                12, 400, 2);
        for (MemorizationCapacity.Point p : pts) {
            System.out.printf("Memorization sweep hidden 8: %d strings (%.0f bits) -> %.0f bits memorised%n", p.strings(),
                    p.datasetBits(), p.memorizedBits());
        }
        assertTrue(pts.get(0).memorizedBits() > 0.5 * pts.get(0).datasetBits(), pts.toString());
        assertTrue(pts.get(4).memorizedBits() < 0.5 * pts.get(4).datasetBits(), pts.toString());
    }
}
