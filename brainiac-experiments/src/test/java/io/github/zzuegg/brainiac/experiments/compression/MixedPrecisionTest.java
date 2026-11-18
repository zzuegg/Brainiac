package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.compress.HessianAwareBitAllocation;

import java.util.List;

import org.junit.jupiter.api.Test;

class MixedPrecisionTest {

    @Test
    void allocationRespectsTheBudgetAndFavoursSensitiveLayers() {
        double[][] omega = {{100, 10, 1}, {1, 0.1, 0.01}};
        int[] bits = HessianAwareBitAllocation.allocate(omega, new int[] {2, 4, 8}, new long[] {100, 100}, 5);
        assertEquals(5, HessianAwareBitAllocation.averageBits(bits, new long[] {100, 100}), 1e-9);
        assertEquals(8, bits[0], "the sensitive layer gets the bits");
    }

    @Test
    void hessianAwareMixedPrecisionAtTwoAndAHalfBits() {
        List<CompactRow> rows = CompactModels.mixedPrecision(2.5, 120, 1);
        StringBuilder line = new StringBuilder("8.5.10 HAWQ-V2 bit allocation at 2.5 bits, digits acc @avg bits:");
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.3f @%.2f;", r.method(), r.metric(), r.cost()));
        }
        System.out.println(line);
        CompactRow uniform2 = rows.get(1);
        CompactRow hawq = rows.get(3);
        assertTrue(hawq.cost() <= 2.5 + 1e-9, "HAWQ stays within the budget: " + hawq);
        assertTrue(hawq.metric() >= uniform2.metric() - 0.02, "HAWQ at 2.5 bits is no worse than uniform 2 bits");
    }
}
