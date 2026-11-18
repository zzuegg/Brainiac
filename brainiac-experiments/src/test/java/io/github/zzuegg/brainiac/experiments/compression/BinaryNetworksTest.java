package io.github.zzuegg.brainiac.experiments.compression;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BinaryNetworksTest {

    @Test
    void binaryModesAgainstTernaryAndDorefa() {
        List<CompactRow> rows = CompactModels.binary(80, 1);
        StringBuilder line = new StringBuilder("11.4.11 binary nets, digits acc (bits/weight):");
        for (CompactRow r : rows.subList(0, rows.size() - 1)) {
            line.append(String.format(" %s %.3f (%.2f);", r.method(), r.metric(), r.cost()));
        }
        CompactRow kernel = rows.getLast();
        line.append(String.format(" xnor kernel %.1fx faster than float at batch 1 (%.4f ms)", kernel.metric(),
                kernel.millis()));
        System.out.println(line);
        for (CompactRow r : rows.subList(0, rows.size() - 1)) {
            assertTrue(r.metric() > 0.2, "every variant learns above chance (0.1): " + r);
        }
        assertTrue(kernel.metric() > 1, "the popcount kernel beats the float kernel");
    }
}
