package io.github.zzuegg.brainiac.experiments.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class RectangularDiscoveryTest {

    @Test
    void searchesForRankEleven() {
        List<KernelTuning.Row> rows = KernelTuning.discovery(2, 2, 3, 11, 60, 300, 1, 400, 20000, 1);
        StringBuilder line = new StringBuilder("8.5.16 2x2x3 rank-11 search (found, base=naive rank):");
        for (KernelTuning.Row r : rows) {
            line.append(String.format(" %s %.3g (base %.3g) [%s];", r.method(), r.metric(), r.baseline(), r.detail()));
        }
        System.out.println(line);
        boolean any = false;
        for (KernelTuning.Row r : rows) {
            any |= r.metric() == 1;
        }
        assertTrue(any, "some search finds a sub-naive algorithm");
    }
}
