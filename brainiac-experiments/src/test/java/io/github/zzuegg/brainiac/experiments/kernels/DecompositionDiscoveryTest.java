package io.github.zzuegg.brainiac.experiments.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DecompositionDiscoveryTest {

    @Test
    void searchesForStrassen() {
        List<KernelTuning.Row> rows = KernelTuning.discovery(2, 2, 2, 7, 100, 100, 2, 2000, 20000, 1);
        StringBuilder line = new StringBuilder("8.5.16 2x2x2 rank-7 search (found, base=naive rank):");
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
