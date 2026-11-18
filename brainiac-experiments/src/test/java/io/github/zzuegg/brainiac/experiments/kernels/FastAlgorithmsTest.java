package io.github.zzuegg.brainiac.experiments.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class FastAlgorithmsTest {

    @Test
    void winogradAndStrassenAgainstCore() {
        List<KernelTuning.Row> rows = KernelTuning.fastAlgorithms(new int[] {256, 512}, 30, 1);
        StringBuilder line = new StringBuilder("8.5.15 fast algorithms (speed-up, base=rel err; CNN acc):");
        for (KernelTuning.Row r : rows) {
            line.append(String.format(" %s %.3g (base %.3g) [%s];", r.method(), r.metric(), r.baseline(), r.detail()));
        }
        System.out.println(line);
        for (KernelTuning.Row r : rows) {
            if (r.method().contains("speed-up")) {
                assertTrue(r.baseline() < 1e-3, "exact up to rounding: " + r);
            }
        }
        assertEquals(find(rows, "cnn fp32 direct"), find(rows, "cnn fp32 F(2,3)"), 0.02,
                "fp32 Winograd keeps the CNN's accuracy");
    }

    private static double find(List<KernelTuning.Row> rows, String name) {
        return rows.stream().filter(r -> r.method().equals(name)).findFirst().orElseThrow().metric();
    }
}
