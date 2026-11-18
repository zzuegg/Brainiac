package io.github.zzuegg.brainiac.experiments.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class AutoTuningTest {

    @Test
    void costModelAgainstRandomGridAndEvolution() {
        List<KernelTuning.Row> rows = KernelTuning.autotune(128, 32, 24, 1);
        StringBuilder line = new StringBuilder("8.5.14 autotune 128^3 GFLOP/s:");
        for (KernelTuning.Row r : rows) {
            line.append(String.format(" %s %.3g (base %.3g) [%s];", r.method(), r.metric(), r.baseline(), r.detail()));
        }
        System.out.println(line);
        assertTrue(rows.get(rows.size() - 1).metric() < 1e-4, "every schedule computes the product");
        for (KernelTuning.Row r : rows.subList(1, 5)) {
            assertTrue(r.metric() > 0.05, "tuned kernels run: " + r);
        }
    }
}
