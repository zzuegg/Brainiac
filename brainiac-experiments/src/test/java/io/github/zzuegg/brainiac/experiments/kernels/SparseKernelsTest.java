package io.github.zzuegg.brainiac.experiments.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SparseKernelsTest {

    @Test
    void sparseFormatsAgainstDense() {
        List<KernelTuning.Row> rows = KernelTuning.sparse(512, 64, new double[] {0.5, 0.75, 0.9, 0.95}, 1);
        StringBuilder line = new StringBuilder("8.5.18 sparse kernels (speed-up vs core dense, base=rel err):");
        for (KernelTuning.Row r : rows) {
            line.append(String.format(" %s %.3g (base %.3g) [%s];", r.method(), r.metric(), r.baseline(), r.detail()));
        }
        System.out.println(line);
        for (KernelTuning.Row r : rows) {
            assertTrue(r.baseline() < 1e-4, "sparse kernels compute the product: " + r);
        }
    }
}
