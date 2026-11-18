package io.github.zzuegg.brainiac.experiments.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class GraphRewritingTest {

    @Test
    void equalitySaturationAgainstGreedy() {
        List<KernelTuning.Row> rows = KernelTuning.graphs(1);
        StringBuilder line = new StringBuilder("8.5.17 graph rewriting (cost, base=original):");
        for (KernelTuning.Row r : rows) {
            line.append(String.format(" %s %.3g (base %.3g) [%s];", r.method(), r.metric(), r.baseline(), r.detail()));
        }
        System.out.println(line);
        for (KernelTuning.Row r : rows) {
            assertTrue(r.metric() <= r.baseline() + 1e-6, "rewriting never costs more: " + r);
            assertTrue(r.detail().contains("err=") && !r.detail().contains("NaN"), r.detail());
        }
    }
}
