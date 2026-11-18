package io.github.zzuegg.brainiac.experiments.symbolic;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LogicInferenceTest {

    private static LogicInference.Row find(List<LogicInference.Row> rows, String task, String method) {
        return rows.stream().filter(r -> r.task().startsWith(task) && r.method().startsWith(method)).findFirst()
            .orElseThrow();
    }

    private static String brief(List<LogicInference.Row> rows) {
        return rows.stream().map(LogicInference.Row::toString).collect(Collectors.joining("; "));
    }

    @Test
    void knowledgeCompilation() {
        List<LogicInference.Row> rows = LogicInference.compilation(new int[] {4, 6, 7}, 30, 1);
        System.out.printf("13.2.11 compilation (value vs ref): %s%n", brief(rows));
        for (LogicInference.Row r : rows) {
            if (r.method().contains("models") || r.task().startsWith("grid-path-") && !r.task().contains("WMC")
                    && !r.task().contains("valid")) {
                assertEquals(r.reference(), r.value(), 1e-6 * r.reference(), r.toString());
            }
        }
        assertEquals(find(rows, "exactly-1", "DNNF WMC").reference(), find(rows, "exactly-1", "DNNF WMC").value(),
            1e-4);
        assertEquals(find(rows, "grid-path-3x3", "DNNF").reference(), find(rows, "grid-path-3x3", "DNNF").value(),
            1e-9);
        assertTrue(find(rows, "BN n=30 det=0.5", "WMC (DNNF) max error").value() < 1e-6, "WMC is exact");
        List<LogicInference.Row> sl = LogicInference.semanticLossPaths(4, 300, 1500, 5.0, 1);
        System.out.printf("13.2.11 semantic loss on paths (valid rate, ref = exact rate, size = test SL): %s%n",
            brief(sl));
        assertTrue(sl.get(1).value() > sl.get(0).value() + 0.3, "compiled semantic loss raises validity");
    }
}
