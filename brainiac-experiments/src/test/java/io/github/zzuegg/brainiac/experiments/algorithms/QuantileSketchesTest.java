package io.github.zzuegg.brainiac.experiments.algorithms;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class QuantileSketchesTest {

    @Test
    void sketchesBeatReservoirAtTheTails() {
        List<SketchShootout.QuantileRow> rows = SketchShootout.quantiles(200_000, 600, 1);
        for (SketchShootout.QuantileRow r : rows) {
            System.out.printf("12.2.7 %-15s %-30s %6d stored, rank error q.5 %.5f q.9 %.5f q.99 %.5f q.999 %.5f,"
                    + " relative error q.999 %.4f%n", r.stream(), r.method(), r.stored(), r.rank50(), r.rank90(),
                    r.rank99(), r.rank999(), r.relative999());
        }
        for (SketchShootout.QuantileRow r : rows) {
            if (r.method().startsWith("KLL") || r.method().startsWith("Greenwald") || r.method().startsWith("t-d")) {
                assertTrue(r.rank50() < 0.02 && r.rank99() < 0.01, r.toString());
            }
        }
        SketchShootout.QuantileRow td = pick(rows, "Pareto (1.5)", "t-digest");
        SketchShootout.QuantileRow res = pick(rows, "Pareto (1.5)", "reservoir");
        assertTrue(td.relative999() < res.relative999(), "t-digest beats reservoir at the 0.999 tail");
    }

    private static SketchShootout.QuantileRow pick(List<SketchShootout.QuantileRow> rows, String stream,
            String method) {
        return rows.stream().filter(r -> r.stream().equals(stream) && r.method().startsWith(method)).findFirst()
                .orElseThrow();
    }
}
