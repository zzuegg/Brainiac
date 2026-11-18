package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PinnSecondOrderTest {

    @Test
    void naturalGradientAndNewtonCgSharpenPoisson() {
        PdeShootout.Pde pde = PdeShootout.poisson();
        List<PdeShootout.Row> rows = PdeShootout.optimisers(pde, 12, 2, 64, 200, 30, 10, 40, 1);
        StringBuilder line = new StringBuilder("9.5.15 Poisson, rel L2 (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.2e (%.0f);", r.method(), r.error(), r.millis())));
        System.out.println(line);
        for (PdeShootout.Row r : rows) {
            assertTrue(Double.isFinite(r.error()), line.toString());
        }
        double adam = rows.get(0).error();
        double engd = rows.get(2).error();
        assertTrue(engd < Math.max(adam, 0.05), line.toString());
    }
}
