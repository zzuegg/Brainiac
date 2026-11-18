package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.PinnTraining;

import java.util.List;

import org.junit.jupiter.api.Test;

class PinnHardFixesTest {

    @Test
    void marchingAndExactPeriodicityAttackFastAdvection() {
        PdeShootout.Pde pde = PdeShootout.advection(30);
        List<PdeShootout.Row> rows = PdeShootout.hardFixes(pde, PdeShootout::advection, new double[] {1, 10, 30}, 16,
                2, () -> new PinnTraining.Options().adam(150, 5e-3f).collocation(192), 4, 0, 1);
        StringBuilder line = new StringBuilder("9.5.14 advection beta=30, rel L2 (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.3f (%.0f);", r.method(), r.error(), r.millis())));
        System.out.println(line);
        double plain = rows.get(0).error();
        double best = rows.stream().mapToDouble(PdeShootout.Row::error).min().orElseThrow();
        for (PdeShootout.Row r : rows) {
            assertTrue(Double.isFinite(r.error()), line.toString());
        }
        assertTrue(best <= plain, line.toString());
    }
}
