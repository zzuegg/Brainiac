package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.PinnTraining;

import java.util.List;

import org.junit.jupiter.api.Test;

class PinnTrainingFixesTest {

    @Test
    void trainingFixesOnAdvection() {
        PdeShootout.Pde pde = PdeShootout.advection(5);
        List<PdeShootout.Row> rows = PdeShootout.fixes(pde, 16, 3,
                () -> new PinnTraining.Options().adam(250, 5e-3f).lbfgs(30).collocation(128), 1.0, 1);
        StringBuilder line = new StringBuilder("8.3.2 advection beta=5, rel L2 (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.3f (%.0f);", r.method(), r.error(), r.millis())));
        System.out.println(line);
        rows.forEach(r -> assertTrue(Double.isFinite(r.error()), line.toString()));
    }
}
