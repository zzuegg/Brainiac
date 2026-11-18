package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.PinnTraining;

import java.util.List;

import org.junit.jupiter.api.Test;

class PikanDepthSweepTest {

    @Test
    void gatedKanDepthSweep() {
        List<PdeShootout.Row> rows = PdeShootout.depthSweep(PdeShootout.poisson(), 16, new int[] {1, 3},
                () -> new PinnTraining.Options().adam(100, 3e-3f).lbfgs(5).collocation(64), true, 1);
        StringBuilder line = new StringBuilder("8.3.3 Poisson depth sweep (KAN), rel L2 (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.4f (%.0f);", r.method(), r.error(), r.millis())));
        System.out.println(line);
        rows.forEach(r -> assertTrue(Double.isFinite(r.error()), line.toString()));
    }
}
