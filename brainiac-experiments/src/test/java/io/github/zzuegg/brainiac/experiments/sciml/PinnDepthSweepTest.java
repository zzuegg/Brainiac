package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.PinnTraining;

import java.util.List;

import org.junit.jupiter.api.Test;

class PinnDepthSweepTest {

    @Test
    void pirateNetStaysTrainableWhenDeep() {
        List<PdeShootout.Row> rows = PdeShootout.depthSweep(PdeShootout.poisson(), 16, new int[] {1, 4},
                () -> new PinnTraining.Options().adam(200, 3e-3f).lbfgs(20).collocation(64), false, 1);
        StringBuilder line = new StringBuilder("8.3.3 Poisson depth sweep, rel L2 (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.4f (%.0f);", r.method(), r.error(), r.millis())));
        System.out.println(line);
        rows.forEach(r -> assertTrue(Double.isFinite(r.error()), line.toString()));
        assertTrue(rows.get(3).error() < rows.get(2).error(), line.toString());   // PirateNet x4 vs MLP x4
    }
}
