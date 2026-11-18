package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.PhysicsInformedNetwork;
import io.github.zzuegg.brainiac.experimental.sciml.PinnTraining;

import java.util.List;

import org.junit.jupiter.api.Test;

class PinnHardPdeTest {

    @Test
    void plainPinnStrugglesOnShocksAndStiffReactions() {
        StringBuilder line = new StringBuilder("8.3.1 hard PDEs, rel L2 (ms):");
        for (PdeShootout.Pde pde : List.of(PdeShootout.burgers(), PdeShootout.allenCahn(),
                PdeShootout.advection(10))) {
            PdeShootout.Row pinn = PdeShootout.pinn("PINN", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), 20, 3,
                    1, 1), new PinnTraining.Options().adam(200, 5e-3f).lbfgs(30).collocation(256), 1);
            PdeShootout.Row coarse = PdeShootout.coarseSolver(pde);
            line.append(String.format(" [%s] PINN %.3f (%.0f); coarse solver %.1e (%.1f);", pde.name(), pinn.error(),
                    pinn.millis(), coarse.error(), coarse.millis()));
            assertTrue(Double.isFinite(pinn.error()), line.toString());
        }
        System.out.println(line);
    }
}
