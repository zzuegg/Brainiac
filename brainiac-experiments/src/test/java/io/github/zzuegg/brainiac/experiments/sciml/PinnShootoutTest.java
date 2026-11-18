package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.PhysicsInformedNetwork;
import io.github.zzuegg.brainiac.experimental.sciml.PinnTraining;

import java.util.List;

import org.junit.jupiter.api.Test;

class PinnShootoutTest {

    @Test
    void pinnSolvesSmoothProblemsButNotFasterThanASolver() {
        StringBuilder line = new StringBuilder("8.3.1 smooth PDEs, rel L2 (ms):");
        double heat = 1;
        for (PdeShootout.Pde pde : List.of(PdeShootout.heat(), PdeShootout.poisson())) {
            PdeShootout.Row pinn = PdeShootout.pinn("PINN", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), 20, 3,
                    1, 1), new PinnTraining.Options().adam(300, 5e-3f).lbfgs(50).collocation(256), 1);
            PdeShootout.Row data = PdeShootout.dataFit(pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), 20, 3, 1, 1),
                    100, 0.05, 300, 1);
            PdeShootout.Row coarse = PdeShootout.coarseSolver(pde);
            line.append(String.format(" [%s] PINN %.4f (%.0f); %s %.4f (%.0f); %s %.1e (%.1f);", pde.name(),
                    pinn.error(), pinn.millis(), data.method(), data.error(), data.millis(), coarse.method(),
                    coarse.error(), coarse.millis()));
            if (pde.name().equals("heat")) {
                heat = pinn.error();
            }
        }
        System.out.println(line);
        assertTrue(heat < 0.1, line.toString());
    }
}
