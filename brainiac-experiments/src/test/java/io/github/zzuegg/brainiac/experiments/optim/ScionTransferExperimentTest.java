package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.optim.ScionTransferExperiment.Method;
import io.github.zzuegg.brainiac.experiments.optim.ScionTransferExperiment.Sweep;

import org.junit.jupiter.api.Test;

class ScionTransferExperimentTest {

    @Test
    void scionsLearningRateTransfersFromNarrowToWide() {
        int[] widths = {32, 128};
        float[] rates = {0.003f, 0.01f, 0.03f, 0.1f, 0.3f};
        Sweep scion = ScionTransferExperiment.sweep(Method.SCION, widths, rates, 150, 1);
        Sweep muon = ScionTransferExperiment.sweep(Method.MUON_MUP, widths, rates, 150, 1);
        Sweep adam = ScionTransferExperiment.sweep(Method.ADAM, widths, rates, 150, 1);
        System.out.printf("Scion lr transfer 32->128: best-rate shift %d grid steps, regret %.3f, best loss %.3f; "
                + "Muon+muP shift %d, regret %.3f, best %.3f; plain Adam shift %d, regret %.3f, best %.3f%n",
                scion.shift(), scion.regret(), scion.loss()[1][scion.best(1)], muon.shift(), muon.regret(),
                muon.loss()[1][muon.best(1)], adam.shift(), adam.regret(), adam.loss()[1][adam.best(1)]);
        assertTrue(scion.regret() < 0.02f, "the narrow model's best rate is (nearly) best for the wide one");
        assertTrue(scion.loss()[1][scion.best(1)] < 1.25f, "and trains it");
    }
}
