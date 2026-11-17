package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.optim.DepthTransferExperiment.Sweep;

import org.junit.jupiter.api.Test;

class DepthTransferExperimentTest {

    @Test
    void completePsLearningRateTransfersAcrossDepth() {
        int[] depths = {2, 4, 8, 16};
        float[] rates = {1e-3f, 3e-3f, 1e-2f, 3e-2f, 1e-1f};
        Sweep sp = DepthTransferExperiment.sweep(0, depths, rates, 16, 120, 1);
        Sweep mup = DepthTransferExperiment.sweep(0.5, depths, rates, 16, 120, 1);
        Sweep complete = DepthTransferExperiment.sweep(1, depths, rates, 16, 120, 1);
        StringBuilder line = new StringBuilder();
        for (Sweep s : new Sweep[] {complete, mup, sp}) {
            line.append(" |");
            for (int d = 0; d < depths.length; d++) {
                line.append(String.format(" L%d %.0e", depths[d], rates[s.best(d)]));
            }
        }
        System.out.printf("Depth transfer 2->16 (best lr per depth, CompleteP | depth-muP | SP):%s; regret %.3f / "
                + "%.3f / %.3f, shift %d / %d / %d%n", line, complete.regret(), mup.regret(), sp.regret(),
                complete.maxShift(), mup.maxShift(), sp.maxShift());
        assertTrue(complete.regret() <= sp.regret() + 0.005f, "CompleteP transfers at least as well as SP");
        assertTrue(complete.maxShift() <= 1, "the best rate stays within one grid step");
    }
}
