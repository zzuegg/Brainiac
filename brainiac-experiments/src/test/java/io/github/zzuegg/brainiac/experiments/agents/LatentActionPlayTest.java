package io.github.zzuegg.brainiac.experiments.agents;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class LatentActionPlayTest {

    @Test
    void discoversActionsFromFramesAlone() {
        LatentActionPlay.Result r = LatentActionPlay.run(4000, 1);
        System.out.printf("latent actions from action-free frames: action recovery %.3f (chance 0.25), next-cell "
                + "accuracy driven by latent codes %.3f vs action-conditioned world model %.3f%n",
                r.actionRecovery(), r.latentCellAccuracy(), r.baselineCellAccuracy());
        assertTrue(r.actionRecovery() > 0.7, "the codes name the moves");
        assertTrue(r.latentCellAccuracy() > 0.9, "the codes control the model");
        assertTrue(r.latentCellAccuracy() > r.baselineCellAccuracy() - 0.05, "close to the labeled baseline");
    }
}
