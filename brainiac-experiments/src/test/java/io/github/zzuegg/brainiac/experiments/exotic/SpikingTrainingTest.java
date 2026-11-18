package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.exotic.SpikingClassifier.SpikeLoss;
import io.github.zzuegg.brainiac.experiments.exotic.SpikingClassifier.Surrogate;

import org.junit.jupiter.api.Test;

class SpikingTrainingTest {

    @Test
    void tetAndAdaptiveSurrogatesAtFewTimeSteps() {
        double base = SpikingClassifier.direct(4, SpikeLoss.MEAN_OUTPUT, Surrogate.FAST_SIGMOID, 1, 200, 1);
        double tet = SpikingClassifier.direct(4, SpikeLoss.TET, Surrogate.FAST_SIGMOID, 1, 200, 1);
        double rect = SpikingClassifier.direct(4, SpikeLoss.MEAN_OUTPUT, Surrogate.RECTANGULAR, 1, 200, 1);
        double sched = SpikingClassifier.direct(4, SpikeLoss.MEAN_OUTPUT, Surrogate.SCHEDULED, 1, 200, 1);
        double grid = SpikingClassifier.direct(4, SpikeLoss.MEAN_OUTPUT, Surrogate.GRID, 1, 200, 1);
        double adaptive = SpikingClassifier.direct(4, SpikeLoss.MEAN_OUTPUT, Surrogate.ADAPTIVE, 1, 200, 1);
        System.out.printf("SNN direct T 4 (1 hidden): mean-output fast-sigmoid %.3f  TET %.3f  rectangular %.3f  "
                + "scheduled %.3f  A2SG grid %.3f  adaptive %.3f%n", base, tet, rect, sched, grid, adaptive);
        double base8 = SpikingClassifier.direct(8, SpikeLoss.MEAN_OUTPUT, Surrogate.FAST_SIGMOID, 1, 200, 1);
        double tet8 = SpikingClassifier.direct(8, SpikeLoss.TET, Surrogate.FAST_SIGMOID, 1, 200, 1);
        System.out.printf("SNN direct T 8 (1 hidden): mean-output fast-sigmoid %.3f  TET %.3f%n", base8, tet8);
        assertTrue(base > 0.7, "direct training learns at T = 4: " + base);
        assertTrue(tet > base - 0.03, "TET is not worse than the mean-output loss");
    }
}
