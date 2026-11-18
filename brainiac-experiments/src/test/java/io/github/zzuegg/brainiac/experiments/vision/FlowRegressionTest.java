package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class FlowRegressionTest {

    @Test
    void cnnRegressesSmallMotions() {
        double[] cnn = DensePrediction.regressedFlow(DensePrediction.FlowNet.CNN, false, 150, 1);
        double[] zero = DensePrediction.classicalFlow(DensePrediction.FlowMethod.ZERO, false, 0.02f, 1);
        System.out.printf("flow EPE all/moving, 0-4 px: CNN regressor %.3f/%.3f, zero flow %.3f/%.3f%n", cnn[0],
                cnn[1], zero[0], zero[1]);
        assertTrue(cnn[1] < zero[1], "the regressor learns some motion");
    }
}
