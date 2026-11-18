package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TailsTest {

    @Test
    void extremeQuantilesAndSpot() {
        Map<String, double[]> t = TailsAndDependence.tails(2000, 8, 1);
        System.out.printf("Extreme quantiles (13.5.13) {rel err @1e-3, @1e-4}: %s%n", NeuralLikelihoodTest.line(t));
        Map<String, double[]> a = TailsAndDependence.anomalies(20000, 2);
        System.out.printf("SPOT q=1e-3 (13.5.13) {FA, recall, FA drift, recall drift}: %s%n",
                NeuralLikelihoodTest.line(a));
        assertTrue(t.get("POT-MLE")[1] < t.get("Gaussian tail")[1], "POT beats the Gaussian tail at 1e-4");
        assertTrue(a.get("SPOT")[0] < 5e-3, "SPOT keeps its false-alarm rate near the target");
        assertTrue(a.get("DSPOT")[2] < a.get("SPOT")[2] + 1e-3, "DSPOT handles the drift");
    }
}
