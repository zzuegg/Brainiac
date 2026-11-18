package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class DependenceTest {

    @Test
    void vineCopulas() {
        Map<String, double[]> d = TailsAndDependence.dependence(1000, false, 1);
        System.out.printf("Vines (13.5.14) {held-out log c, |dL01|, |dU02|, rel P_low, rel P_high, tree}: %s%n",
                NeuralLikelihoodTest.line(d));
        assertTrue(d.get("R_VINE")[0] > d.get("Gaussian copula")[0], "the vine beats the Gaussian copula");
        assertTrue(d.get("R_VINE")[5] >= 2.0 / 3, "the first tree is recovered");
    }

    @Test
    void functionalData() {
        Map<String, double[]> f = TailsAndDependence.functional(200, 2);
        System.out.printf("Functional (13.5.15) {angles, recon RMSE, y RMSE, beta err | AUROC mag, shape}: %s%n",
                NeuralLikelihoodTest.line(f));
        assertTrue(f.get("PACE (sparse)")[2] < f.get("interpolate+PCA (sparse)")[2], "PACE reconstructs better");
        assertTrue(f.get("functional boxplot")[0] > 0.9, "the boxplot catches magnitude outliers");
    }
}
