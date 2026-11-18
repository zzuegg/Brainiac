package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class StaticDesignTest {

    @Test
    void eigEstimatorsAndOptimisers() {
        Map<String, double[]> e = ExperimentalDesign.estimators(10, 1);
        System.out.printf("EIG estimators (13.5.8) {bias, sd}: %s%n", NeuralLikelihoodTest.line(e));
        Map<String, double[]> s = ExperimentalDesign.staticDesigns(2);
        System.out.printf("Static designs (13.5.8) {EIG, evaluations}: %s%n", NeuralLikelihoodTest.line(s));
        assertTrue(e.get("PCE L=99")[0] < 0.05 && e.get("NCE L=99")[0] > -0.05, "PCE and NCE bracket the EIG");
        assertTrue(s.get("location: PCE ascent")[0] > 1.3 * s.get("location: random design")[0],
                "PCE ascent beats a random design");
    }
}
