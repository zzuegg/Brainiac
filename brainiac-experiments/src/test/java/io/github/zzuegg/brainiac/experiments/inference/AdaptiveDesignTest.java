package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class AdaptiveDesignTest {

    @Test
    void deepAdaptiveDesign() {
        Map<String, double[]> a = ExperimentalDesign.adaptive(10, 200, 6, 1);
        System.out.printf("Adaptive design T=10 (13.5.9) {sPCE, RMSE, sPCE shifted, RMSE shifted, ms/step}: %s%n",
                NeuralLikelihoodTest.line(a));
        assertTrue(a.get("DAD")[0] > a.get("RANDOM")[0], "DAD beats random designs");
    }
}
