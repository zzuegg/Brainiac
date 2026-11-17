package io.github.zzuegg.brainiac.experiments.exotic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class KolmogorovArnoldFitTest {
    @Test
    void gridExtensionContinuesTrainingOnAFinerGrid() {
        KolmogorovArnoldFit k = KolmogorovArnoldFit.of(3, 5, 2e-2f, 11);
        float before = k.testRmse();
        k.train(300);
        float coarse = k.testRmse();
        k.refine(10);
        float refined = k.testRmse();
        k.train(300);
        float fine = k.testRmse();
        System.out.printf("KAN rmse: init %.4f, G=3 %.4f, refined %.4f, G=10 %.4f%n", before, coarse, refined, fine);
        assertTrue(coarse < 0.2f * before);
        assertTrue(refined < 1.5f * coarse + 1e-3f, "refinement lost the fit");
        assertTrue(fine < 0.7f * coarse, "finer grid did not help");
    }
}
