package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class KernelHypothesisTestsTest {

    @Test
    void mmdDetectsShapeDifferencesTheTTestMisses() {
        Map<String, double[]> r = KernelHypothesisTests.twoSample(100, 24, 1);
        System.out.printf("Two-sample power (n=100, alpha=0.05): %s%n", ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("null: MMD (median)")[0] <= 0.2, "MMD keeps its level");
        assertTrue(r.get("variance shift: MMD (median)")[0] > r.get("variance shift: t-test")[0],
                "MMD sees a variance change");
    }

    @Test
    void hsicDetectsNonLinearDependence() {
        Map<String, double[]> r = KernelHypothesisTests.independence(100, 24, 2);
        System.out.printf("Independence power (n=100, alpha=0.05): %s%n", ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("ring: HSIC (permutation)")[0] > 0.8, "HSIC finds the ring");
        assertTrue(r.get("ring: Pearson")[0] < 0.3, "Pearson misses the ring");
        assertTrue(r.get("null: HSIC (gamma)")[0] <= 0.2, "HSIC keeps its level");
    }
}
