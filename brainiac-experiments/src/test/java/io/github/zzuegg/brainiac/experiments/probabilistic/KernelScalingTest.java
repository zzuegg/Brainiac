package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class KernelScalingTest {

    @Test
    void nystromAndFalkonScaleKernelRidge() {
        Map<String, double[]> r = KernelScaling.compare("spirals", new int[] {1000, 20000, 50000},
                new int[] {50, 200}, 1000, 1);
        System.out.printf("Spirals {test error, fit s}: %s%n", ProbabilisticProgramsTest.line(r));
        Map<String, double[]> c = KernelScaling.compare("circles", new int[] {1000}, new int[] {20}, 1000, 2);
        System.out.printf("Circles {test error, fit s}: %s%n", ProbabilisticProgramsTest.line(c));
        double exact = r.get("exact KRR n=1000")[0];
        assertTrue(r.get("FALKON n=1000 m=200")[0] < exact + 0.03, "FALKON matches exact KRR");
        assertTrue(r.get("FALKON n=20000 m=200")[0] <= r.get("RFF+ridge n=20000 m=200")[0] + 0.01,
                "FALKON at least as good as RFF at equal m");
    }
}
