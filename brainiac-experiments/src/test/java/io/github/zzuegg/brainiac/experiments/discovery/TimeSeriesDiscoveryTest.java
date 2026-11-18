package io.github.zzuegg.brainiac.experiments.discovery;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TimeSeriesDiscoveryTest {

    @Test
    void pcmciControlsFalsePositivesUnderAutocorrelation() {
        Map<String, double[]> v = TimeSeriesDiscovery.compare(TimeSeriesDiscovery.Kind.VAR, 6, 500, 2, 0, 1);
        System.out.printf("11.3.5 VAR(2) d=6 {AUROC, FPR@0.05}: %s%n", DiscoveryShootoutTest.line(v));
        Map<String, double[]> a = TimeSeriesDiscovery.compare(TimeSeriesDiscovery.Kind.AUTOCORRELATED, 6, 500, 2, 0,
                2);
        System.out.printf("11.3.5 autocorrelated VAR(2) {AUROC, FPR@0.05}: %s%n", DiscoveryShootoutTest.line(a));
        assertTrue(v.get("Granger (conditional VAR)")[0] > 0.85, DiscoveryShootoutTest.line(v));
        assertTrue(a.get("PCMCI (ParCorr)")[1] <= a.get("lagged correlation")[1], DiscoveryShootoutTest.line(a));
        assertTrue(a.get("PCMCI (ParCorr)")[1] < 0.15, DiscoveryShootoutTest.line(a));
    }

    @Test
    void neuralGrangerOnNonlinearSystems() {
        Map<String, double[]> r = TimeSeriesDiscovery.compare(TimeSeriesDiscovery.Kind.NONLINEAR, 5, 400, 1, 300, 3);
        System.out.printf("11.3.5 nonlinear lagged d=5 {AUROC, FPR}: %s%n", DiscoveryShootoutTest.line(r));
        Map<String, double[]> l = TimeSeriesDiscovery.compare(TimeSeriesDiscovery.Kind.LORENZ96, 6, 400, 1, 300, 4);
        System.out.printf("11.3.5 Lorenz-96 d=6 {AUROC, FPR}: %s%n", DiscoveryShootoutTest.line(l));
        assertTrue(r.get("neural Granger (cMLP)")[0] > 0.6, DiscoveryShootoutTest.line(r));
    }
}
