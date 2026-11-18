package io.github.zzuegg.brainiac.experiments.robustness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.robust.DistributionalRobustness;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DistributionallyRobustTest {

    @Test
    void dualWeightsConcentrateOnLargeLosses() {
        double[] l = {0.1, 0.2, 3.0, 0.15, 2.0};
        double[] cvar = DistributionalRobustness.weights(DistributionalRobustness.Kind.CVAR, l, 0.4);
        assertEquals(0.5, cvar[2], 1e-9);
        assertEquals(0.5, cvar[4], 1e-9);
        double[] chi = DistributionalRobustness.weights(DistributionalRobustness.Kind.CHI_SQUARE, l, 1);
        double sum = 0;
        for (double w : chi) {
            sum += w;
        }
        assertEquals(1, sum, 1e-9);
        assertTrue(chi[2] > chi[4] && chi[4] > chi[1], "chi-square weights follow the losses");
    }

    @Test
    void droWithoutGroupLabels() {
        Map<String, double[]> r = DistributionallyRobust.subpopulation(1000, 0.95, 0, 400, 1);
        r.forEach((k, v) -> System.out.printf("DRO backgrounds %s: avg acc %.3f worst-group %.3f%n", k, v[0], v[1]));
        assertTrue(r.get("CVaR (alpha 0.1)")[1] > r.get("ERM")[1] || r.get("chi-square (lambda 0.3)")[1]
                > r.get("ERM")[1], "a DRO set raises worst-group accuracy over ERM");
        assertTrue(r.get("GroupDRO (group labels)")[1] >= r.get("ERM")[1], "GroupDRO oracle");
        Map<String, double[]> noisy = DistributionallyRobust.subpopulation(1000, 0.95, 0.1, 400, 1);
        noisy.forEach((k, v) -> System.out.printf(
                "DRO backgrounds, 10%% flipped labels %s: avg acc %.3f worst-group %.3f%n", k, v[0], v[1]));
    }

    @Test
    void wassersteinLogisticAgainstPgd() {
        double[] eps = {0, 0.1, 0.2};
        Map<String, double[]> r = DistributionallyRobust.perturbation(600, eps, 0.1, 2);
        r.forEach((k, v) -> System.out.printf(
                "Wasserstein DRO %s: acc %.3f, at eps 0.1 %.3f, at eps 0.2 %.3f (%.0f ms)%n", k, v[0], v[1], v[2],
                v[3]));
        assertTrue(r.get("Wasserstein DRO")[2] > r.get("logistic")[2], "WDRO robust accuracy at eps 0.2");
        assertTrue(r.get("Wasserstein DRO")[3] < r.get("PGD training")[3], "WDRO is cheaper than PGD training");
    }
}
