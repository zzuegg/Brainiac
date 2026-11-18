package io.github.zzuegg.brainiac.experiments.robustness;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class CertifiedRobustnessTest {

    private static final double[] RADII = {0.25, 0.5, 1.0};

    @Test
    void smoothingNeedsNoiseTraining() {
        AdversarialRobustness.Smoothed plain = AdversarialRobustness.smoothing(0.25f, false, RADII, 60, 1000, 1);
        AdversarialRobustness.Smoothed noisy = AdversarialRobustness.smoothing(0.25f, true, RADII, 60, 1000, 1);
        AdversarialRobustness.Smoothed wide = AdversarialRobustness.smoothing(0.5f, true, RADII, 60, 1000, 1);
        double[] reference = AdversarialRobustness.pgdL2Reference(0.5f, RADII, 1);
        for (AdversarialRobustness.Smoothed s : new AdversarialRobustness.Smoothed[] {plain, noisy, wide}) {
            System.out.printf("Smoothing %-26s certified l2 acc at 0.25/0.5/1.0: %s  abstain %.3f  samples %d%n",
                    s.model(), fmt(s.certified()), s.abstain(), s.samples());
        }
        System.out.printf("Smoothing reference: l2 PGD-AT (eps 0.5) empirical acc at 0.25/0.5/1.0: %s%n",
                fmt(reference));
        assertTrue(noisy.certified()[1] >= plain.certified()[1], "noise training helps");
        assertTrue(noisy.certified()[0] > 0.3, noisy.toString());
    }

    @Test
    void pgdModelsAreRobustButNotVerifiable() {
        AdversarialRobustness.Verified[] rows = new AdversarialRobustness.Verified[4];
        int i = 0;
        for (AdversarialRobustness.Certified c : AdversarialRobustness.Certified.values()) {
            rows[i++] = AdversarialRobustness.certified(c, 0.05f, 300, 1);
        }
        for (AdversarialRobustness.Verified v : rows) {
            System.out.printf("Verified linf 0.05 %-9s clean %.3f  PGD %.3f  IBP-verified %.3f  CROWN-verified %.3f"
                    + "  train %5.0f ms%n", v.model(), v.clean(), v.pgd(), v.ibp(), v.crown(), v.millis());
        }
        assertTrue(rows[1].pgd() > rows[1].ibp(), "PGD-AT is more robust empirically than IBP can verify");
        assertTrue(rows[2].ibp() > rows[1].ibp() + 0.2, "IBP training makes models IBP-verifiable");
        for (AdversarialRobustness.Verified v : rows) {
            assertTrue(v.crown() >= v.ibp() - 1e-9, "CROWN is at least as tight as IBP: " + v);
            assertTrue(v.pgd() >= v.crown() - 1e-9, "no verified input falls to PGD: " + v);
        }
    }

    private static String fmt(double[] v) {
        return Arrays.toString(Arrays.stream(v).map(x -> Math.round(x * 1000) / 1000.0).toArray());
    }
}
