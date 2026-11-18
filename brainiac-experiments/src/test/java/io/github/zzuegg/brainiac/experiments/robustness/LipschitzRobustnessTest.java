package io.github.zzuegg.brainiac.experiments.robustness;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class LipschitzRobustnessTest {

    private static final double[] RADII = {0.25, 0.5, 1.0};

    @Test
    void groupSortLipschitzNetworksCertifyByMargin() {
        for (AdversarialRobustness.LipschitzKind k : AdversarialRobustness.LipschitzKind.values()) {
            AdversarialRobustness.Lipschitz l = AdversarialRobustness.lipschitz(k, RADII, 250, 1);
            System.out.printf("Lipschitz %-16s clean %.3f  certified l2 at 0.25/0.5/1.0 %s  PGD %s  train %5.0f ms%n",
                    l.model(), l.clean(), fmt(l.certified()), fmt(l.pgd()), l.millis());
            assertTrue(l.clean() > 0.5, l.toString());
            for (int r = 0; r < RADII.length; r++) {
                assertTrue(l.pgd()[r] >= l.certified()[r] - 1e-9, "certificates hold against PGD: " + l.model());
            }
        }
    }

    private static String fmt(double[] v) {
        return Arrays.toString(Arrays.stream(v).map(x -> Math.round(x * 1000) / 1000.0).toArray());
    }
}
