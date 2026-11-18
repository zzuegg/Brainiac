package io.github.zzuegg.brainiac.experiments.robustness;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class CorruptionRobustnessTest {

    @Test
    void augmentationSchemesAreComparedByMeanCorruptionError() {
        List<AdversarialRobustness.Corrupted> rows = AdversarialRobustness.corruptions(300, 1);
        for (AdversarialRobustness.Corrupted r : rows) {
            double[] ce = r.corruptionError();
            System.out.printf("Corruptions %-14s clean %.3f mCE %.3f (noise %.2f blur %.2f contrast %.2f elastic %.2f "
                    + "quantise %.2f; sev-5 errors %.2f %.2f %.2f %.2f %.2f)%n", r.method(), r.clean(), r.mce(), ce[0],
                    ce[1], ce[2], ce[3], ce[4], r.error()[0][4], r.error()[1][4], r.error()[2][4], r.error()[3][4],
                    r.error()[4][4]);
        }
        AdversarialRobustness.Corrupted standard = rows.get(0);
        double best = Double.MAX_VALUE;
        for (AdversarialRobustness.Corrupted r : rows) {
            assertTrue(Double.isFinite(r.mce()), r.method().toString());
            best = Math.min(best, r.mce());
        }
        assertTrue(standard.clean() > 0.85, "standard clean " + standard.clean());
        assertTrue(best < 1.0, "some scheme beats standard augmentation, best mCE " + best);
    }
}
