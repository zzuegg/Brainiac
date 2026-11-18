package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.ValueScalingStudy;
import org.junit.jupiter.api.Test;

class PlasticityAndScaleScalingTest {

    @Test
    void tdCriticsOverAWidthBatchGridAndThePassiveCritic() {
        PlasticityAndScale.Scaling s = PlasticityAndScale.scaling(12, new int[] {16, 64, 128}, new int[] {16, 128}, 300,
                1);
        StringBuilder b = new StringBuilder();
        for (ValueScalingStudy.Point p : s.grid()) {
            b.append(String.format("w%d b%d held-out TD %.3f MC %.2f, ", p.width(), p.batch(), p.heldOutTd(),
                    p.heldOutMc()));
        }
        System.out.printf("W10 4.17 cart-pole TD critics: %s passive: own %.3f, wide targets %.3f, wide %.3f; "
                + "best batch = %.1f w^%.2f%n", b, s.passive()[0].heldOutTd(), s.passive()[1].heldOutTd(),
                s.passive()[2].heldOutTd(), s.bestBatchLaw()[0], s.bestBatchLaw()[1]);
        assertTrue(s.grid().size() == 6);
        for (ValueScalingStudy.Point p : s.grid()) {
            assertTrue(Double.isFinite(p.heldOutTd()) && Double.isFinite(p.heldOutMc()), p.toString());
        }
    }
}
