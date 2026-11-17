package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.GanExperiment.Constraint;

import org.junit.jupiter.api.Test;

class GanExperimentTest {

    @Test
    void spectrallyNormalizedWassersteinGanMovesTowardsTheData() {
        float[] d = GanExperiment.run(Constraint.SPECTRAL_NORM, 300, 1);
        System.out.println("SN-WGAN: sliced W1 " + d[0] + " -> " + d[1]);
        assertTrue(d[1] < 0.5f * d[0] && d[1] < 0.45f, d[0] + " -> " + d[1]);
    }

    @Test
    void gradientPenaltyGanStaysStable() {
        float[] d = GanExperiment.run(Constraint.GRADIENT_PENALTY, 100, 1);
        System.out.println("WGAN-GP: sliced W1 " + d[0] + " -> " + d[1]);
        assertTrue(Float.isFinite(d[1]) && d[1] < 2 * d[0], d[0] + " -> " + d[1]);
    }

    @Test
    void gradientPenaltyCriticEstimatesTheWassersteinDistance() {
        float[] e = GanExperiment.criticEstimate(3f, 300, 2);
        System.out.println("WGAN-GP critic: estimate " + e[0] + ", gradient norm " + e[1]);
        // the soft penalty trades: for the optimal linear critic with slope a, minimizing
        // -3a + 10 (a - 1)^2 gives a = 1 + 3 / 20, so the estimate is 3 * 1.15
        assertEquals(1.15f, e[1], 0.08f);
        assertEquals(3.45f, e[0], 0.25f);
    }
}
