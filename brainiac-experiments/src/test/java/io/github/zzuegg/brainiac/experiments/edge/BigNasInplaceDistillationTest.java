package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.edge.EdgeDeployment.OfaFollowUp;
import io.github.zzuegg.brainiac.experiments.edge.EdgeDeployment.OfaFollowUpRow;

import org.junit.jupiter.api.Test;

class BigNasInplaceDistillationTest {

    @Test
    void inplaceDistillationAgainstProgressiveShrinking() {
        // at 300 steps, 6 deployments, 60 points (26 s): BigNAS largest 0.520 smallest 0.465, progressive
        // 0.915 / 0.780, largest only 0.935 / 0.200, random 0.860 / 0.830, from scratch 0.935 / 0.915;
        // predictor tau 0.412 (24 points) and 0.481 (60 points)
        OfaFollowUp f = EdgeDeployment.onceForAllFollowUp(new int[] {120}, 4, 40, 1);
        for (OfaFollowUpRow r : f.rows()) {
            System.out.println("once-for-all 120 steps: " + r);
        }
        System.out.printf("predictor Kendall tau: 24 points %.3f, %d points %.3f%n", f.tauFew(), f.points(),
                f.tauMany());
        OfaFollowUpRow bignas = f.rows().get(0);
        assertTrue(bignas.largest() >= 0 && bignas.largest() <= 1, bignas.toString());
        assertTrue(Double.isFinite(f.tauMany()), "tau is measured");
    }
}
