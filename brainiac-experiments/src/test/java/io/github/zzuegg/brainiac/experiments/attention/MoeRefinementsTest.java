package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MoeRefinementsTest {

    @Test
    void expertChoiceBalancesAndCapacityRoutingLearns() {
        List<MoeRefinements.Row> rows = MoeRefinements.routers(400, 1);
        System.out.print(MoeRefinements.table(rows));
        double zero = MoeRefinements.zeroPredictorMse();
        MoeRefinements.Row topK = rows.get(0);
        MoeRefinements.Row ec = rows.get(1);
        MoeRefinements.Row tc = rows.get(2);
        MoeRefinements.Row balanced = rows.get(3);
        System.out.printf("val MSE (zero predictor %.4f): expert choice %.4f (load CV %.3f), token choice cf 1.25 %.4f "
                + "(CV %.3f, dropped %.3f), +balance %.4f (CV %.3f) vs top-2 %.4f (CV %.3f)%n", zero,
                ec.validationMse(), ec.loadImbalance(), tc.validationMse(), tc.loadImbalance(), tc.dropped(),
                balanced.validationMse(), balanced.loadImbalance(), topK.validationMse(), topK.loadImbalance());
        for (MoeRefinements.Row r : rows) {
            assertTrue(r.validationMse() < 0.6 * zero, r.name());
        }
        assertEquals(0, ec.loadImbalance(), 1e-9);
        assertTrue(balanced.loadImbalance() < tc.loadImbalance());
        assertTrue(tc.dropped() < 0.1);
    }

    @Test
    void upcyclingBeatsContinuedDenseTraining() {
        List<MoeRefinements.Row> rows = MoeRefinements.upcycling(600, 300, 1);
        System.out.print(MoeRefinements.table(rows));
        System.out.printf("val MSE after dense pre-training: upcycled %.4f, drop-upcycled %.4f vs dense continued %.4f "
                + "vs MoE from scratch at equal compute %.4f%n", rows.get(1).validationMse(),
                rows.get(2).validationMse(), rows.get(0).validationMse(), rows.get(3).validationMse());
        assertTrue(rows.get(1).validationMse() < rows.get(0).validationMse());
        assertTrue(rows.get(2).validationMse() < rows.get(0).validationMse() * 1.05);
    }
}
