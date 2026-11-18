package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class WatermarkShootoutTest {

    @Test
    void tournamentAgainstGreenList() {
        WatermarkShootout.Setup setup = WatermarkShootout.setup(250, 1);
        Map<WatermarkShootout.Scheme, WatermarkShootout.Row> rows = WatermarkShootout.compare(setup, 100, 50,
                new int[] {25, 50}, new double[] {0.2}, 1.0, 2);
        rows.forEach((s, row) -> System.out.printf("Watermark %-10s TPR@1%%FPR by length %s  20%% edits %s  "
                + "judge log-lik %.3f  distinct-2 %.3f%n", s, row.tprByLength(), row.tprEdited(),
                row.judgeLogLikelihood(), row.distinct2()));
        WatermarkShootout.Row none = rows.get(WatermarkShootout.Scheme.NONE);
        WatermarkShootout.Row tour = rows.get(WatermarkShootout.Scheme.TOURNAMENT);
        assertTrue(tour.tprByLength().get(50) > 0.3, "the tournament watermark is detected at 50 tokens");
        assertTrue(tour.tprByLength().get(50) >= rows.get(WatermarkShootout.Scheme.GREEN_1).tprByLength().get(50),
                "the tournament is at least as detectable as the delta-1 green list");
        assertTrue(none.tprByLength().get(50) < 0.1, "unwatermarked text is rarely flagged");
        assertTrue(tour.judgeLogLikelihood() > rows.get(WatermarkShootout.Scheme.GREEN_2).judgeLogLikelihood(),
                "the tournament costs less quality than the delta-2 green list");
    }
}
