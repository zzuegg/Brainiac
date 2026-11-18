package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.eval.BudgetRankingStudy;

import java.util.List;

import org.junit.jupiter.api.Test;

class BudgetRankingsTest {

    @Test
    void forecastingRankingAgainstBudgetAndEarlyPrediction() {
        long t0 = System.nanoTime();
        BudgetRankings.Curves c = BudgetRankings.forecasting(List.of("DLinear", "PatchTST", "vanilla transformer",
                "N-HiTS"), 10, 3, 2, new long[] {11});
        BudgetRankingStudy.Result r = BudgetRankings.study(c, 3, 200);
        System.out.print(BudgetRankingStudy.table(r));
        System.out.printf("budget rankings (forecasting, 10..80 steps, 1 seed): tau(1x vs 8x) %.2f [%.2f, %.2f], "
                + "tau(4x) %.2f, crossovers %s, %.1f s%n", r.tau()[0], r.tauLow()[0], r.tauHigh()[0], r.tau()[2],
                r.crossovers(), (System.nanoTime() - t0) / 1e9);
        assertEquals(1.0, r.tau()[3], 1e-9);
        assertTrue(r.tau()[0] >= -1 && r.tau()[0] <= 1);
        for (double f : new double[] {0.1, 0.25}) {
            BudgetRankings.EarlyRanking e = BudgetRankings.earlyRanking(c, f, null);
            System.out.println("early ranking (forecasting) " + e);
            assertTrue(Double.isFinite(e.ensemble()) && Double.isFinite(e.powerLaw()));
        }
    }

    @Test
    void kendallTauAndCrossovers() {
        assertEquals(1.0, BudgetRankingStudy.kendallTau(new double[] {1, 2, 3}, new double[] {2, 5, 9}), 1e-12);
        assertEquals(-1.0, BudgetRankingStudy.kendallTau(new double[] {1, 2, 3}, new double[] {3, 2, 1}), 1e-12);
        double[][][] curves = {{{0.8, 0.75, 0.7, 0.65}}, {{0.9, 0.85, 0.6, 0.5}}};
        BudgetRankingStudy.Result r = BudgetRankingStudy.analyse(List.of("a", "b"), new int[] {1, 2, 4, 8}, curves,
                true, 0, 1);
        assertEquals(1, r.crossovers().size());
        assertEquals("b", r.crossovers().get(0).winner());
        assertEquals(4, r.crossovers().get(0).budget());
        assertEquals("a", r.crossovers().get(0).leaderAtSmallest());
    }

}
