package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.eval.BudgetRankingStudy;
import io.github.zzuegg.brainiac.experimental.eval.CurveExtrapolator;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class BudgetRankingsTest {

    @Test
    void forecastingRankingAgainstBudgetAndEarlyPrediction() {
        long t0 = System.nanoTime();
        BudgetRankings.Curves c = BudgetRankings.forecasting(List.of("DLinear", "PatchTST", "vanilla transformer",
                "FEDformer", "N-HiTS"), 20, 4, 2, new long[] {11, 12});
        BudgetRankingStudy.Result r = BudgetRankings.study(c, 4, 200);
        System.out.print(BudgetRankingStudy.table(r));
        System.out.printf("budget rankings (forecasting, 20..320 steps, 2 seeds): tau(1x vs 16x) %.2f [%.2f, %.2f], "
                + "tau(4x) %.2f, crossovers %s, %.1f s%n", r.tau()[0], r.tauLow()[0], r.tauHigh()[0], r.tau()[2],
                r.crossovers(), (System.nanoTime() - t0) / 1e9);
        assertEquals(1.0, r.tau()[4], 1e-9);
        assertTrue(r.tau()[0] >= -1 && r.tau()[0] <= 1);
        for (double f : new double[] {0.1, 0.25}) {
            BudgetRankings.EarlyRanking e = BudgetRankings.earlyRanking(c, f, null);
            System.out.println("early ranking (forecasting) " + e);
            assertTrue(Double.isFinite(e.ensemble()) && Double.isFinite(e.powerLaw()));
        }
    }

    @Test
    void extrapolationBeatsLastValueOnPriorCurves() {
        SplittableRandom r = new SplittableRandom(3);
        CurveExtrapolator.Pfn pfn = new CurveExtrapolator.Pfn(10, 40, 24, 1, 0.003f, new SplittableRandom(4));
        double loss = pfn.pretrain(250, 32, 5);
        double[] err = new double[4];
        int curves = 30;
        double[] x = new double[20];
        for (int i = 0; i < x.length; i++) {
            x[i] = 0.1 * (i + 1) / x.length;
        }
        for (int k = 0; k < curves; k++) {
            double[] full = CurveExtrapolator.samplePriorCurve(x, r);
            double[] y = java.util.Arrays.copyOf(full, x.length);
            double truth = full[x.length];
            err[0] += Math.abs(y[y.length - 1] - truth) / curves;
            err[1] += Math.abs(CurveExtrapolator.powerLawLeastSquares(x, y).predict(1) - truth) / curves;
            err[2] += Math.abs(CurveExtrapolator.ensemble(x, y, 1, 300, k).mean() - truth) / curves;
            err[3] += Math.abs(pfn.predict(x, y, 1).mean() - truth) / curves;
        }
        System.out.printf("curve extrapolation from 10%% of the budget, 30 prior curves: MAE last value %.4f, power law "
                + "%.4f, ensemble %.4f, LC-PFN %.4f (PFN loss %.2f)%n", err[0], err[1], err[2], err[3], loss);
        assertTrue(err[2] < err[0], "the ensemble beats the last value");
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

    @Test
    void mixerCurvesThroughTheSpecs() {
        Map<String, LayerSpec> all = BudgetRankings.mixerLineup();
        Map<String, LayerSpec> some = new LinkedHashMap<>();
        for (String k : List.of("S4D", "Mamba", "transformer", "gated NoPE")) {
            some.put(k, all.get(k));
        }
        BudgetRankings.Curves c = BudgetRankings.mixers(some, MixerShootout.recall(4, 4, 2), 10, 3, 2,
                new long[] {1});
        BudgetRankingStudy.Result r = BudgetRankings.study(c, 3, 0);
        System.out.print("mixer lineup on recall, 10..80 steps:\n" + BudgetRankingStudy.table(r));
        System.out.println("mixer crossovers: " + r.crossovers());
        assertEquals(4, r.methods().size());
    }
}
