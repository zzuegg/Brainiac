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
    void forecastingRankingAgainstBudget() {
        long t0 = System.nanoTime();
        BudgetRankings.Curves c = BudgetRankings.forecasting(List.of("DLinear", "PatchTST", "vanilla transformer",
                "Autoformer", "FEDformer", "TimesNet", "N-HiTS"), 20, 4, 2, new long[] {11, 12});
        BudgetRankingStudy.Result r = BudgetRankings.study(c, 4, 200);
        System.out.print(BudgetRankingStudy.table(r));
        System.out.printf("budget rankings (forecasting, 20..320 steps, 2 seeds): tau(1x vs 16x) %.2f [%.2f, %.2f], "
                + "tau(4x) %.2f, crossovers %s, %.1f s%n", r.tau()[0], r.tauLow()[0], r.tauHigh()[0], r.tau()[2],
                r.crossovers(), (System.nanoTime() - t0) / 1e9);
        assertEquals(1.0, r.tau()[4], 1e-9);
        assertTrue(r.tau()[0] < 1.0, "the 1x ranking should differ from the 16x ranking");
    }
}
