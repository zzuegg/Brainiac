package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.experimental.eval.BudgetRankingStudy;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class MixerBudgetRankingsTest {

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
