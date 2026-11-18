package io.github.zzuegg.brainiac.experiments.causal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DoubleMlTest {

    @Test
    void crossFittedScoresRemoveTheConfoundingBias() {
        Map<String, HeterogeneousEffects.AteSummary> r = HeterogeneousEffects.averageEffect(600, 6, 15, 1);
        StringBuilder line = new StringBuilder("ATE (truth 1) bias/RMSE/coverage95:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f/%.2f |", k, v.bias(), v.rmse(), v.coverage())));
        System.out.println(line);
        double naive = Math.abs(r.get("naive").bias());
        assertTrue(Math.abs(r.get("DML-IRM (forest, cross-fit)").bias()) < naive, line.toString());
        assertTrue(Math.abs(r.get("DML-PLR (forest, cross-fit)").bias()) < naive, line.toString());
    }
}
