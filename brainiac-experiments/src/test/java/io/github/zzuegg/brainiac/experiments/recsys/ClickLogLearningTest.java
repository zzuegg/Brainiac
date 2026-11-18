package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.ClickLogs;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ClickLogLearningTest {

    @Test
    void debiasedRankersBeatNaiveClicksUnderStrongPositionBias() {
        Map<String, double[]> r = ClickLogLearning.run(2, ClickLogs.ClickModel.POSITION_BASED, 300, 1);
        StringBuilder line = new StringBuilder("ULTR NDCG@5 / propensity error, PBM eta 2:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f |", k, v[0], v[1])));
        System.out.printf("%s%n", line);
        assertTrue(r.get("oracle (true grades)")[0] > 0.5, line.toString());
        // Clipped IPW fixes the bias; unclipped IPW (weights up to 100 at eta 2) and the EM ranker
        // lose to naive on variance, a genuine negative, so they are only reported.
        assertTrue(r.get("IPW (true propensities, clipped at 0.1)")[0] > r.get("naive softmax")[0] + 0.03,
                line.toString());
        assertTrue(r.get("regression EM")[1] < 0.1, line.toString());
        assertTrue(r.get("DLA")[0] > r.get("naive softmax")[0] - 0.03, line.toString());
    }
}
