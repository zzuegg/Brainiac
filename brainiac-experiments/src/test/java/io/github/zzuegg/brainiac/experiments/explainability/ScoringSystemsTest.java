package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableModels.ScoreRow;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class ScoringSystemsTest {

    @Test
    void optimisedScoresBeatRounding() {
        // {nonlinear, scale, k}: non-integer planted log-odds, and an interaction under a tight budget
        double[][] settings = {{0, 0.4, 5}, {1, 1, 3}};
        for (double[] s : settings) {
            boolean nonlinear = s[0] > 0;
            List<ScoreRow> rows = InterpretableModels.scoring(1000, 10, (int) s[2], nonlinear, s[1], 1);
            for (ScoreRow r : rows) {
                System.out.printf("Scores %s scale %.1f k=%d %-16s train loss %.4f test loss %.4f AUROC %.3f "
                        + "ECE %.3f size %2d time %7.1f ms%s %s%n", nonlinear ? "nonlinear" : "additive ", s[1], r.k(),
                        r.method(), r.trainLoss(), r.logLoss(), r.auroc(), r.ece(), r.size(), r.millis(),
                        r.certified() ? " certified" : "", Arrays.toString(r.points()));
            }
            ScoreRow slim = rows.get(0);
            ScoreRow faster = rows.get(1);
            ScoreRow rounded = rows.get(2);
            assertTrue(slim.trainLoss() <= rounded.trainLoss() + 1e-9, "RiskSLIM is no worse than rounding");
            assertTrue(faster.trainLoss() <= rounded.trainLoss() + 1e-9, "FasterRisk is no worse than rounding");
        }
    }
}
