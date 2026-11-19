package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.Method;
import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.RoarRow;
import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.RoarStudy;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class RemoveAndRetrainTest {

    private static String fmt(double[] v) {
        return Arrays.toString(Arrays.stream(v).mapToObj(d -> String.format("%.3f", d)).toArray());
    }

    @Test
    void roarSeparatesInformativeAttributionsFromRandom() {
        RoarStudy s = ExplanationEvaluation.roar(new double[] {0.1, 0.2, 0.5}, 1);
        System.out.printf("ROAR MLP test accuracy %.3f, fractions %s%n", s.modelAccuracy(), fmt(s.fractions()));
        RoarRow random = null;
        RoarRow best = null;
        for (RoarRow r : s.rows()) {
            System.out.printf("ROAR %-20s ROAR %s KAR %s deletion AUC %.3f recovery %.2f%n", r.method(), fmt(r.roar()),
                    fmt(r.kar()), r.deletionAuc(), r.recovery());
            if (r.method() == Method.RANDOM) {
                random = r;
            } else if (best == null || r.roar()[1] < best.roar()[1]) {
                best = r;
            }
        }
        System.out.printf("ROAR method pairs ordered differently by deletion AUC and ROAR at 20%%: %d of %d%n",
                s.discordantPairs(), s.rows().size() * (s.rows().size() - 1) / 2);
        assertTrue(s.modelAccuracy() > 0.8, "the explained MLP learned the rule");
        assertTrue(best.roar()[1] < random.roar()[1] - 0.1, "the best attribution's ROAR drop beats random");
        assertTrue(best.kar()[1] > random.kar()[1] + 0.1, "the best attribution keeps more under KAR than random");
    }
}
