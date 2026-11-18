package io.github.zzuegg.brainiac.experiments.robustness;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class AdversarialRobustnessTest {

    private static final float[] EVAL = {0.05f, 0.1f, 0.2f};

    @Test
    void adversarialTrainingHoldsRobustAccuracyOnDigits() {
        Map<AdversarialTraining.Method, AdversarialRobustness.Adversarial> rows =
                new EnumMap<>(AdversarialTraining.Method.class);
        for (AdversarialTraining.Method m : AdversarialTraining.Method.values()) {
            AdversarialRobustness.Adversarial a = AdversarialRobustness.adversarial(
                    AdversarialRobustness.Task.DIGITS_MLP, m, 0.1f, EVAL, 200, 1);
            rows.put(m, a);
            print("digits MLP", a);
        }
        AdversarialRobustness.Adversarial std = rows.get(AdversarialTraining.Method.STANDARD);
        AdversarialRobustness.Adversarial pgd = rows.get(AdversarialTraining.Method.PGD_AT);
        AdversarialRobustness.Adversarial fast = rows.get(AdversarialTraining.Method.FAST_FGSM);
        assertTrue(std.clean() > 0.9, std.toString());
        assertTrue(pgd.robust()[1] > std.robust()[1] + 0.2, "PGD-AT " + pgd.robust()[1] + " vs " + std.robust()[1]);
        assertTrue(fast.robust()[1] > std.robust()[1] + 0.1, "fast FGSM " + fast.robust()[1]);
        assertTrue(fast.millis() < pgd.millis(), "fast FGSM is cheaper");
    }

    private static void print(String task, AdversarialRobustness.Adversarial a) {
        System.out.printf("Adversarial %-10s %-9s clean %.3f  PGD-20 eps 0.05/0.1/0.2: %.3f %.3f %.3f  train %5.0f ms%n",
                task, a.method(), a.clean(), a.robust()[0], a.robust()[1], a.robust()[2], a.millis());
    }
}
