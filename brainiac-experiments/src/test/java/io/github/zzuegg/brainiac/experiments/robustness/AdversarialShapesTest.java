package io.github.zzuegg.brainiac.experiments.robustness;

import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;

import org.junit.jupiter.api.Test;

class AdversarialShapesTest {

    private static final float[] EVAL = {0.05f, 0.1f, 0.2f};

    @Test
    void adversarialTrainingOnShapesCnn() {
        for (AdversarialTraining.Method m : new AdversarialTraining.Method[] {AdversarialTraining.Method.STANDARD,
            AdversarialTraining.Method.FAST_FGSM, AdversarialTraining.Method.PGD_AT}) {
            print("shapes CNN", AdversarialRobustness.adversarial(AdversarialRobustness.Task.SHAPES_CNN, m, 0.1f,
                    EVAL, 80, 1));
        }
    }

    private static void print(String task, AdversarialRobustness.Adversarial a) {
        System.out.printf("Adversarial %-10s %-9s clean %.3f  PGD-20 eps 0.05/0.1/0.2: %.3f %.3f %.3f  train %5.0f ms%n",
                task, a.method(), a.clean(), a.robust()[0], a.robust()[1], a.robust()[2], a.millis());
    }
}
