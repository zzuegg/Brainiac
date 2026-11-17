package io.github.zzuegg.brainiac.experiments.exotic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DiffractiveShapeClassifierTest {
    @Test
    void trainedPhaseMasksSortShapesOntoDetectors() {
        DiffractiveShapeClassifier d = DiffractiveShapeClassifier.of(32, 2, 1, 12, 5e-2f, 1);
        double before = d.accuracy();
        float first = d.trainStep(16);
        float last = first;
        for (int s = 0; s < 120; s++) {
            last = d.trainStep(16);
        }
        double after = d.accuracy();
        System.out.printf("diffractive: loss %.3f -> %.3f, accuracy %.2f -> %.2f%n", first, last, before, after);
        assertTrue(after > 0.9, "accuracy " + after);
    }
}
