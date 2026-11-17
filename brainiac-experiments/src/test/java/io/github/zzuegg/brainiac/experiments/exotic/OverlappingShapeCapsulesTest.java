package io.github.zzuegg.brainiac.experiments.exotic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class OverlappingShapeCapsulesTest {
    @Test
    void capsulesRecognizeSingleAndOverlaidShapes() {
        OverlappingShapeCapsules c = OverlappingShapeCapsules.of(3, 1e-2f, 5);
        float first = c.trainStep(64);
        float last = first;
        for (int s = 0; s < 250; s++) {
            last = c.trainStep(64);
        }
        double single = c.accuracy(false, 200, 99);
        double pairs = c.accuracy(true, 200, 98);
        System.out.printf("capsules: loss %.3f -> %.3f, single %.2f, pairs %.2f%n", first, last, single, pairs);
        assertTrue(last < 0.5f * first);
        assertTrue(single > 0.9, "single " + single);
        assertTrue(pairs > 0.7, "pairs " + pairs);
    }
}
