package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DetrLearningCurveTest {

    @Test
    void plainDetrLearnsSlowly() {
        double[] curve = DetectionShootout.detr(false, new int[] {300, 600, 900}, 1);
        System.out.printf("DETR-lite mAP@0.5 at 300/600/900 steps: %.3f/%.3f/%.3f%n", curve[0], curve[1], curve[2]);
        assertTrue(curve[2] > curve[0], "set prediction improves with training");
    }
}
