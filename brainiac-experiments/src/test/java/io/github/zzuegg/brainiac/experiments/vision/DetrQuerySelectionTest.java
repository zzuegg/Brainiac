package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.SetPredictionDetector.Options;

import org.junit.jupiter.api.Test;

/** DETR-lite with two-stage query selection on the shape scenes, where plain DETR stays at 0. */
class DetrQuerySelectionTest {

    @Test
    void querySelectionLeavesMapZeroOnScenes() {
        int[] steps = {250, 500};
        double[] detr = DetectionShootout.detrCurve(Options.none(), false, steps, 1);
        double[] qs = DetectionShootout.detrCurve(Options.none().withQuerySelection(true), false, steps, 1);
        System.out.printf("Scenes mAP@0.5 at 250/500 steps: DETR %.3f/%.3f, DETR + query selection %.3f/%.3f%n",
                detr[0], detr[1], qs[0], qs[1]);
        assertTrue(qs[1] > detr[1], "query selection converges faster than learned queries");
    }
}
