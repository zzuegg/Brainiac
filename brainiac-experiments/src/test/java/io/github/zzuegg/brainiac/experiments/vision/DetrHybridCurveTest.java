package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.SetPredictionDetector.Options;

import org.junit.jupiter.api.Test;

/**
 * DETR versus H-DETR with anchor queries and auxiliary losses on document layouts. On the
 * shape scenes both stay at mAP 0 for 700 steps at this size, so only layouts are compared.
 */
class DetrHybridCurveTest {

    @Test
    void hybridMatchingAndAnchorsOnLayouts() {
        int[] steps = {250, 500};
        double[] detr = DetectionShootout.detrCurve(Options.none(), true, steps, 1);
        double[] hybrid = DetectionShootout.detrCurve(new Options(16, 3, true, true), true, steps, 1);
        System.out.printf("Layouts mAP@0.5 at 250/500 steps: DETR %.3f/%.3f, H-DETR + anchors + aux %.3f/%.3f%n",
                detr[0], detr[1], hybrid[0], hybrid[1]);
        assertTrue(detr[1] > 0 && hybrid[1] > 0, "both detectors find layout regions");
        assertTrue(hybrid[1] >= 0 && hybrid[1] <= 1, "mAP is a share");
    }
}
