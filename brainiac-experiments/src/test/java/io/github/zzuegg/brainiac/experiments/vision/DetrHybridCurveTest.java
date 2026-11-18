package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.SetPredictionDetector.Options;

import org.junit.jupiter.api.Test;

class DetrHybridCurveTest {

    @Test
    void hybridMatchingAndAnchorsOnScenesAndLayouts() {
        int[] steps = {150, 300};
        Options[] options = {Options.none(), new Options(16, 3, true, true)};
        String[] names = {"DETR", "H-DETR + anchors + aux"};
        for (boolean layouts : new boolean[] {false, true}) {
            for (int i = 0; i < options.length; i++) {
                double[] c = DetectionShootout.detrCurve(options[i], layouts, steps, 1);
                System.out.printf("%s on %s mAP@0.5 at 150/300 steps: %.3f/%.3f%n", names[i],
                        layouts ? "layouts" : "scenes", c[0], c[1]);
                assertTrue(c[1] >= 0 && c[1] <= 1, "mAP is a share");
            }
        }
    }
}
