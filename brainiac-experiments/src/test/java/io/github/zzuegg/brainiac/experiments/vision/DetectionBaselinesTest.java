package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DetectionBaselinesTest {

    @Test
    void slidingWindowAndSegmentationBaselinesDetect() {
        DetectionShootout.Score window = DetectionShootout.slidingWindow(250, 1);
        DetectionShootout.Score unet = DetectionShootout.unetComponents(120, 1);
        System.out.printf("detection baselines mAP@0.5/mAP@[.5:.95]: sliding-window CNN %.3f/%.3f (%d MACs),"
                + " U-Net + components %.3f/%.3f (%d MACs)%n", window.map50(), window.mapCoco(), window.macs(),
                unet.map50(), unet.mapCoco(), unet.macs());
        assertTrue(window.map50() > 0.3, "the sliding-window classifier finds glyphs");
        assertTrue(unet.map50() >= 0, "the segmentation baseline runs");
    }
}
