package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.AnchorDetector;
import org.junit.jupiter.api.Test;

class DetectionPyramidTest {

    @Test
    void pyramidFindsSmallObjects() {
        int classes = DetectionShootout.CLASSES;
        DetectionShootout.Score fpn = DetectionShootout.anchor(AnchorDetector.Config.retinaNet(classes), 250, 1);
        DetectionShootout.Score flat = DetectionShootout.anchor(new AnchorDetector.Config(classes, false, true, true, 16),
                250, 1);
        System.out.printf("RetinaNet-lite mAP@0.5 all/small: pyramid %.3f/%.3f (%d MACs), coarsest map only %.3f/%.3f"
                + " (%d MACs)%n", fpn.map50(), fpn.mapSmall(), fpn.macs(), flat.map50(), flat.mapSmall(), flat.macs());
        assertTrue(fpn.mapSmall() > flat.mapSmall(), "the pyramid helps on small glyphs");
    }
}
