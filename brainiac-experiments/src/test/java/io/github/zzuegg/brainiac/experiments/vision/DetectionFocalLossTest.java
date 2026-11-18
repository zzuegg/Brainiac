package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.AnchorDetector;
import org.junit.jupiter.api.Test;

class DetectionFocalLossTest {

    @Test
    void focalLossBeatsCrossEntropyForDenseAnchors() {
        int classes = DetectionShootout.CLASSES;
        DetectionShootout.Score focal = DetectionShootout.anchor(AnchorDetector.Config.retinaNet(classes), 250, 1);
        DetectionShootout.Score ce = DetectionShootout.anchor(new AnchorDetector.Config(classes, true, false, true, 16),
                250, 1);
        System.out.printf("RetinaNet-lite mAP@0.5/mAP@[.5:.95]: focal %.3f/%.3f, cross-entropy %.3f/%.3f (%d MACs)%n",
                focal.map50(), focal.mapCoco(), ce.map50(), ce.mapCoco(), focal.macs());
        assertTrue(focal.map50() > ce.map50(), "focal loss detects better than cross-entropy");
        assertTrue(focal.map50() > 0.08, "the anchor detector learns to detect");
    }
}
