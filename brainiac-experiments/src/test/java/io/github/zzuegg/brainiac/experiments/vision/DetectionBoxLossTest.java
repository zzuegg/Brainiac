package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.AnchorDetector;
import org.junit.jupiter.api.Test;

class DetectionBoxLossTest {

    @Test
    void giouAndSmoothL1BothTrainTheBoxHead() {
        int classes = DetectionShootout.CLASSES;
        DetectionShootout.Score giou = DetectionShootout.anchor(AnchorDetector.Config.retinaNet(classes), 250, 1);
        DetectionShootout.Score l1 = DetectionShootout.anchor(new AnchorDetector.Config(classes, true, true, false, 16),
                250, 1);
        System.out.printf("RetinaNet-lite mAP@0.5/mAP@[.5:.95]: GIoU %.3f/%.3f, smooth L1 %.3f/%.3f%n", giou.map50(),
                giou.mapCoco(), l1.map50(), l1.mapCoco());
        assertTrue(giou.map50() > 0.08 && l1.map50() > 0.05, "both box losses train a detector");
    }
}
