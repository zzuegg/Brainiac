package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SegmentationLossTest {

    @Test
    void lovaszOptimisesIntersectionOverUnion() {
        DensePrediction.SegScore ce = DensePrediction.segment(DensePrediction.Segmenter.DEEPLAB,
                DensePrediction.SegLoss.CROSS_ENTROPY, false, 120, 1);
        DensePrediction.SegScore lovasz = DensePrediction.segment(DensePrediction.Segmenter.DEEPLAB,
                DensePrediction.SegLoss.LOVASZ, false, 120, 1);
        System.out.printf("DeepLab-lite (ASPP) mIoU/pixel accuracy: cross-entropy %.3f/%.3f, Lovasz %.3f/%.3f%n",
                ce.miou(), ce.pixelAccuracy(), lovasz.miou(), lovasz.pixelAccuracy());
        assertTrue(lovasz.miou() > ce.miou(), "the Lovasz loss lifts mean IoU");
    }
}
