package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DiceSegmentationTest {

    @Test
    void diceLossTrainsTheSegmenter() {
        DensePrediction.SegScore dice = DensePrediction.segment(DensePrediction.Segmenter.DEEPLAB,
                DensePrediction.SegLoss.DICE, false, 120, 1);
        System.out.printf("DeepLab-lite with soft Dice: mIoU %.3f, pixel accuracy %.3f%n", dice.miou(),
                dice.pixelAccuracy());
        assertTrue(dice.pixelAccuracy() > 0.8, "Dice training labels most pixels");
    }
}
