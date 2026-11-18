package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.Augmentation;

import org.junit.jupiter.api.Test;

class GenerativeAugmentationTest {

    @Test
    void baselinesAtFiveImagesPerClass() {
        double real = SyntheticTraining.augmentation(Augmentation.REAL, 5, 1f, 0, 2, 1).accuracy();
        double mixup = SyntheticTraining.augmentation(Augmentation.SHIFT_MIXUP, 5, 1f, 0, 2, 1).accuracy();
        double vae = SyntheticTraining.augmentation(Augmentation.VAE, 5, 1f, 0, 2, 1).accuracy();
        System.out.printf("augmentation 5 img/class: real %.3f, shift+mixup %.3f, real+VAE %.3f%n", real, mixup, vae);
        assertTrue(real > 0.12 && vae > 0.12, "above chance (0.1)");
        assertTrue(mixup > real, "shift + mixup helps a shift-variant MLP");
        assertTrue(Math.abs(vae - real) < 0.1);
    }
}
