package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SirenImageTest {

    @Test
    void sirenFitsThePicture() {
        SceneFields.FitScore raw = SceneFields.fitImage(SceneFields.ImageNet.RAW, 0, 48, 150, 1);
        SceneFields.FitScore siren = SceneFields.fitImage(SceneFields.ImageNet.SIREN, 0, 48, 150, 1);
        System.out.printf("48x48 picture after 150 steps, PSNR (params): SIREN %.2f dB (%d), raw ReLU MLP %.2f dB"
                + " (%d)%n", siren.psnr(), siren.parameters(), raw.psnr(), raw.parameters());
        assertTrue(siren.psnr() > raw.psnr() + 3, "sine activations fit fine detail that ReLU on raw xy cannot");
    }
}
