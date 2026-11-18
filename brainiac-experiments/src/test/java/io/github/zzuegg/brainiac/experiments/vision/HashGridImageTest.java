package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class HashGridImageTest {

    @Test
    void hashGridFitsFasterThanFourierFeatures() {
        SceneFields.FitScore raw = SceneFields.fitImage(SceneFields.ImageNet.RAW, 0, 48, 150, 1);
        SceneFields.FitScore fourier = SceneFields.fitImage(SceneFields.ImageNet.FOURIER, 0, 48, 150, 1);
        SceneFields.FitScore hash = SceneFields.fitImage(SceneFields.ImageNet.HASH_GRID, 12, 48, 150, 1);
        System.out.printf("48x48 picture after 150 steps, PSNR (s, params): hash grid %.2f dB (%.1f s, %d),"
                + " Fourier %.2f dB (%.1f s, %d), raw MLP %.2f dB (%.1f s, %d)%n", hash.psnr(), hash.seconds(),
                hash.parameters(), fourier.psnr(), fourier.seconds(), fourier.parameters(), raw.psnr(),
                raw.seconds(), raw.parameters());
        assertTrue(hash.psnr() > fourier.psnr() + 5, "the hash grid reaches a higher PSNR in the same steps");
        assertTrue(fourier.psnr() > raw.psnr(), "Fourier features beat raw coordinates");
    }
}
