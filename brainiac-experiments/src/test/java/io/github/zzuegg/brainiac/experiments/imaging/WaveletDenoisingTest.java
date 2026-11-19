package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class WaveletDenoisingTest {

    @Test
    void waveletShrinkage() {
        Map<String, double[]> img = ImageInverseProblems.wavelets(64, 2, new double[] {10 / 255.0, 25 / 255.0,
            50 / 255.0}, 3);
        System.out.printf("13.4.10 wavelet denoising 64x64 {PSNR at sigma 10/25/50}: %s%n", TomographyTest.line(img));
        Map<String, double[]> sig = ImageInverseProblems.waveletSignals(1024, 1 / 7.0, 3, 4);
        System.out.printf("13.4.10 Donoho-Johnstone n 1024 SNR 7 {MSE blocks/bumps/heavisine/doppler}: %s%n",
                TomographyTest.line(sig, "%.4f"));
        assertTrue(img.get("SURE_SHRINK (DB4, soft)")[1] > img.get("VISU_SHRINK (DB4, soft)")[1],
                TomographyTest.line(img));
        assertTrue(img.get("BAYES_SHRINK + cycle spinning 4x4")[1] > img.get("BAYES_SHRINK (DB4, soft)")[1],
                TomographyTest.line(img));
        for (int s = 0; s < 4; s++) {
            assertTrue(sig.get("SURE_SHRINK")[s] < 3 * sig.get("oracle threshold")[s], TomographyTest.line(sig));
        }
    }
}
