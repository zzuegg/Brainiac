package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PhaseRetrievalTest {

    @Test
    void fourierMagnitudes() {
        Map<String, double[]> m = PhaseRetrieval.fourier(16, 6, 600, 0, 3);
        System.out.printf("13.4.3 Fourier magnitudes 16x16 {success, PSNR}: %s%n", TomographyTest.line(m));
        assertTrue(m.get("HIO")[1] >= m.get("ER")[1], TomographyTest.line(m));
    }

    @Test
    void codedDiffraction() {
        Map<String, double[]> clean = PhaseRetrieval.coded(16, new int[] {2, 4}, 4, 600, 0, 5);
        System.out.printf("13.4.3 CDP 16x16 noiseless {success at 2/4 patterns, PSNR at 2/4}: %s%n",
                TomographyTest.line(clean));
        Map<String, double[]> noisy = PhaseRetrieval.coded(16, new int[] {4}, 4, 400, 300, 6);
        System.out.printf("13.4.3 CDP 16x16 Poisson 300 photons {success, PSNR} at 4 patterns: %s%n",
                TomographyTest.line(noisy));
        assertTrue(clean.get("WIRTINGER_FLOW")[1] > clean.get("RANDOM_START_GD")[1], TomographyTest.line(clean));
        assertTrue(clean.get("WIRTINGER_FLOW")[3] > 30, TomographyTest.line(clean));
        assertTrue(noisy.get("PR_DEEP")[1] > noisy.get("WIRTINGER_FLOW")[1], TomographyTest.line(noisy));
    }
}
