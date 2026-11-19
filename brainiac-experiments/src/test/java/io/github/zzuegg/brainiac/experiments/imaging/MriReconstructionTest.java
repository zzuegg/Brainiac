package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class MriReconstructionTest {

    @Test
    void compressedSensing() {
        Map<String, double[]> m = MriReconstruction.classical(32, new int[] {4, 8}, 4, 0.01, 3);
        System.out.printf("13.4.2 MRI 32x32 {PSNR at 4x/8x, SSIM at 8x}: %s%n", TomographyTest.line(m));
        assertTrue(m.get("CS-TV")[0] > m.get("zero filled")[0] + 2, TomographyTest.line(m));
        assertTrue(m.get("CS-TV")[1] > m.get("zero filled")[1] + 2, TomographyTest.line(m));
    }

    @Test
    void unrolledNetworks() {
        Map<String, double[]> m = MriReconstruction.learned(16, 4, 64, 6, 120, 0.01, 5);
        System.out.printf("13.4.2 learned MRI 16x16 4x {PSNR train mask, PSNR new mask, params}: %s%n",
                TomographyTest.line(m));
        assertTrue(m.get("VarNet")[0] > m.get("zero filled")[0] + 1, TomographyTest.line(m));
        assertTrue(m.get("MoDL")[1] > m.get("zero filled")[1], TomographyTest.line(m));
    }
}
