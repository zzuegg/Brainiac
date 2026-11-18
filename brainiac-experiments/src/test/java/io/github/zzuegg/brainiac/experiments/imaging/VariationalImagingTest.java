package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class VariationalImagingTest {

    @Test
    void primalDualAndFilters() {
        Map<String, double[]> g = ImageInverseProblems.variational(64, 3, 0.1, 3);
        System.out.printf("13.4.9 Gaussian sigma 0.1, 64x64 {PSNR, SSIM, ms}: %s%n", TomographyTest.line(g));
        Map<String, double[]> sp = ImageInverseProblems.impulse(64, 3, 0.1, 4);
        System.out.printf("13.4.9 salt-and-pepper 10%% {PSNR, SSIM, ms}: %s%n", TomographyTest.line(sp));
        Map<String, double[]> ramps = ImageInverseProblems.ramps(64, 3, 0.05, 5);
        System.out.printf("13.4.9 ramps sigma 0.05 {PSNR, SSIM, ms}: %s%n", TomographyTest.line(ramps));
        double[] speed = ImageInverseProblems.tvSpeed(64, 0.1, 1e-3, 6);
        System.out.printf("13.4.9 TV-L2 to 0.1%% objective gap: PDHG %.0f it (%.1f ms), gradient descent %.0f it (%.1f ms,"
                + " best gap %.4f)%n", speed[0], speed[1], speed[2], speed[3], speed[4]);
        assertTrue(g.get("TV-L2 (PDHG)")[0] > g.get("noisy")[0] + 5, TomographyTest.line(g));
        assertTrue(sp.get("TV-L1 (PDHG)")[0] > sp.get("TV-L2 (PDHG)")[0], TomographyTest.line(sp));
        assertTrue(ramps.get("TGV (PDHG)")[0] > ramps.get("TV-L2 (PDHG)")[0], TomographyTest.line(ramps));
        assertTrue(speed[0] < speed[2], "PDHG " + speed[0] + " vs GD " + speed[2]);
    }
}
