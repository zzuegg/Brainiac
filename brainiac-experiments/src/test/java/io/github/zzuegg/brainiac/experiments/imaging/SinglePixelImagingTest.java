package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SinglePixelImagingTest {

    @Test
    void singlePixel() {
        Map<String, double[]> r = ImageInverseProblems.singlePixel(32, 256, 3, new double[] {0.02, 0.05, 0.10, 0.25},
                0.05, 0.01, 200, 3);
        System.out.printf("13.4.12 single-pixel 32x32 shapes, noise 0.01 {PSNR at 2/5/10/25%%, SSIM at 2%%}: %s%n",
                TomographyTest.line(r, "%.2f"));
        double[] seq = r.get("Hadamard, sequency order");
        assertTrue(seq[3] > seq[0], TomographyTest.line(r));
        assertTrue(seq[2] > r.get("random +-1, FISTA on Haar")[2], TomographyTest.line(r));
        assertTrue(r.get("random +-1, TV (PDHG)")[3] > r.get("random +-1, FISTA on Haar")[0], TomographyTest.line(r));
        assertTrue(r.get("learned binary patterns + decoder")[0] > r.get("random +-1, FISTA on Haar")[0],
                TomographyTest.line(r));
    }
}
