package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DeepImagePriorTest {

    @Test
    void deepImagePrior() {
        Map<String, double[]> r = ImageInverseProblems.deepImagePrior(32, 1, 300, 25 / 255.0, 0.5, 7);
        System.out.printf("13.4.6 deep image prior 32x32, sigma 25/255 and 50%% pixels {denoise PSNR, inpaint PSNR,"
                + " stop iteration}: %s%n", TomographyTest.line(r, "%.2f"));
        assertTrue(r.get("DIP oracle stop")[0] > r.get("noisy / observed")[0] + 3, TomographyTest.line(r));
        assertTrue(r.get("DIP oracle stop")[0] >= r.get("DIP fixed iterations")[0], TomographyTest.line(r));
        assertTrue(r.get("DIP ES-WMV")[0] > r.get("noisy / observed")[0], TomographyTest.line(r));
        assertTrue(r.get("DIP oracle stop")[1] > r.get("noisy / observed")[1] + 3, TomographyTest.line(r));
    }
}
