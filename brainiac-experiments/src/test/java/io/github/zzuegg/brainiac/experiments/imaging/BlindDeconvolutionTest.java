package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class BlindDeconvolutionTest {

    @Test
    void blindDeconvolution() {
        Map<String, double[]> m = ImageInverseProblems.blind(64, 6, 9, 0.01, 3);
        System.out.printf("13.4.4 blind deconvolution 64x64, 9x9 motion/defocus, sigma 0.01 {kernel similarity, PSNR,"
                + " share error ratio <= 3, median ratio}: %s%n", TomographyTest.line(m));
        assertTrue(m.get("L0 gradient blind")[0] > m.get("Gaussian guess (PnP-HQS TV)")[0], TomographyTest.line(m));
        assertTrue(m.get("L0 gradient blind")[3] < m.get("Gaussian guess (PnP-HQS TV)")[3], TomographyTest.line(m));
    }
}
