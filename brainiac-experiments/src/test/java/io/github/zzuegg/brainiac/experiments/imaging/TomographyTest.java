package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

class TomographyTest {

    static String line(Map<String, double[]> m) {
        return line(m, "%.2f");
    }

    static String line(Map<String, double[]> m, String format) {
        return m.entrySet().stream()
                .map(e -> e.getKey() + " " + Arrays.stream(e.getValue()).mapToObj(v -> String.format(format, v))
                        .collect(Collectors.joining("/")))
                .collect(Collectors.joining("; "));
    }

    @Test
    void sparseViewTomography() {
        int[] angles = {48, 16, 8};
        Map<String, double[]> m = Tomography.compare(32, angles, 3, 400, 13);
        System.out.printf("13.4.1 CT 32x32 {PSNR at 48/16/8 angles, SSIM at 8, ms/image at 8}: %s%n", line(m));
        double[] fbp = m.get("FBP (Ram-Lak)");
        assertTrue(fbp[0] > fbp[2] + 3, line(m));
        assertTrue(m.get("SART-TV")[2] > fbp[2] + 2, line(m));
        assertTrue(m.get("TV (preconditioned PDHG)")[2] > fbp[2] + 5, line(m));
    }

    @Test
    void learnedPrimalDual() {
        Map<String, double[]> m = Tomography.learned(16, 6, 4, 64, 120, 400, 21);
        System.out.printf("13.4.1 learned CT 16x16, 6 angles {PSNR, SSIM, ms/image, params}: %s%n", line(m));
        assertTrue(m.get("learned primal-dual")[0] > m.get("FBP (Shepp-Logan)")[0] + 1, line(m));
    }
}
