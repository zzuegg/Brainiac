package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DemosaicingTest {

    @Test
    void demosaicing() {
        Map<String, double[]> clean = ImageInverseProblems.demosaicing(32, 32, 6, 0, 100, 3);
        System.out.printf("13.4.5 demosaicing 32x32 stripes, sigma 0 {CPSNR, edge RGB error}: %s%n",
                TomographyTest.line(clean, "%.3f"));
        assertTrue(clean.get("MALVAR")[0] > clean.get("BILINEAR")[0] + 1, TomographyTest.line(clean));
        assertTrue(clean.get("BILINEAR")[0] > clean.get("NEAREST")[0], TomographyTest.line(clean));
        assertTrue(clean.get("LEARNED (hard mining)")[0] > clean.get("BILINEAR")[0], TomographyTest.line(clean));
    }
}
