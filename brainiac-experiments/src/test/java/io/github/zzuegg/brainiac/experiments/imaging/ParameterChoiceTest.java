package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ParameterChoiceTest {

    @Test
    void parameterChoice() {
        Map<String, double[]> r = ImageInverseProblems.parameterChoice(32, 2, 0.08, 0.01, 11);
        System.out.printf("13.4.11 parameter choice 32x32 {PSNR, loss to oracle dB, lambda/oracle}: %s%n",
                TomographyTest.line(r, "%.2f"));
        for (String task : new String[] {"Tikhonov deblur", "TV denoise", "wavelet threshold scale"}) {
            assertTrue(r.get(task + " / MC_SURE")[1] < 1.0, task + ": " + TomographyTest.line(r));
        }
        assertTrue(r.get("Tikhonov deblur / GCV")[1] < 1.0, TomographyTest.line(r));
    }
}
