package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** The 8.2.16 PCEN row of {@link SpokenDigitRecognition#frontEndDer}. */
class PcenFrontEndTest {

    @Test
    void pcenBeatsLogMelInNoise() {
        double logMel = SpokenDigitRecognition.frontEndDer(SpokenDigitRecognition.FrontEnd.LOG_MEL, 5, 150, 1);
        double pcen = SpokenDigitRecognition.frontEndDer(SpokenDigitRecognition.FrontEnd.MEL_PCEN, 5, 150, 1);
        System.out.printf("front ends at 5 dB: log-mel DER %.3f, mel + PCEN DER %.3f%n", logMel, pcen);
        assertTrue(pcen < logMel, pcen + " vs " + logMel);
    }
}
