package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** The 8.2.17 audio row: {@link SpokenDigitRecognition#scatteringWords}. */
class ScatteringWordsTest {

    @Test
    void scatteringAtTwoShots() {
        Map<String, Double> r = SpokenDigitRecognition.scatteringWords(2, 10, 1);
        r.forEach((k, v) -> System.out.printf("spoken words 2 shots %s: acc %.3f%n", k, v));
        assertTrue(r.get("scattering + ridge") > r.get("MFCC + ridge"), r.toString());
        assertTrue(r.get("scattering + ridge") > 0.5, r.toString());
    }
}
