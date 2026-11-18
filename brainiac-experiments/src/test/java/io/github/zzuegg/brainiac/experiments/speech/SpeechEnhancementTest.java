package io.github.zzuegg.brainiac.experiments.speech;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SpeakerVoices;
import io.github.zzuegg.brainiac.experiments.speech.SpeechFrontEnds.EnhanceRow;
import io.github.zzuegg.brainiac.experiments.speech.SpeechFrontEnds.Enhancer;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SpeechEnhancementTest {

    @Test
    void classicalAndMaskEnhancers() {
        Map<Double, List<EnhanceRow>> r = SpeechFrontEnds.enhancement(SpeakerVoices.Noise.WHITE, new double[] {0},
                4, 30, 0, 1, Enhancer.NOISY, Enhancer.SPECTRAL_SUBTRACTION, Enhancer.MMSE_STSA,
                Enhancer.MLP_MASK, Enhancer.CRNN_COMPLEX, Enhancer.IDEAL_RATIO_MASK);
        System.out.printf("enhancement, white noise 0 dB, 30 steps: %s%n", r.get(0.0));
        List<EnhanceRow> rows = r.get(0.0);
        assertTrue(rows.get(2).siSnrImprovement() > 1, "MMSE-STSA helps in stationary noise: " + rows.get(2));
        assertTrue(rows.get(5).siSnrImprovement() > rows.get(2).siSnrImprovement(), "the oracle mask bounds it");
    }
}
