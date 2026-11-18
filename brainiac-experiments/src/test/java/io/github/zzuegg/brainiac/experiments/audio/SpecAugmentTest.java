package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Encoder;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Features;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.StringScore;

import org.junit.jupiter.api.Test;

class SpecAugmentTest {

    @Test
    void specAugmentRegularisesASmallTrainingSet() {
        StringScore plain = SpokenDigitRecognition.strings(Encoder.MLP, Features.BANDS, false, 20, 250, 48, 1);
        StringScore augmented = SpokenDigitRecognition.strings(Encoder.MLP, Features.BANDS, true, 20, 250, 48, 1);
        System.out.printf("48 training strings, 250 CTC steps: %s against %s%n", augmented, plain);
        assertTrue(augmented.digitErrorRate() < 0.3, augmented.toString());
    }
}
