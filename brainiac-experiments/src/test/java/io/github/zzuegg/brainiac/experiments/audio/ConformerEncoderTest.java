package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Encoder;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Features;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.StringScore;

import org.junit.jupiter.api.Test;

class ConformerEncoderTest {

    @Test
    void conformerLearnsDigitStringsWithCtc() {
        StringScore conformer = SpokenDigitRecognition.strings(Encoder.CONFORMER, Features.BANDS, false, 20, 200, 1);
        StringScore mlp = SpokenDigitRecognition.strings(Encoder.MLP, Features.BANDS, false, 20, 200, 1);
        System.out.printf("digit strings, 200 CTC steps at 20 dB: %s; baseline %s%n", conformer, mlp);
        assertTrue(conformer.digitErrorRate() < 0.3, conformer.toString());
    }
}
