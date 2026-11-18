package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Encoder;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Features;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.StringScore;

import org.junit.jupiter.api.Test;

class MambaEncoderTest {

    @Test
    void bidirectionalMambaLearnsDigitStringsWithCtc() {
        StringScore mamba = SpokenDigitRecognition.strings(Encoder.MAMBA, Features.BANDS, false, 20, 150, 1);
        StringScore mlp = SpokenDigitRecognition.strings(Encoder.MLP, Features.BANDS, false, 20, 150, 1);
        System.out.printf("digit strings, 150 CTC steps at 20 dB: %s; baseline %s%n", mamba, mlp);
        assertTrue(mamba.digitErrorRate() < 0.4, mamba.toString());
    }
}
