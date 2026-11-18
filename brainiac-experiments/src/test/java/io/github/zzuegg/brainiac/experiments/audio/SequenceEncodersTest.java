package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Encoder;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Features;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.StringScore;

import org.junit.jupiter.api.Test;

class SequenceEncodersTest {

    @Test
    void transformerAndBiGruAgainstTheFrameStackedMlp() {
        StringScore transformer = SpokenDigitRecognition.strings(Encoder.TRANSFORMER, Features.BANDS, false, 20, 200,
                1);
        StringScore gru = SpokenDigitRecognition.strings(Encoder.BIGRU, Features.BANDS, false, 20, 200, 1);
        StringScore mlp = SpokenDigitRecognition.strings(Encoder.MLP, Features.BANDS, false, 20, 200, 1);
        System.out.printf("digit strings, 200 CTC steps at 20 dB: %s; %s; baseline %s%n", transformer, gru, mlp);
        assertTrue(transformer.digitErrorRate() < 0.4, transformer.toString());
        assertTrue(mlp.digitErrorRate() < 0.15, mlp.toString());
    }
}
