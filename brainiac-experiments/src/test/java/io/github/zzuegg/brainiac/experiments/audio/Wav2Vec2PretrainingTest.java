package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.MaskedSpeechPretraining;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Encoder;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.PretrainScore;

import org.junit.jupiter.api.Test;

class Wav2Vec2PretrainingTest {

    @Test
    void wav2vec2PretrainingThenFewLabelFineTuning() {
        PretrainScore s = SpokenDigitRecognition.pretraining(Encoder.TRANSFORMER,
                MaskedSpeechPretraining.Mode.WAV2VEC2, 200, 30, 120, 1);
        System.out.printf("30 labelled strings, 120 CTC steps: DER after wav2vec 2.0 pretraining %.3f, from scratch "
                + "%.3f%n", s.pretrainedDer(), s.scratchDer());
        assertTrue(s.pretrainedDer() < 0.9, "fine-tuning still learns: " + s.pretrainedDer());
    }
}
