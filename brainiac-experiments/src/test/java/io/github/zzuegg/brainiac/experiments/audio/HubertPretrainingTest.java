package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.MaskedSpeechPretraining;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.Encoder;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.PretrainScore;

import org.junit.jupiter.api.Test;

class HubertPretrainingTest {

    @Test
    void hubertPretrainingThenFewLabelFineTuning() {
        PretrainScore s = SpokenDigitRecognition.pretraining(Encoder.TRANSFORMER, MaskedSpeechPretraining.Mode.HUBERT,
                200, 30, 120, 1);
        System.out.printf("30 labelled strings, 120 CTC steps: DER after HuBERT pretraining %.3f, from scratch %.3f%n",
                s.pretrainedDer(), s.scratchDer());
        assertTrue(s.pretrainedDer() < 0.9, "fine-tuning still learns: " + s.pretrainedDer());
    }
}
