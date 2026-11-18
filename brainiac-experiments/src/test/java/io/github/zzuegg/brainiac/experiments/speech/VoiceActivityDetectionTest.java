package io.github.zzuegg.brainiac.experiments.speech;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SpeakerVoices;
import io.github.zzuegg.brainiac.experiments.speech.SpeechFrontEnds.Detector;
import io.github.zzuegg.brainiac.experiments.speech.SpeechFrontEnds.VadRow;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class VoiceActivityDetectionTest {

    @Test
    void statisticalAndNeuralDetectors() {
        Map<String, List<VadRow>> r = SpeechFrontEnds.vad(new SpeakerVoices.Noise[] {SpeakerVoices.Noise.WHITE,
            SpeakerVoices.Noise.BABBLE}, new double[] {10, 0}, 3, 120, 1, Detector.ENERGY, Detector.SOHN,
                Detector.SOHN_HMM, Detector.CRNN);
        System.out.printf("VAD (F1/AUC), 120 CRNN steps: %s%n", r);
        List<VadRow> white10 = r.get("WHITE@10");
        assertTrue(white10.get(1).auc() > 0.7, "Sohn's ratio separates speech at 10 dB: " + white10);
    }
}
