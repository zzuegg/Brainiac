package io.github.zzuegg.brainiac.experimental.speech;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SpeakerVoices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SpeechBlocksTest {

    @Test
    void equalErrorRateAndDcf() {
        double[] scores = {0.9, 0.8, 0.7, 0.3, 0.2, 0.75};
        boolean[] target = {true, true, true, false, false, false};
        double eer = SpeakerEmbedding.equalErrorRate(scores, target);
        System.out.printf("EER of a toy trial list: %.3f, minDCF %.3f%n", eer,
                SpeakerEmbedding.minDcf(scores, target, 0.05));
        assertTrue(eer > 0.2 && eer < 0.5, "one impostor outscores one target: " + eer);
        assertEquals(0.0, SpeakerEmbedding.equalErrorRate(new double[] {1, 0}, new boolean[] {true, false}), 1e-9);
    }

    @Test
    void pldaSeparatesSpeakersInAGaussianModel() {
        SplittableRandom r = new SplittableRandom(1);
        List<double[]> x = new ArrayList<>();
        int[] spk = new int[200];
        double[][] centres = new double[20][4];
        for (double[] c : centres) {
            for (int i = 0; i < 4; i++) {
                c[i] = 2 * r.nextGaussian();
            }
        }
        for (int n = 0; n < 200; n++) {
            spk[n] = n % 20;
            double[] v = new double[4];
            for (int i = 0; i < 4; i++) {
                v[i] = centres[spk[n]][i] + 0.5 * r.nextGaussian();
            }
            x.add(v);
        }
        SpeakerEmbedding.Plda plda = SpeakerEmbedding.Plda.fit(x, spk);
        assertTrue(plda.score(x.get(0), x.get(20)) > plda.score(x.get(0), x.get(1)), "same speaker scores higher");
    }

    @Test
    void lossesRunAndShapesHold() {
        SplittableRandom r = new SplittableRandom(2);
        SpeakerEmbedding net = new SpeakerEmbedding(8, 8, 6, true, r);
        Var e = net.forward(Var.of(Tensor.normal(Shape.of(6, 20, 8), 0, 1, r)));
        assertEquals(Shape.of(6, 6), e.shape());
        Var ge2e = SpeakerEmbedding.ge2e(e, 3, 2, Var.of(Tensor.scalar(10)), Var.of(Tensor.scalar(-5)));
        Var aam = SpeakerEmbedding.aamSoftmax(e, Var.of(Tensor.normal(Shape.of(6, 3), 0, 1, r)),
                new int[] {0, 1, 2, 0, 1, 2}, 16, 0.2f);
        ge2e.add(aam).backward();
        assertTrue(Float.isFinite(ge2e.value().item()) && Float.isFinite(aam.value().item()));
    }

    @Test
    void diarizationErrorRateMapsSpeakers() {
        boolean[][] ref = {{true, true, false, false}, {false, false, true, true}};
        boolean[][] swapped = {ref[1], ref[0]};
        assertEquals(0.0, Diarization.errorRate(swapped, ref)[0], 1e-9);
        boolean[][] one = {{true, true, true, true}};
        assertEquals(0.5, Diarization.errorRate(one, ref)[0], 1e-9);
    }

    @Test
    void voiceActivityOnACleanConversation() {
        SplittableRandom r = new SplittableRandom(3);
        SpeakerVoices.Conversation c = SpeakerVoices.conversation(SpeakerVoices.speakers(2, 4), 4, 0, SpeakerVoices.Noise.WHITE,
                20, r);
        double auc = VoiceActivity.auc(VoiceActivity.sohn(c.wave(), SpeakerVoices.FRAME), c.speech());
        System.out.printf("Sohn VAD AUC at 20 dB: %.3f%n", auc);
        assertTrue(auc > 0.8, "speech frames have a higher likelihood ratio: " + auc);
    }
}
