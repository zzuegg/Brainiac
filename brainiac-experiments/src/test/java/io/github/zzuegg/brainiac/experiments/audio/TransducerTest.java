package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition.TransducerScore;

import org.junit.jupiter.api.Test;

class TransducerTest {

    @Test
    void transducerUsesLabelContextOnMarkovStrings() {
        TransducerScore s = SpokenDigitRecognition.transducer(5, 400, 1);
        System.out.printf("Markov digit strings at 5 dB, DER: RNN-T beam 4 %.3f, RNN-T greedy %.3f, CTC %.3f; "
                + "mean emission position RNN-T %.2f, CTC %.2f%n", s.rnntBeamDer(), s.rnntGreedyDer(), s.ctcDer(),
                s.rnntEmissionFrame(), s.ctcEmissionFrame());
        assertTrue(s.rnntBeamDer() < 0.15, "transducer beam DER " + s.rnntBeamDer());
        assertTrue(s.rnntBeamDer() <= s.rnntGreedyDer(), "beam search is no worse than greedy");
    }
}
