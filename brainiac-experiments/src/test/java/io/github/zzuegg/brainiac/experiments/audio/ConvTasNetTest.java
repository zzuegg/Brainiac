package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.SourceSeparation.Method;
import io.github.zzuegg.brainiac.experiments.audio.SourceSeparation.Score;

import org.junit.jupiter.api.Test;

class ConvTasNetTest {

    @Test
    void convTasNetSeparatesTwoSpeakersWithUpit() {
        Score s = SourceSeparation.run(150, 1, Method.CONV_TASNET)[0];
        System.out.printf("two-speaker digits, 150 steps: %s (STFT-mask MLP 4.8/5.8 dB, IRM 22.0/20.9 dB in "
                + "SourceSeparationTest)%n", s);
        assertTrue(s.improvement0() > 1, s.toString());
    }
}
