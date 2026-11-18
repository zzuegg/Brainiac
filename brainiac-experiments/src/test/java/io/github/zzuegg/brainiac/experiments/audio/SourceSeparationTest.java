package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.SourceSeparation.Method;
import io.github.zzuegg.brainiac.experiments.audio.SourceSeparation.Score;

import org.junit.jupiter.api.Test;

class SourceSeparationTest {

    @Test
    void stftMaskWithPitAgainstIcaAndTheOracle() {
        Score[] s = SourceSeparation.run(150, 1, Method.STFT_MASK, Method.FASTICA, Method.IDEAL_RATIO_MASK);
        System.out.printf("two-speaker digits, 150 steps: %s; %s; %s%n", s[0], s[1], s[2]);
        assertTrue(s[0].improvement0() > 2, "the mask MLP separates: " + s[0]);
        assertTrue(s[2].improvement0() > s[0].improvement0(), "the oracle mask bounds it");
    }
}
