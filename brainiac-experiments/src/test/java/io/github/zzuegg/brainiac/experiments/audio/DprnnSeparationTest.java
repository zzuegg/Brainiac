package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** The 8.2.18 DPRNN row of {@link SourceSeparation}. */
class DprnnSeparationTest {

    @Test
    void dprnnLearns() {
        SourceSeparation.Score s = SourceSeparation.run(30, 1, SourceSeparation.Method.DPRNN)[0];
        System.out.println("separation: " + s);
        assertTrue(s.improvement0() > 0.3, s.toString());
    }
}
