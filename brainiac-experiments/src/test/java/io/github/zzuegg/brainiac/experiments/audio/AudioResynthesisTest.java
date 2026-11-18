package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.audio.AudioResynthesis.Score;

import org.junit.jupiter.api.Test;

class AudioResynthesisTest {

    @Test
    void ddspResynthesisesAndTransfersTimbre() {
        Score[] s = AudioResynthesis.run(60, 1);
        System.out.printf("melodies, 60 steps: %s; baseline %s; B's own audio against A playing its notes: %.3f%n",
                s[0], s[1], s[2].transferDistance());
        assertTrue(s[0].spectralDistance() < s[1].spectralDistance(), "DDSP beats the sample-level decoder");
        assertTrue(s[0].f0ErrorCents() < 200, "DDSP keeps the pitch: " + s[0].f0ErrorCents());
        assertTrue(s[0].transferDistance() < s[2].transferDistance(), "timbre transfer moves B towards A");
    }
}
