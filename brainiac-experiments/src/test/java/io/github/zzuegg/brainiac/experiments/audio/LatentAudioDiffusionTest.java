package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class LatentAudioDiffusionTest {

    @Test
    void latentAgainstMelDiffusionForSpokenDigits() {
        List<AudioGeneration.LatentRow> rows = AudioGeneration.latentGeneration(150, 200, 3, 12, 1);
        StringBuilder line = new StringBuilder("9.5.12 class-conditional digits, judge acc / MMD / NFE (ms):");
        for (AudioGeneration.LatentRow r : rows) {
            line.append(String.format(" %s %.2f/%.3f/%d (%.0f);", r.model(), r.classAccuracy(), r.mmd(),
                    r.evaluations(), r.millis()));
            assertTrue(Double.isFinite(r.mmd()), line.toString());
        }
        System.out.println(line);
    }
}
