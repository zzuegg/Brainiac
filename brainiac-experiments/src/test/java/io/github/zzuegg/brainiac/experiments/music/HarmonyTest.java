package io.github.zzuegg.brainiac.experiments.music;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class HarmonyTest {

    @Test
    void chordRecognition() {
        List<RhythmAndHarmony.Row> rows = RhythmAndHarmony.harmony(16, 6, 200, 31);
        StringBuilder line = new StringBuilder("11.5.4 chords, unseen timbres, WCSR/segmentation:");
        for (RhythmAndHarmony.Row r : rows) {
            line.append(String.format(" %s %.3f/%.3f;", r.method(), r.f1(), r.second()));
        }
        System.out.println(line);
        assertTrue(rows.get(1).f1() > 0.3, line.toString());
        assertTrue(rows.get(1).f1() >= rows.get(0).f1() - 0.02, line.toString());
    }
}
