package io.github.zzuegg.brainiac.experiments.music;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class RhythmTest {

    @Test
    void onsetsAndBeats() {
        List<RhythmAndHarmony.Row> rows = RhythmAndHarmony.rhythm(10, 3, 100, 21);
        String cond = "";
        StringBuilder line = new StringBuilder();
        for (RhythmAndHarmony.Row r : rows) {
            if (!r.condition().equals(cond)) {
                if (!line.isEmpty()) {
                    System.out.println(line);
                }
                cond = r.condition();
                line = new StringBuilder("11.5.3 " + cond + " F1 (CMLt):");
            }
            line.append(String.format(" %s %.2f", r.method(), r.f1()));
            if (!Double.isNaN(r.second())) {
                line.append(String.format(" (%.2f)", r.second()));
            }
            line.append(';');
        }
        System.out.println(line);
        double flux = rows.stream().filter(r -> r.condition().equals("steady no drums")
                && r.method().equals("onset SPECTRAL_FLUX")).findFirst().orElseThrow().f1();
        assertTrue(flux > 0.3, "spectral flux onset F1 " + flux);
        for (RhythmAndHarmony.Row r : rows) {
            assertTrue(r.f1() >= 0 && r.f1() <= 1, r.toString());
        }
    }
}
