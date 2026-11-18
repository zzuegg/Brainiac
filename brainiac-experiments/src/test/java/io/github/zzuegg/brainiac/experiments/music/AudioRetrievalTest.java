package io.github.zzuegg.brainiac.experiments.music;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class AudioRetrievalTest {

    @Test
    void fingerprints() {
        List<AudioRetrieval.Row> rows = AudioRetrieval.fingerprinting(30, 12, 80, 41);
        String cond = "";
        StringBuilder line = new StringBuilder();
        for (AudioRetrieval.Row r : rows) {
            if (!r.condition().equals(cond)) {
                if (!line.isEmpty()) {
                    System.out.println(line);
                }
                cond = r.condition();
                line = new StringBuilder("11.5.5 " + cond + " top-1 (ms/query):");
            }
            line.append(String.format(" %s %.2f (%.1f);", r.method(), r.score(), r.second()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).score() > 0.6, "landmarks at 20 dB " + rows.get(0).score());
    }

    @Test
    void soundEvents() {
        List<AudioRetrieval.Row> rows = AudioRetrieval.eventDetection(24, 8, 100, 43);
        StringBuilder line = new StringBuilder("11.5.6 sound events, event F1 / segment ER:");
        for (AudioRetrieval.Row r : rows) {
            line.append(String.format(" %s %.3f/%.3f;", r.method(), r.score(), r.second()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).score() > 0.1, line.toString());
    }
}
